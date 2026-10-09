"""Exercise real local Unix sockets and WebSocket frames, including recovery."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unix_rpc import UnixRpc, RpcError
from supervisor import Supervisor
from test_supervisor import FakeRpc


def exact(sock, n):
    result = bytearray()
    while len(result) < n:
        data = sock.recv(n - len(result))
        if not data: raise EOFError
        result.extend(data)
    return bytes(result)


def receive(sock):
    a, b = exact(sock, 2)
    n = b & 127
    if n == 126: n = struct.unpack('!H', exact(sock, 2))[0]
    elif n == 127: n = struct.unpack('!Q', exact(sock, 8))[0]
    if not b & 128: raise AssertionError('client frames must be masked')
    mask = exact(sock, 4)
    data = exact(sock, n)
    return a & 15, bytes(x ^ mask[i % 4] for i, x in enumerate(data))


def frame(payload, opcode=1, final=True):
    n = len(payload)
    head = bytes([(128 if final else 0) | opcode])
    if n < 126: head += bytes([n])
    elif n < 65536: head += bytes([126]) + struct.pack('!H', n)
    else: head += bytes([127]) + struct.pack('!Q', n)
    return head + payload


class Server:
    def __init__(self, handler, fragmented=False, bad_handshake=False):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / 'rpc.sock')
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(self.path)
        self.listener.listen(1)
        self.handler = handler
        self.fragmented = fragmented
        self.bad_handshake = bad_handshake
        self.errors = []
        self.pongs = 0
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        try:
            with self.listener.accept()[0] as sock:
                sock.settimeout(3)
                header = bytearray()
                while not header.endswith(b'\r\n\r\n'): header.extend(exact(sock, 1))
                values = dict(line.split(': ', 1) for line in header.decode().split('\r\n')[1:] if ': ' in line)
                accept = base64.b64encode(hashlib.sha1((values['Sec-WebSocket-Key'] +
                    '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
                if self.bad_handshake: accept = 'wrong'
                sock.sendall(('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n'
                              'Connection: Upgrade\r\nSec-WebSocket-Accept: ' + accept + '\r\n\r\n').encode())
                if self.bad_handshake: return
                while True:
                    opcode, data = receive(sock)
                    if opcode == 10:
                        self.pongs += 1
                        continue
                    msg = json.loads(data)
                    if 'id' not in msg: continue
                    try:
                        result = {} if msg['method'] == 'initialize' else self.handler(msg['method'], msg.get('params', {}))
                        reply = {'id': msg['id'], 'result': result}
                    except RpcError as e:
                        reply = {'id': msg['id'], 'error': e.error}
                    encoded = json.dumps(reply).encode()
                    # Notifications are interleaved with responses on the real daemon.
                    sock.sendall(frame(b'{"method":"test/notification","params":{}}'))
                    if self.fragmented:
                        cut = len(encoded) // 2
                        sock.sendall(frame(encoded[:cut], final=False) + frame(b'ping', 9) +
                                     frame(encoded[cut:], opcode=0))
                    else: sock.sendall(frame(encoded))
        except (EOFError, ConnectionResetError, BrokenPipeError): pass
        except BaseException as e: self.errors.append(e)

    def __enter__(self): return self
    def __exit__(self, *_):
        self.worker.join(4)
        self.listener.close()
        self.temp.cleanup()
        if self.worker.is_alive(): raise AssertionError('mock server did not finish')
        if self.errors: raise self.errors[0]


class TransportTests(unittest.TestCase):
    def test_handshake_masking_lengths_fragmentation_and_ping(self):
        with Server(lambda m, p: p, fragmented=True) as server:
            with UnixRpc(server.path, timeout=2) as rpc:
                for n in [5, 500, 70000]:
                    payload = {'data': 'x' * n}
                    self.assertEqual(rpc.call('echo', payload), payload)
            self.assertGreaterEqual(server.pongs, 3)

    def test_invalid_upgrade_is_rejected(self):
        with Server(lambda m, p: {}, bad_handshake=True) as server:
            with self.assertRaisesRegex(RuntimeError, 'upgrade'):
                UnixRpc(server.path, timeout=2)

    def test_rpc_errors_are_reported(self):
        def handler(m, p): raise RpcError({'code': -32601, 'message': 'unsupported'})
        with Server(handler) as server:
            with UnixRpc(server.path, timeout=2) as rpc:
                with self.assertRaisesRegex(RpcError, 'unsupported'): rpc.call('missing')

    def test_end_to_end_quota_reset_then_single_resumption(self):
        fake = FakeRpc()
        fake.goal['status'] = 'usageLimited'
        fake.turn.update(status='failed', error={'codexErrorInfo': 'usageLimitExceeded'})
        fake.limits = {'ordinaryUsageAllowed': False, 'rateLimits': {
            'primary': {'usedPercent': 100, 'resetsAt': 50}}}
        cfg = {'repo': '/tmp/xca', 'branch': 'dev_cuda', 'thread_id': 'test',
               'objective': 'port', 'model': 'gpt-6.1-sol', 'resume_prompt': 'continue', 'idle_grace': 120}
        supervisor = Supervisor(cfg, {}, lambda s: None, lambda: 'dev_cuda')
        with Server(fake.call, fragmented=True) as server:
            with UnixRpc(server.path, timeout=2) as rpc, patch('builtins.print'):
                self.assertEqual(supervisor.tick(rpc, now=0)[0], 'waiting_quota')
                self.assertEqual(supervisor.tick(rpc, now=60)[0], 'waiting_quota')
                fake.limits['ordinaryUsageAllowed'] = True
                fake.limits['rateLimits']['primary']['usedPercent'] = 0
                self.assertEqual(supervisor.tick(rpc, now=70)[0], 'idle_grace')
                self.assertEqual(supervisor.tick(rpc, now=190)[0], 'resumed')
                self.assertEqual(supervisor.tick(rpc, now=220)[0], 'thread_active')
        self.assertEqual(len([m for m, _ in fake.calls if m == 'turn/start']), 1)


if __name__ == '__main__': unittest.main()

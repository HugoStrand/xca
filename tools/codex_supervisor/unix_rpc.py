"""Small JSON-RPC client for Codex's local WebSocket-over-Unix transport.

No credentials, external dependencies, TCP listener, or second app-server process.
RFC 6455 framing includes masking, fragmented messages, ping/pong and close.
"""

import base64
import hashlib
import json
import os
import socket
import struct
import time


class RpcError(RuntimeError):
    def __init__(self, error):
        super().__init__(error.get("message", "JSON-RPC error"))
        self.error = error


class UnixRpc:
    MAX_MESSAGE = 16 * 1024 * 1024

    def __init__(self, path, timeout=30):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.timeout = timeout
        self.buffer = bytearray()
        self.sequence = 0
        try:
            self.sock.connect(str(path))
            key = base64.b64encode(os.urandom(16)).decode()
            request = ("GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                       f"Sec-WebSocket-Key: {key}\r\n\r\n")
            self.sock.sendall(request.encode())
            while b"\r\n\r\n" not in self.buffer:
                self._receive()
                if len(self.buffer) > 65536:
                    raise RuntimeError("WebSocket upgrade headers too large")
            raw, remaining = bytes(self.buffer).split(b"\r\n\r\n", 1)
            self.buffer = bytearray(remaining)
            lines = raw.decode("ascii").split("\r\n")
            headers = {k.lower(): v.strip() for k, v in
                       (line.split(":", 1) for line in lines[1:])}
            expected = base64.b64encode(hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
            if (lines[0].split()[1] != "101"
                    or headers.get("sec-websocket-accept", "") != expected
                    or headers.get("upgrade", "").lower() != "websocket"
                    or "upgrade" not in headers.get("connection", "").lower()):
                raise RuntimeError("Invalid WebSocket upgrade")
            self.call("initialize", {
                "clientInfo": {"name": "xca_dense_cuda_supervisor", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            })
            self.send({"method": "initialized", "params": {}})
        except BaseException:
            self.sock.close()
            raise

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _receive(self):
        part = self.sock.recv(65536)
        if not part:
            raise ConnectionError("Codex daemon disconnected")
        self.buffer.extend(part)

    def _take(self, size):
        while len(self.buffer) < size:
            self._receive()
        value = bytes(self.buffer[:size])
        del self.buffer[:size]
        return value

    def _frame(self, opcode, payload):
        size = len(payload)
        header = bytes([0x80 | opcode])
        if size < 126:
            header += bytes([0x80 | size])
        elif size < 65536:
            header += bytes([0x80 | 126]) + struct.pack("!H", size)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", size)
        mask = os.urandom(4)
        self.sock.sendall(header + mask + bytes(x ^ mask[i % 4] for i, x in enumerate(payload)))

    def send(self, message):
        self._frame(1, json.dumps(message, separators=(",", ":")).encode())

    def read(self):
        message = bytearray()
        started = False
        while True:
            first, second = self._take(2)
            opcode, final = first & 15, bool(first & 128)
            if first & 0x70 or second & 128:
                raise RuntimeError("Unsupported server WebSocket frame")
            size = second & 127
            if size == 126:
                size = struct.unpack("!H", self._take(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self._take(8))[0]
            if size > self.MAX_MESSAGE or len(message) + size > self.MAX_MESSAGE:
                raise RuntimeError("WebSocket message too large")
            if opcode >= 8 and (size > 125 or not final):
                raise RuntimeError("Invalid WebSocket control frame")
            payload = self._take(size)
            if opcode == 8:
                raise ConnectionError("Codex daemon closed connection")
            if opcode == 9:
                self._frame(10, payload)
                continue
            if opcode == 10:
                continue
            if opcode == 1 and not started:
                started = True
            elif opcode != 0 or not started:
                raise RuntimeError("Invalid WebSocket message sequence")
            message.extend(payload)
            if final:
                return json.loads(message)

    def call(self, method, params=None):
        self.sequence += 1
        request_id = self.sequence
        self.send({"id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timeout waiting for {method}")
                self.sock.settimeout(remaining)
                message = self.read()
                if message.get("id") == request_id and "method" not in message:
                    if "error" in message:
                        raise RpcError(message["error"])
                    return message["result"]
                # This observer never grants approval or answers questions. The normal
                # clients and the configured automatic reviewer retain those controls.
                if "id" in message and "method" in message:
                    raise RuntimeError("Observer received a server request; use the normal client")
        finally:
            self.sock.settimeout(self.timeout)

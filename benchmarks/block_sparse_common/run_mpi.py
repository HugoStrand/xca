#!/usr/bin/env python3
"""Launch the MPI ownership benchmark using a configured build-tree environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--environment', required=True)
parser.add_argument('--launcher', default='mpirun')
parser.add_argument('--launcher-arg', action='append', default=[])
parser.add_argument('--worker-prefix', action='append', default=[], help='Repeat to prefix each worker, e.g. nsys profile arguments')
parser.add_argument('--python', help='Site launcher alias resolving to the configured interpreter')
parser.add_argument('--ranks', type=int, choices=range(1, 5), default=2)
parser.add_argument('--record', required=True)
parser.add_argument('--timeout', type=int, default=240)
parser.add_argument('worker_args', nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.timeout < 1 or args.timeout > 600:
    parser.error('Timeout must be1..600seconds')
settings = json.loads(Path(args.environment).read_text())
interpreter = args.python or settings['python']
if not os.path.samefile(interpreter, settings['python']):
    raise RuntimeError('Interpreter alias differs from configured Python')
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
env['PYTHONPATH'] = os.pathsep.join(settings['python_paths']+[env.get('PYTHONPATH', '')])
worker_args = args.worker_args[1:] if args.worker_args[:1] == ['--'] else args.worker_args
command = [args.launcher, *args.launcher_arg, '-n', str(args.ranks), *args.worker_prefix, interpreter,
           str(Path(__file__).with_name('mpi_ownership.py').resolve()), '--ranks', str(args.ranks), *worker_args]
record = Path(args.record)
with record.open('x') as f:
    json.dump(dict(command=command, environment_file=str(Path(args.environment).resolve()),
                   settings=settings, PYTHONPATH=env['PYTHONPATH'], timeout_s=args.timeout), f, indent=2)
    f.write('\n')
# A private process group permits bounded cleanup of launcher and its workers.
import signal
from datetime import datetime, timezone
started = datetime.now(timezone.utc).isoformat()
process = subprocess.Popen(command, env=env, start_new_session=True)
try:
    status = process.wait(timeout=args.timeout)
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    status = 124
with record.open() as f:
    result = json.load(f)
result.update(returncode=status, launcher_pid=process.pid, started_utc=started,
              finished_utc=datetime.now(timezone.utc).isoformat())
record.write_text(json.dumps(result, indent=2)+'\n')
raise SystemExit(status)

#!/usr/bin/env python3
"""Run a registered CTest Python case under MPI with its build-tree environment."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    from mpi4py import MPI

    expected = int(sys.argv[2])
    actual = MPI.COMM_WORLD.Get_size()
    if actual != expected:
        raise RuntimeError(f'Expected {expected} MPI ranks, got {actual}')
    print(f'Qualification MPI rank {MPI.COMM_WORLD.Get_rank()}/{actual}', flush=True)
    sys.argv = sys.argv[3:]
    runpy.run_path(sys.argv[0], run_name='__main__')
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctest-json', required=True)
    parser.add_argument('--test', default='Py_block_sparse_cuda')
    parser.add_argument('--record', required=True)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--launcher', default='mpprun')
    parser.add_argument('--launcher-arg', action='append', default=[])
    parser.add_argument('test_args', nargs='*')
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('--ranks must be positive')
    tests = json.loads(Path(args.ctest_json).read_text())['tests']
    test = next(t for t in tests if t['name'] == args.test)
    props = {p['name']: p['value'] for p in test['properties']}
    env = os.environ.copy()
    for entry in props.get('ENVIRONMENT', []):
        key, value = entry.split('=', 1)
        env[key] = value
    for entry in props.get('ENVIRONMENT_MODIFICATION', []):
        key, change = entry.split('=', 1)
        operation, value = change.split(':', 1)
        if operation == 'path_list_prepend':
            env[key] = value + (os.pathsep + env[key] if env.get(key) else '')
        elif operation == 'path_list_append':
            env[key] = (env[key] + os.pathsep if env.get(key) else '') + value
        elif operation == 'set':
            env[key] = value
        elif operation == 'unset':
            env.pop(key, None)
        else:
            raise RuntimeError(f'Unsupported CTest environment modification: {operation}')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [args.launcher, *args.launcher_arg, test['command'][0],
               str(Path(__file__).resolve()), '--worker', str(args.ranks),
               *test['command'][1:], *args.test_args]
    Path(args.record).write_text(json.dumps({
        'command': command, 'working_directory': props.get('WORKING_DIRECTORY'),
        'expected_ranks': args.ranks,
        'ctest_environment': props.get('ENVIRONMENT', []),
        'ctest_environment_modification': props.get('ENVIRONMENT_MODIFICATION', []),
        'resolved_PYTHONPATH': env.get('PYTHONPATH'),
    }, indent=2) + '\n')
    raise SystemExit(subprocess.run(command, cwd=props.get('WORKING_DIRECTORY'), env=env).returncode)

#!/usr/bin/env python3
"""Record isolated compact/cached composed DLR edges and explicit storage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='compact-01')
    args = parser.parse_args()
    for value in (args.machine, args.campaign):
        if not value or Path(value).name != value or value in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    binary = ROOT/'cbuild-cuda/benchmarks/experiments/block_sparse_cuda/compact_dlr/block_sparse_compact_dlr_benchmark'
    source = Path(__file__).parent
    files = [p for p in source.glob('*') if p.is_file()]
    files += [ROOT/'c++/triqs_xca/block_sparse/cuda'/n for n in ('dlr_ops.cpp', 'dlr_ops.hpp', 'device_ops.cpp', 'device_ops.hpp', 'memory.hpp')]
    manifest = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    host=platform.node(), gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
                    compiler=subprocess.check_output(['clang++-19', '--version'], text=True).splitlines()[0],
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    cases=[], thread_environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    folder = source/args.machine/args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **manifest['thread_environment'])
    for M, K, N, lam in ((1,1,1,20),(3,5,2,20),(16,16,16,20),(64,96,48,20),(256,256,256,20),(3,5,2,120)):
        name = f'{M}-{K}-{N}-lambda{lam}'
        command = [str(binary), '--m', str(M), '--k', str(K), '--n', str(N), '--lambda', str(lam), '--samples', '3']
        start = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        (folder/(name+'.csv')).write_text(result.stdout)
        if result.stderr: (folder/(name+'.log')).write_text(result.stderr)
        manifest['cases'].append(dict(name=name, command=command, returncode=result.returncode, elapsed_seconds=time.monotonic()-start))
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if result.returncode != 0: raise SystemExit(result.stderr)
        if len(list(csv.DictReader(result.stdout.splitlines()))) != 6: raise SystemExit('Unexpected row count')
        print(name, 'passed', flush=True)


if __name__ == '__main__': main()

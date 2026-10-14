#!/usr/bin/env python3
"""Record bounded actual-Fock mu slices; no atom_diag or full solver evaluation."""
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
    parser.add_argument('--build', type=Path, default=ROOT/'cbuild-cuda')
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='fock-slabs-01')
    args = parser.parse_args()
    for value in (args.machine, args.campaign):
        if not value or Path(value).name != value or value in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    binary = args.build.resolve()/'benchmarks/experiments/block_sparse_cuda/mapped_special/block_sparse_mapped_slab_benchmark'
    source = Path(__file__).parent
    files = [p for p in source.glob('*') if p.is_file()]
    files += [*ROOT.glob('c++/triqs_xca/block_sparse/cuda/**/*.cpp'),
              *ROOT.glob('c++/triqs_xca/block_sparse/cuda/**/*.hpp'),
              *ROOT.glob('c++/triqs_xca/block_sparse/cuda/**/*.cu')]
    manifest = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    host=platform.node(), cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                    gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
                    compiler=subprocess.check_output(['clang++-19', '--version'], text=True).splitlines()[0],
                    nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    thread_environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), cases=[])
    folder = source/args.machine/args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **manifest['thread_environment'])
    for modes, rank, rows in ((4, 3, 3), (10, 26, 32), (14, 26, 32), (14, 26, 128)):
        for direction in ('annihilation', 'creation'):
            name = f'n{modes}-r{rank}-rows{rows}-{direction}'
            command = [str(binary), '--modes', str(modes), '--rank', str(rank), '--rows', str(rows),
                       '--direction', direction, '--samples', '3']
            start = time.monotonic()
            case = dict(name=name, command=command)
            try:
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
                (folder/(name+'.csv')).write_text(run.stdout)
                if run.stderr:
                    (folder/(name+'.log')).write_text(run.stderr)
                case['returncode'] = run.returncode
            except subprocess.TimeoutExpired as error:
                for suffix, data in (('csv', error.stdout), ('log', error.stderr)):
                    if data:
                        (folder/(name+'.'+suffix)).write_bytes(data if isinstance(data, bytes) else data.encode())
                case['timeout_seconds'] = 120
            case['elapsed_seconds'] = time.monotonic()-start
            manifest['cases'].append(case)
            (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if case.get('returncode') != 0:
                raise SystemExit('Slab case failed; inspect retained raw output')
            data = list(csv.DictReader(run.stdout.splitlines()))
            if len(data) != 12:
                raise SystemExit('Unexpected sample count')
            print(name, 'passed', flush=True)


if __name__ == '__main__':
    main()

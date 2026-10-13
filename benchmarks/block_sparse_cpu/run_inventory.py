#!/usr/bin/env python3
"""Record bounded CPU path inventories and analytic large-sector estimates."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    files = [*root.glob('c++/triqs_xca/block_sparse/cuda/*.[ch]pp'),
             *root.glob('benchmarks/block_sparse_common/*.[ch]pp'), Path(__file__),
             Path(__file__).with_name('block_sparse_inventory.cpp')]
    files += [root / 'c++/triqs_xca' / name for name in ('backbone.cpp', 'backbone.hpp', 'hyb.cpp', 'block_sparse/block_op.cpp', 'block_sparse/evaluation_path.hpp')]
    manifest = {'schema': 'block-sparse-inventory-v1', 'host': platform.node(),
                'architecture': platform.machine(), 'platform': platform.platform(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'cpu_info': Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'cases': [], 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    manifest['triqs_root'] = env.get('TRIQS_ROOT')
    if manifest['triqs_root']:
        prefix = Path(manifest['triqs_root'])
        manifest['triqs_header_hashes'] = {str(p.relative_to(prefix)): hashlib.sha256(p.read_bytes()).hexdigest() for p in prefix.glob('include/triqs/hilbert_space/*.hpp')}
    try:
        manifest['compiler'] = subprocess.check_output(['clang++-19', '--version'], text=True).splitlines()[0]
    except FileNotFoundError:
        manifest['compiler'] = None
    try:
        manifest['gpu_info'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest['gpu_info'] = None
    rows = []
    cases = [(False, n, regime, dyn) for n in (4, 6, 8, 10)
             for regime in ('occupation', 'number', 'mixed') for dyn in (False, True)]
    cases += [(True, n, regime, False) for n in (10, 14) for regime in ('occupation', 'number', 'mixed')]
    expanded = []
    for analytic, modes, regime, dynamic in cases:
        if analytic:
            expanded.append((analytic,modes,regime,dynamic,'crossing',0))
        else:
            fields = modes + (2 if dynamic else 0)
            group = (2*fields)**2
            total = 8*group
            for topology in ('crossing','chain'):
                for first in (0,group,max(0,total-32)):
                    expanded.append((analytic,modes,regime,dynamic,topology,first))
    for analytic, modes, regime, dynamic, topology, first in expanded:
        name = f'{"analytic" if analytic else "actual"}-n{modes}-{regime}{"-dynamic" if dynamic else ""}'
        name += f'-{topology}-first{first}'
        command = [str(binary), '--modes', str(modes), '--partition', regime, '--count', '32', '--order', '3', '--topology',topology,'--first',str(first)]
        if analytic:
            command += ['--analytic', '--rank', '26']
        else:
            command += ['--shapes', str(output / (name + '-shapes.csv'))]
        if dynamic:
            command += ['--dynamic']
        started = time.monotonic()
        case = {'name': name, 'command': command}
        try:
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
            case['returncode'] = result.returncode
            (output / (name + '.log')).write_text(result.stderr)
            (output / (name + '.csv')).write_text(result.stdout)
            if result.returncode == 0:
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                if len(parsed) != 1:
                    raise ValueError('Expected exactly one inventory row')
                rows.extend(parsed)
        except subprocess.TimeoutExpired:
            case['timeout_seconds'] = 15
        case['elapsed_seconds'] = time.monotonic() - started
        manifest['cases'].append(case)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if rows:
        with (output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if any(case.get('returncode') != 0 for case in manifest['cases']):
        raise SystemExit('Inventory contains failed/timed-out cases; see manifest.json')


if __name__ == '__main__':
    main()

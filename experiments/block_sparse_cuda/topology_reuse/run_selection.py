#!/usr/bin/env python3
"""Record distributed bounded topology cohorts and nonzero index offsets."""
import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'cbuild-cuda')
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='selection-01')
    args = parser.parse_args()
    for name in (args.machine, args.campaign):
        if not name or Path(name).name != name or name in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    folder = Path(__file__).parent / args.machine / args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    binary = args.build.resolve() / 'benchmarks/experiments/block_sparse_cuda/topology_reuse/block_sparse_prefix_benchmark'
    spec = importlib.util.spec_from_file_location('xca_diag', ROOT / 'python/triqs_xca/diag.py')
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    manifest = dict(host=platform.node(), platform=platform.platform(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    triqs_root=os.environ.get('TRIQS_ROOT'),
                    gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                 '--format=csv,noheader'], text=True).strip(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), source_hashes={}, cohorts=[], cases=[])
    files = [Path(__file__), ROOT / 'python/triqs_xca/diag.py']
    files += list(Path(__file__).parent.glob('*.cpp')) + list(Path(__file__).parent.glob('*.inc'))
    files += [Path(__file__).with_name('CMakeLists.txt')]
    files += list((ROOT / 'c++/triqs_xca/block_sparse/cuda').glob('*.cpp'))
    for path in files:
        manifest['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for order in (3, 4, 6):
        start = time.monotonic()
        connected, scanned = [], 0
        for parity, pairing in itertools.islice(diag.all_pairings(order), 4096):
            scanned += 1
            if diag.is_connected(pairing):
                connected.append(pairing)
        count = min(6, len(connected))
        positions = [i * (len(connected) - 1) // (count - 1) for i in range(count)]
        selected = [connected[i] for i in positions]
        path = folder / f'o{order}.txt'
        path.write_text(f'{count} {order}\n' + ''.join(f'{a} {b}\n' for pairing in selected for a, b in pairing))
        manifest['cohorts'].append(dict(order=order, scanned_pairings=scanned, scan_cap=4096,
                                        connected_count=len(connected), positions=positions, selected=selected,
                                        discovery_seconds=time.monotonic() - start,
                                        scope='Evenly spaced within bounded generator prefix; not population sampling'))
    cases = [(order, partition, first, 4) for order in (3, 4, 6)
             for partition in ('number', 'mixed', 'occupation') for first in (0, 7, 31)]
    cases.append((3, 'number', 7, 6))
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for order, partition, first, modes in cases:
        name = f'n{modes}-o{order}-{partition}-first{first}'
        command = [str(binary), str(folder / f'o{order}.txt'), partition, 'ordered_mapped', str(first), '3', str(modes)]
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
        (folder / f'{name}.stdout').write_text(result.stdout)
        (folder / f'{name}.stderr').write_text(result.stderr)
        manifest['cases'].append(dict(name=name, command=command, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{name}: {result.stderr}')
        data = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(data) == 9 and all(float(row['max_abs_error']) == 0 for row in data)
        assert all(int(row['charged_cache_bytes']) <= 8 * 1024 * 1024 for row in data)
        rows.extend(dict(case=name, **row) for row in data)
        print(name, flush=True)
    with (folder / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

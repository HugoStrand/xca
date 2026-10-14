#!/usr/bin/env python3
"""Measure private device prefix caching against independent complete calls."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'cbuild-cuda')
    parser.add_argument('--machine', required=True)
    parser.add_argument('--cohorts', type=Path, required=True)
    parser.add_argument('--campaign', default='cache-01')
    args = parser.parse_args()
    for name in (args.machine, args.campaign):
        if not name or Path(name).name != name or name in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    binary = args.build.resolve() / 'benchmarks/experiments/block_sparse_cuda/topology_reuse/block_sparse_prefix_benchmark'
    folder = Path(__file__).parent / args.machine / args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(host=platform.node(), platform=platform.platform(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    triqs_root=os.environ.get('TRIQS_ROOT'),
                    gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                 '--format=csv,noheader'], text=True).strip(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), source_hashes={}, cases=[])
    files = list(Path(__file__).parent.glob('*.inc')) + list(Path(__file__).parent.glob('*.cpp'))
    files += [Path(__file__), Path(__file__).with_name('CMakeLists.txt')]
    files += list((ROOT / 'c++/triqs_xca/block_sparse/cuda').glob('*.cpp'))
    for path in files:
        manifest['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for order in (3, 4, 6):
        topology = args.cohorts.resolve() / f'o{order}.txt'
        (folder / topology.name).write_bytes(topology.read_bytes())
        for partition in ('number', 'mixed', 'occupation'):
            for backend in ('library', 'ordered_mapped'):
                name = f'o{order}-{partition}-{backend}'
                command = [str(binary), str(topology), partition, backend]
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
                (folder / f'{name}.stdout').write_text(result.stdout)
                (folder / f'{name}.stderr').write_text(result.stderr)
                manifest['cases'].append(dict(name=name, command=command, returncode=result.returncode))
                if result.returncode:
                    raise RuntimeError(f'{name}: {result.stderr}')
                data = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(data) == 9
                assert all(int(row['charged_cache_bytes']) <= 8 * 1024 * 1024 for row in data)
                if order == 3 and partition == 'number':
                    assert all(int(row['hits']) > 0 for row in data)
                assert all(float(row['max_abs_error']) == 0 for row in data)
                rows.extend(dict(case=name, **row) for row in data)
                print(name, flush=True)
    with (folder / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

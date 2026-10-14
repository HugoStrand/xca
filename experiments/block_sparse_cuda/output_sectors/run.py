#!/usr/bin/env python3
"""Record paired complete self-energy output-streaming timings and sampled memory."""
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
    parser.add_argument('--campaign', default='outputs-01')
    args = parser.parse_args()
    for name in (args.machine, args.campaign):
        if not name or Path(name).name != name or name in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    source = Path(__file__).parent
    folder = source / args.machine / args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    binary = args.build.resolve() / 'benchmarks/experiments/block_sparse_cuda/output_sectors/block_sparse_output_sector_benchmark'
    files = [p for p in source.glob('*') if p.is_file() and (p.suffix in ('.cpp', '.hpp', '.py') or p.name == 'CMakeLists.txt')]
    files += list((ROOT / 'c++/triqs_xca/block_sparse/cuda').glob('*.cpp'))
    files += [ROOT / 'benchmarks/block_sparse_common/fixture.hpp', ROOT / 'benchmarks/block_sparse_common/call_runner.hpp']
    manifest = dict(host=platform.node(), platform=platform.platform(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    triqs_root=os.environ.get('TRIQS_ROOT'),
                    gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, cases=[],
                    memory_scope='Sampled global free-memory drop relative to each call entry, not a continuous or process-private peak; RSS is cumulative for the paired process')
    cases = [(4, 'number', 2, 0, 3), (4, 'mixed', 3, 7, 3), (4, 'occupation', 3, 7, 3),
             (6, 'number', 2, 7, 3), (8, 'number', 1, 0, 1), (10, 'number', 1, 0, 1),
             (6, 'mixed', 2, 0, 3)]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    rows = []
    for modes, partition, order, first, count in cases:
        name = f'n{modes}-{partition}-o{order}'
        command = [str(binary), '--modes', str(modes), '--partition', partition, '--order', str(order),
                   '--first', str(first), '--count', str(count), '--samples', '3']
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
        (folder / f'{name}.stdout').write_text(result.stdout)
        (folder / f'{name}.stderr').write_text(result.stderr)
        manifest['cases'].append(dict(name=name, command=command, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{name}: {result.stderr}')
        data = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(data) == 21 and all(float(row['max_abs_error']) == 0 for row in data)
        for phase in ('first', 'warm', 'changed_G'):
            for sample in {row['sample'] for row in data if row['phase'] == phase}:
                pair = [row for row in data if row['phase'] == phase and row['sample'] == sample]
                resident = next(row for row in pair if row['route'] == 'resident')
                streamed = next(row for row in pair if row['route'] == 'streamed')
                moved = next(row for row in pair if row['route'] == 'resident_move')
                assert moved['device_bound_bytes'] == resident['device_bound_bytes']
                assert int(resident['device_bound_bytes']) - int(streamed['device_bound_bytes']) == int(resident['full_output_bytes']) - int(resident['maximum_tile_bytes'])
        rows.extend(dict(case=name, **row) for row in data)
        print(name, flush=True)
    with (folder / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

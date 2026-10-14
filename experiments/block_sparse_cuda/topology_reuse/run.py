#!/usr/bin/env python3
"""Record bounded structural prefix inventories, without timing or numeric reuse."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'cbuild')
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='prefix-01')
    args = parser.parse_args()
    for name in (args.machine, args.campaign):
        if not name or Path(name).name != name or name in ('.', '..'):
            parser.error('Machine and campaign must be single folder names')
    binary = args.build.resolve() / 'benchmarks/experiments/block_sparse_cuda/topology_reuse/block_sparse_topology_reuse_probe'
    spec = importlib.util.spec_from_file_location('xca_diag', ROOT / 'python/triqs_xca/diag.py')
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    folder = Path(__file__).parent / args.machine / args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(host=platform.node(), platform=platform.platform(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    triqs_root=os.environ.get('TRIQS_ROOT'),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    source_hashes={}, cohorts=[], cases=[])
    for path in (Path(__file__), Path(__file__).with_name('probe.cpp'), ROOT / 'python/triqs_xca/diag.py',
                 ROOT / 'c++/triqs_xca/block_sparse/cuda/path.hpp',
                 ROOT / 'c++/triqs_xca/block_sparse/cuda/diagram_evaluator.cpp'):
        manifest['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for order in (2, 3, 4, 6):
        selected, scanned = [], 0
        for parity, pairing in itertools.islice(diag.all_pairings(order), 4096):
            scanned += 1
            if diag.is_connected(pairing):
                selected.append(pairing)
                if len(selected) == 6:
                    break
        if not selected:
            raise RuntimeError(f'No connected topology within scan cap for order {order}')
        topology_file = folder / f'o{order}.txt'
        topology_file.write_text(f'{len(selected)} {order}\n' + ''.join(
            f'{a} {b}\n' for pairing in selected for a, b in pairing))
        manifest['cohorts'].append(dict(order=order, scanned_pairings=scanned, scan_cap=4096,
                                        selected=selected, selection='first connected; biased, not population estimate'))
        for partition in ('number', 'mixed', 'occupation'):
            for zero_every in (0, 2):
                name = f'o{order}-{partition}-z{zero_every}'
                command = [str(binary), '--topologies', str(topology_file), '--modes', '4',
                           '--partition', partition, '--count', '2', '--path-limit', '4096',
                           '--zero-every', str(zero_every)]
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
                (folder / f'{name}.stdout').write_text(result.stdout)
                (folder / f'{name}.stderr').write_text(result.stderr)
                manifest['cases'].append(dict(name=name, command=command, returncode=result.returncode))
                if result.returncode:
                    raise RuntimeError(f'{name}: {result.stderr}')
                case_rows = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(case_rows) == 3
                if order == 2 or zero_every:
                    assert all(int(row['cross_topology_keys']) == 0 for row in case_rows)
                rows.extend(dict(case=name, **row) for row in case_rows)
    # Resource cap and invalid-input controls exercise the actual executable.
    valid = folder / 'o3.txt'
    limited = subprocess.run([str(binary), '--topologies', str(valid), '--path-limit', '1'],
                             env=env, text=True, capture_output=True, timeout=60)
    assert limited.returncode == 0 and all(row['truncated'] == '1' for row in csv.DictReader(io.StringIO(limited.stdout)))
    malformed = folder / 'malformed.txt'
    malformed.write_text('2 3\n0 3\n')
    invalid = subprocess.run([str(binary), '--topologies', str(malformed)], env=env,
                             text=True, capture_output=True, timeout=60)
    assert invalid.returncode != 0 and 'Missing topology entries' in invalid.stderr
    manifest['controls'] = dict(path_cap=limited.stdout, malformed_stderr=invalid.stderr)
    with (folder / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(folder)


if __name__ == '__main__':
    main()

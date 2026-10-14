#!/usr/bin/env python3
"""Bounded sequential complete-call comparison; run in the CUDA environment."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
build = args.build.resolve()
out = args.output.resolve()
if out.exists():
    raise SystemExit('Use a fresh output directory to preserve prior campaigns')
out.mkdir(parents=True)
binaries = {
    'reference': build / 'benchmarks/block_sparse_cuda/block_sparse_cuda_benchmark',
    'stream': build / 'benchmarks/experiments/block_sparse_cuda/stream_fragments/block_sparse_stream_fragment_benchmark',
    'host_input': build / 'benchmarks/experiments/block_sparse_cuda/host_inputs/block_sparse_host_input_benchmark',
}
manifest = {
    'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'binaries': {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for key, path in binaries.items()},
    'samples_per_route': 3,
    'policy': 'sequential runs; one OMP/BLAS thread; route order alternates by case; shared warm complete-call fixture',
    'cases': [],
}
rows = []
summary = []
cases = [
    ('n4-number-spgf', 4, 'number', 'spgf', False),
    ('n6-number-spgf', 6, 'number', 'spgf', False),
    ('n8-number-spgf', 8, 'number', 'spgf', False),
    ('n6-mixed-spgf', 6, 'mixed', 'spgf', False),
    ('n4-number-dynamic-correlator', 4, 'number', 'correlator', True),
    ('n6-number-dynamic-correlator', 6, 'number', 'correlator', True),
    ('n10-number-oca', 10, 'number', 'se', False),
]
for index, (name, modes, partition, observable, dynamic) in enumerate(cases):
    group = []
    for route in (('reference', 'host_input', 'stream') if index % 2 == 0 else ('stream', 'host_input', 'reference')):
        command = [str(binaries[route]), '--modes', str(modes), '--partition', partition,
                   '--observable', observable, '--order', '2', '--first', '0', '--count', '1', '--samples', '3']
        if dynamic:
            command.append('--dynamic')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        process = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=120)
        (out / f'{name}-{route}.csv').write_text(process.stdout)
        (out / f'{name}-{route}.log').write_text(process.stderr)
        manifest['cases'].append({'name': name, 'route': route, 'command': command, 'returncode': process.returncode})
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if process.returncode:
            raise SystemExit(f'{name}/{route} failed; retained diagnostics, stopped campaign')
        data = list(csv.DictReader(io.StringIO(process.stdout)))
        assert len(data) == 3, (name, route)
        for row in data:
            row.update(case=name, route=route)
        rows += data
        group += data
        summary.append(dict(case=name, route=route, samples=3,
                            median_ms=statistics.median(float(x['call_ms']) for x in data),
                            bound_bytes=data[0]['device_bound_bytes'],
                            sampled_peak_delta=max(int(x['device_peak_delta']) for x in data),
                            max_abs_error=max(float(x['max_abs_error']) for x in data)))
    # Exact reference/candidate outputs are tested directly by C++; these fields
    # also detect a recording mismatch in ordinary complete-call benchmark runs.
    for key in ('signal_norm', 'max_abs_error', 'proposed', 'accepted', 'pruned'):
        assert len({x[key] for x in group}) == 1, (name, key)
    print(name, [x for x in summary if x['case'] == name], flush=True)
for filename, data in [('samples.csv', rows), ('summary.csv', summary)]:
    with (out / filename).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
print(f'Recorded {len(rows)} samples', flush=True)

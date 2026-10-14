#!/usr/bin/env python3
"""Record bounded GH200 cohorts without requiring a Git checkout."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revisions', type=Path, required=True)
    parser.add_argument('--cohort', choices=['initial', 'scalar', 'density', 'signs', 'resident', 'staging', 'streams'], default='initial')
    args = parser.parse_args()
    binaries = {
        'cpu': args.build/'benchmarks/block_sparse_cpu/block_sparse_cpu_benchmark',
        'library': args.build/'benchmarks/block_sparse_cuda/block_sparse_cuda_benchmark',
        'ordered': args.build/'benchmarks/block_sparse_cuda/block_sparse_cuda_ordered_benchmark',
    }
    # Explicit bounded windows; retain rejected-path cases alongside active work.
    cases = [
        ('scalar-se', 4, 'occupation', 'se', 3, False, 0),
        ('scalar-se-next', 4, 'occupation', 'se', 3, False, 1),
        ('mixed-correlator', 4, 'mixed', 'correlator', 2, True, 0),
        ('number-spgf', 6, 'number', 'spgf', 2, True, 0),
        ('number-se', 6, 'number', 'se', 3, False, 0),
        ('number-n8', 8, 'number', 'se', 1, False, 0),
        ('number-n10', 10, 'number', 'se', 1, False, 0),
    ]
    cases = [(*case, 1, 'crossing') for case in cases]
    if args.cohort == 'scalar':
        experiment = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_paths'
        binaries = {
            'cpu': binaries['cpu'], 'ordered': binaries['ordered'],
            'cpu_constants': experiment/'block_sparse_cpu_scalar_constants_benchmark',
            'cuda_constants': experiment/'block_sparse_scalar_constants_benchmark',
        }
        cases = [
            ('scalar-zero', 4, 'occupation', 'se', 3, False, 0, 1, 'crossing'),
            ('scalar-active', 4, 'occupation', 'se', 3, False, 1, 1, 'crossing'),
            ('scalar-range', 6, 'occupation', 'se', 2, False, 1, 8, 'crossing'),
            ('scalar-n10', 10, 'occupation', 'se', 2, False, 1, 1, 'crossing'),
            ('scalar-order6', 4, 'occupation', 'se', 6, False, 1, 8, 'chain'),
            ('mixed-fallback', 4, 'mixed', 'correlator', 2, True, 0, 1, 'crossing'),
        ]
    if args.cohort in ('density', 'signs', 'resident', 'staging', 'streams'):
        experiment = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_sets'
        binaries = {
            'cpu': experiment/'block_sparse_cpu_density_sets_benchmark',
            'cpu_batched': experiment/'block_sparse_cpu_batched_density_sets_benchmark',
            'general': experiment/'block_sparse_general_density_sets_benchmark',
            'batched': experiment/'block_sparse_scalar_density_sets_benchmark',
        }
        cases = [
            ('density-n4-nca', 4, 'occupation', 'se', 1, True, 0, 2, 'crossing'),
            ('density-n4-oca', 4, 'occupation', 'se', 2, True, 1, 8, 'crossing'),
            ('density-n6-oca', 6, 'occupation', 'se', 2, True, 1, 8, 'crossing'),
            ('density-n8-oca', 8, 'occupation', 'se', 2, True, 1, 1, 'crossing'),
        ]
        if args.cohort == 'signs':
            signs = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_signs'
            binaries = {
                'cpu_baseline': experiment/'block_sparse_cpu_batched_density_sets_benchmark',
                'gpu_baseline': experiment/'block_sparse_scalar_density_sets_benchmark',
                'cpu_signs': signs/'block_sparse_cpu_density_signs_benchmark',
                'gpu_signs': signs/'block_sparse_scalar_density_signs_benchmark',
            }
        if args.cohort == 'resident':
            signs = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_signs'
            resident = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_resident'
            binaries = {
                'cpu_signs': signs/'block_sparse_cpu_density_signs_benchmark',
                'gpu_signs': signs/'block_sparse_scalar_density_signs_benchmark',
                'gpu_resident': resident/'block_sparse_scalar_density_resident_benchmark',
            }
        if args.cohort == 'staging':
            signs = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_signs'
            resident = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_resident'
            staging = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_staging'
            binaries = {
                'cpu_signs': signs/'block_sparse_cpu_density_signs_benchmark',
                'gpu_resident': resident/'block_sparse_scalar_density_resident_benchmark',
                'gpu_staging': staging/'block_sparse_scalar_density_staging_benchmark',
            }
        if args.cohort == 'streams':
            signs = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_signs'
            staging = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_staging'
            streams = args.build/'benchmarks/experiments/block_sparse_cuda/scalar_streams'
            binaries = {
                'cpu_signs': signs/'block_sparse_cpu_density_signs_benchmark',
                'gpu_staging': staging/'block_sparse_scalar_density_staging_benchmark',
                'gpu_streams': streams/'block_sparse_scalar_density_streams_benchmark',
            }
    manifest = dict(revisions=json.loads(args.revisions.read_text()),
                    cohort=args.cohort,
                    host=platform.node(), platform=platform.platform(),
                    job_id=os.environ.get('SLURM_JOB_ID'), samples_per_case=3,
                    timeout_per_process_seconds=120, cases=[],
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    binaries={key: dict(path=str(binary.resolve()),
                              sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                              for key, binary in binaries.items()})
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name, modes, partition, observable, order, dynamic, first, count, topology in cases:
        for backend, binary in binaries.items():
            command = [str(binary.resolve()), '--modes', str(modes), '--partition', partition,
                       '--observable', observable, '--order', str(order), '--first', str(first),
                       '--count', str(count), '--topology', topology, '--samples', '3']
            if dynamic:
                command.append('--dynamic')
            start = time.monotonic()
            record = dict(name=name, backend=backend, command=command)
            try:
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
                record['returncode'] = result.returncode
                (args.output/f'{name}-{backend}.csv').write_text(result.stdout)
                (args.output/f'{name}-{backend}.log').write_text(result.stderr)
                if result.returncode == 0:
                    data = list(csv.DictReader(io.StringIO(result.stdout)))
                    if len(data) != 3:
                        raise ValueError('Unexpected sample count')
                    rows.extend(dict(case=name, **row) for row in data)
            except subprocess.TimeoutExpired as error:
                record['timeout'] = True
                for suffix, content in [('csv', error.stdout), ('log', error.stderr)]:
                    if content:
                        (args.output/f'{name}-{backend}.{suffix}').write_bytes(
                            content if isinstance(content, bytes) else content.encode())
            record['elapsed_seconds'] = time.monotonic()-start
            manifest['cases'].append(record)
            (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if rows:
                with (args.output/'samples.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            print(name, backend, record.get('returncode', 'timeout'), flush=True)
            if record.get('returncode') != 0:
                raise SystemExit('Stop cohort on failure; inspect retained logs before retrying')


if __name__ == '__main__':
    main()

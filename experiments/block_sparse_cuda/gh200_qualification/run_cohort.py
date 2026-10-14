#!/usr/bin/env python3
"""Record a bounded three-backend GH200 cohort without requiring a Git checkout."""
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
    manifest = dict(revisions=json.loads(args.revisions.read_text()),
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
    for name, modes, partition, observable, order, dynamic, first in cases:
        for backend, binary in binaries.items():
            command = [str(binary.resolve()), '--modes', str(modes), '--partition', partition,
                       '--observable', observable, '--order', str(order), '--first', str(first),
                       '--count', '1', '--samples', '3']
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

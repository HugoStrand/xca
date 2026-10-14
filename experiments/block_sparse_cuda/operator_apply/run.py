#!/usr/bin/env python3
"""Record bounded E3 operator controls in isolated processes."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='operator-01')
    args = parser.parse_args()
    for label in (args.machine, args.campaign):
        if not label or Path(label).name != label or label in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    binary = args.binary.resolve()
    folder = HERE/args.machine/args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    prefix = Path(env['TRIQS_ROOT'])
    sources = [HERE/'block_sparse_operator_apply.cpp', HERE/'kernels.cu', HERE/'kernels.hpp', HERE/'CMakeLists.txt', Path(__file__),
               *ROOT.glob('c++/triqs_xca/block_sparse/cuda/*.cpp'), *ROOT.glob('c++/triqs_xca/block_sparse/cuda/*.hpp')]
    manifest = dict(schema='block-sparse-operator-v1', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    platform=platform.platform(), host=platform.node(), triqs_root=str(prefix),
                    cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                    gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
                    binary_sha256=digest(binary),
                    compiler=subprocess.check_output(['clang++-19', '--version'], text=True).splitlines()[0],
                    nvcc=subprocess.check_output(['nvcc', '--version'], text=True),
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    nda_library_sha256=digest((prefix/'lib/libnda_c.so').resolve()), cases=[])
    rows = []
    for modes in (2, 4, 6, 8, 10, 14):
        for terms in (1, modes):
            for width in (1, 16):
                name = f'n{modes}-terms{terms}-width{width}'
                command = [str(binary), '--modes', str(modes), '--terms', str(terms), '--width', str(width), '--samples', '3', '--reuse', '8']
                case = dict(name=name, command=command)
                start = time.monotonic()
                try:
                    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
                    case['returncode'] = run.returncode
                    (folder/(name+'.csv')).write_text(run.stdout)
                    if run.stderr:
                        (folder/(name+'.log')).write_text(run.stderr)
                    if run.returncode == 0:
                        data = list(csv.DictReader(io.StringIO(run.stdout)))
                        if len(data) != 18:
                            case['validation_error'] = 'Expected three routes, two directions and three samples'
                            case['returncode'] = 1
                        else:
                            rows.extend(data)
                except subprocess.TimeoutExpired as error:
                    case['timeout_seconds'] = 120
                    for suffix, content in (('csv', error.stdout), ('log', error.stderr)):
                        if content:
                            (folder/(name+'.'+suffix)).write_bytes(content if isinstance(content, bytes) else content.encode())
                case['elapsed_seconds'] = time.monotonic()-start
                manifest['cases'].append(case)
                (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                if rows:
                    with (folder/'samples.csv').open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                print(name, case.get('returncode', 'timeout'), f"{case['elapsed_seconds']:.2f}s", flush=True)
    if any(case.get('returncode') != 0 for case in manifest['cases']):
        raise SystemExit('Campaign has failures; inspect raw logs and manifest')


if __name__ == '__main__':
    main()

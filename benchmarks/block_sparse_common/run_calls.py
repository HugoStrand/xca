#!/usr/bin/env python3
"""Record isolated bounded CPU/CUDA reference calls with per-machine provenance."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-binary', type=Path, required=True)
    parser.add_argument('--cuda-binary', type=Path, required=True)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='reference-01')
    parser.add_argument('--timeout', type=float, default=120.)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    for label in (args.machine, args.campaign):
        if not label or Path(label).name != label or label in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    common = ROOT/'benchmarks/block_sparse_common'/args.machine/args.campaign
    common.mkdir(parents=True, exist_ok=False)
    sources = [*ROOT.glob('c++/triqs_xca/block_sparse/**/*.cpp'),
               *ROOT.glob('c++/triqs_xca/block_sparse/**/*.hpp'),
               *ROOT.glob('benchmarks/block_sparse_common/*.[ch]pp'),
               *ROOT.glob('benchmarks/block_sparse_*/*_benchmark.cpp'), Path(__file__)]
    sources += [*ROOT.glob('experiments/block_sparse_cuda/**/*.cpp'),
                *ROOT.glob('experiments/block_sparse_cuda/**/*.hpp'),
                *ROOT.glob('experiments/block_sparse_cuda/**/CMakeLists.txt')]
    sources += [ROOT/'c++/triqs_xca'/p for p in ('backbone.cpp', 'backbone.hpp', 'hyb.cpp', 'hyb.hpp', 'operator_statistics.hpp')]
    manifest = dict(schema='block-sparse-call-v1', host=platform.node(), platform=platform.platform(),
                    cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    triqs_root=env.get('TRIQS_ROOT'), cases=[],
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    manifest['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    if manifest['triqs_root']:
        prefix = Path(manifest['triqs_root'])
        headers = [p for directory in ('cppdlr', 'triqs/atom_diag', 'triqs/hilbert_space')
                   for p in (prefix/'include'/directory).glob('**/*.hpp')]
        manifest['triqs_header_hashes'] = {str(p.relative_to(prefix)): hashlib.sha256(p.read_bytes()).hexdigest() for p in headers}
    manifest['compiler'] = subprocess.check_output(['clang++-19', '--version'], text=True).splitlines()[0]
    manifest['gpu_info'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    binaries = {'cpu': args.cpu_binary.resolve(), 'cuda': args.cuda_binary.resolve()}
    manifest['binary_sha256'] = {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in binaries.items()}
    rows = []
    cases = [(4, regime, 'se', 2, False, 3) for regime in ('occupation', 'number', 'mixed')]
    cases += [(4, regime, observable, 2, True, 3) for regime in ('occupation', 'number', 'mixed') for observable in ('spgf', 'correlator')]
    cases += [(6, regime, 'se', 3, False, 3) for regime in ('occupation', 'number', 'mixed')]
    cases += [(8, 'number', 'se', 2, False, 3), (10, 'number', 'se', 2, False, 1), (10, 'occupation', 'se', 2, False, 1)]
    for modes, partition, observable, order, dynamic, samples in cases:
        name = f'n{modes}-{partition}-{observable}-o{order}' + ('-dynamic' if dynamic else '')
        for backend, binary in binaries.items():
            folder = ROOT/('benchmarks/block_sparse_cpu' if backend == 'cpu' else 'benchmarks/block_sparse_cuda')/args.machine/args.campaign
            folder.mkdir(parents=True, exist_ok=True)
            command = [str(binary), '--modes', str(modes), '--partition', partition, '--observable', observable,
                       '--order', str(order), '--count', '1', '--samples', str(samples)]
            if dynamic: command.append('--dynamic')
            case = dict(name=name, backend=backend, command=command)
            start = time.monotonic()
            try:
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout)
                case['returncode'] = result.returncode
                (folder/(name+'.csv')).write_text(result.stdout)
                if result.stderr: (folder/(name+'.log')).write_text(result.stderr)
                if result.returncode == 0:
                    data = list(csv.DictReader(io.StringIO(result.stdout)))
                    if len(data) != samples: raise ValueError('Unexpected benchmark sample count')
                    rows.extend(data)
            except subprocess.TimeoutExpired as error:
                case['timeout_seconds'] = args.timeout
                for suffix, content in (('csv', error.stdout), ('log', error.stderr)):
                    if content: (folder/(name+'.'+suffix)).write_bytes(content if isinstance(content, bytes) else content.encode())
            case['elapsed_seconds'] = time.monotonic()-start
            manifest['cases'].append(case)
            (common/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if rows:
                with (common/'samples.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            print(name, backend, case.get('returncode', 'timeout'), f"{case['elapsed_seconds']:.2f}s", flush=True)
    if any(case.get('returncode') != 0 for case in manifest['cases']):
        raise SystemExit('Campaign has failures/timeouts; inspect manifest and raw logs')


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Record bounded complete-call special-vertex retention comparisons."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT/'cbuild-cuda')
    parser.add_argument('--machine', required=True)
    parser.add_argument('--campaign', default='kappa-01')
    parser.add_argument('--with-mapped', action='store_true', help='Include the native mapped-kappa variant')
    args = parser.parse_args()
    for name in (args.machine, args.campaign):
        if not name or Path(name).name != name or name in ('.', '..'):
            parser.error('Machine/campaign must be single folder names')
    base = args.build.resolve()/'benchmarks'
    binaries = {
        'reference': base/'block_sparse_cuda/block_sparse_cuda_ordered_benchmark',
        'slots2': base/'experiments/block_sparse_cuda/special_vertex/block_sparse_kappa_2_benchmark',
        'slots16': base/'experiments/block_sparse_cuda/special_vertex/block_sparse_kappa_16_benchmark',
    }
    if args.with_mapped:
        binaries['mapped'] = base/'experiments/block_sparse_cuda/mapped_special/block_sparse_mapped_special_benchmark'
    sources = [*ROOT.glob('c++/triqs_xca/block_sparse/**/*.hpp'),
               *ROOT.glob('c++/triqs_xca/block_sparse/**/*.cpp'),
               *ROOT.glob('c++/triqs_xca/block_sparse/**/*.cu'),
               *ROOT.glob('c++/triqs_xca/block_sparse/**/CMakeLists.txt'),
               ROOT/'benchmarks/block_sparse_cuda/block_sparse_cuda_ordered_benchmark.cpp',
               *[ROOT/'c++/triqs_xca'/n for n in ('backbone.cpp', 'backbone.hpp', 'hyb.cpp', 'hyb.hpp')],
               *ROOT.glob('benchmarks/block_sparse_common/*.[ch]pp'),
               *Path(__file__).parent.glob('*'),
               *ROOT.glob('experiments/block_sparse_cuda/mapped_special/*')]
    sources = [p for p in sources if p.is_file()]
    manifest = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    host=platform.node(), platform=platform.platform(),
                    cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                    triqs_root=os.environ.get('TRIQS_ROOT'),
                    compiler=subprocess.check_output(['clang++-19', '--version'], text=True).splitlines()[0],
                    gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
                    binary_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in binaries.items()},
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    cases=[], thread_environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    if manifest['triqs_root']:
        prefix = Path(manifest['triqs_root'])
        headers = [p for directory in ('cppdlr', 'triqs/atom_diag', 'triqs/hilbert_space')
                   for p in (prefix/'include'/directory).glob('**/*.hpp')]
        manifest['triqs_header_hashes'] = {str(p.relative_to(prefix)): hashlib.sha256(p.read_bytes()).hexdigest() for p in headers}
    envelopes = []
    for modes, rank in ((10, 23), (14, 26)):
        maximum = math.comb(modes, modes//2)
        function = math.comb(2*modes, modes)*rank*16
        tile = maximum*maximum*rank*16
        for cap in (0, 2, 16):
            slots = min(cap, modes)
            envelopes.append(dict(modes=modes, rank=rank, cap=cap, slots=slots,
                                  function_payload_bytes=function, maximum_rank_buffer_bytes=tile,
                                  extra_cache_envelope_bytes=slots*tile,
                                  variable_reference_envelope_bytes=2*function+13*tile,
                                  variable_candidate_envelope_bytes=2*function+(13+slots)*tile))
    manifest['analytic_number_symmetry_envelopes'] = envelopes
    manifest['analytic_column_map_envelopes'] = [dict(modes=n, rank=r, fields=n, maximum_dimension=math.comb(n,n//2),
                                                     extra_map_bytes=n*(math.comb(n,n//2)*24+r*16))
                                                for n,r in ((10,23),(14,26))]
    manifest['analytic_envelope_scope'] = 'Binomial dimensions only, no atom_diag/evaluation; variable device envelope excludes fixed tables/operator reserve/host/opaque allocations. Rank 26 is the retained fourteen-mode inventory proxy, not a new diagonalization.'
    folder = ROOT/'benchmarks/block_sparse_common'/args.machine/args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    raw = ROOT/'benchmarks/block_sparse_cuda'/args.machine/args.campaign
    raw.mkdir(parents=True, exist_ok=False)
    rows = []
    env = dict(os.environ, **manifest['thread_environment'])
    # NCA includes both directions. Larger selected calls keep diagonalization
    # and CPU reference time bounded; no N=14 atom_diag construction is requested.
    cases = [(4, 'number', 1, 2), (4, 'number', 2, 1), (4, 'mixed', 2, 1),
             (4, 'occupation', 2, 1), (6, 'number', 2, 1),
             (8, 'number', 1, 2), (10, 'number', 1, 1)]
    for modes, partition, order, count in cases:
        name = f'n{modes}-{partition}-o{order}'
        for backend, binary in binaries.items():
            command = [str(binary), '--modes', str(modes), '--partition', partition, '--observable', 'se',
                       '--order', str(order), '--count', str(count), '--samples', '5']
            start = time.monotonic()
            try:
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
            except subprocess.TimeoutExpired as error:
                for suffix, data in (('csv', error.stdout), ('log', error.stderr)):
                    if data:
                        (raw/(name+'-'+backend+'.'+suffix)).write_bytes(data if isinstance(data, bytes) else data.encode())
                manifest['cases'].append(dict(name=name, backend=backend, command=command, timeout_seconds=120,
                                              elapsed_seconds=time.monotonic()-start))
                (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                raise SystemExit('Benchmark timed out; inspect retained manifest and partial output')
            (raw/(name+'-'+backend+'.csv')).write_text(result.stdout)
            if result.stderr:
                (raw/(name+'-'+backend+'.log')).write_text(result.stderr)
            manifest['cases'].append(dict(name=name, backend=backend, command=command,
                                          returncode=result.returncode, elapsed_seconds=time.monotonic()-start))
            (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if result.returncode:
                raise SystemExit(result.stderr)
            data = list(csv.DictReader(result.stdout.splitlines()))
            if len(data) != 5:
                raise SystemExit('Unexpected sample count')
            rows.extend(data)
            with (folder/'samples.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(name, backend, 'passed', flush=True)


if __name__ == '__main__':
    main()

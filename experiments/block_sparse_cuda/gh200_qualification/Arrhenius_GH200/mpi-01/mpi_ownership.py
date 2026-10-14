#!/usr/bin/env python3
"""Measure bounded MPI self-energy scheduling without changing the evaluator."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
from mpi4py import MPI
from mpi_fixture import make_solver

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


def collective(operation):
    """Every rank reaches error coordination once, including ranks with no jobs."""
    value, error = None, None
    try:
        value = operation()
    except Exception:
        error = traceback.format_exc()
    errors = COMM.allgather(error)
    if any(errors):
        raise RuntimeError('\n'.join(e for e in errors if e))
    return value


def flat(output):
    return np.concatenate([np.asarray(block.data).reshape(-1) for _, block in output])


def bounds(count, rank):
    chunk, extra = divmod(count, SIZE)
    first = rank*chunk+min(rank, extra)
    return first, first+chunk+(rank < extra)


def compare(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise RuntimeError('Invalid result shape or nonfinite output')
    error = float(np.max(np.abs(actual-reference), initial=0))
    norm = float(np.max(np.abs(reference), initial=0))
    if error > 1e-10*norm+1e-14:
        raise RuntimeError(f'CPU/control mismatch: {error=} {norm=}')
    return error, norm


def record_reduce(send):
    """Explicit barrier exposes arrival wait; payload latency excludes this wait."""
    start = time.perf_counter()
    COMM.Barrier()
    wait = time.perf_counter()-start
    received = np.empty_like(send)
    start = time.perf_counter()
    COMM.Allreduce(send, received, op=MPI.SUM)
    reduction = time.perf_counter()-start
    return received, wait, reduction


def run(args):
    if SIZE != args.ranks:
        raise RuntimeError(f'Expected {args.ranks} ranks, got {SIZE}')
    if COMM.allreduce(RANK+1) != SIZE*(SIZE+1)//2:
        raise RuntimeError('MPI collective preflight failed')
    output = Path(args.output)
    collective(lambda: output.mkdir(parents=True, exist_ok=False) if RANK == 0 else None)
    start = time.perf_counter()
    solver = collective(lambda: make_solver(args, args.backend))
    setup_s = time.perf_counter()-start
    topologies = [np.array(t, np.int32) for t in (
        [[0, 3], [1, 4], [2, 5]], [[0, 2], [1, 4], [3, 5]])]
    counts = collective(lambda: [solver.d.get_num_self_energy_backbones(t) for t in topologies])
    if any(args.first+args.count > count for count in counts):
        raise ValueError(f'Window exceeds topology counts {counts}')
    # CPU control is outside all measured cohorts; only rank zero evaluates it.
    start = time.perf_counter()
    def cpu_control():
        if RANK != 0:
            return None
        cpu = make_solver(args, 'cpu')
        return np.array([flat(cpu.d.compute_self_energy_range(cpu.G0, t, i, i+1))
                         for t in topologies for i in range(args.first, args.first+args.count)])
    expected = COMM.bcast(collective(cpu_control), root=0)
    control_s = time.perf_counter()-start
    jobs, width = expected.shape
    if expected.dtype != np.complex128:
        raise RuntimeError('Expected complex128 output')
    per_index = []
    phases = []
    canonical = None
    policies = ('topology_split', 'flat_cyclic', 'solver_ranges')
    # Warm every policy; all measurements still retain zero/weak indexed jobs.
    profiler = None
    if args.profile:
        import ctypes
        import ctypes.util
        profiler = ctypes.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so')
    for sample in range(-1, args.samples):
        if sample == 0 and profiler is not None:
            if profiler.cudaProfilerStart() != 0:
                raise RuntimeError('cudaProfilerStart failed')
        for policy in policies:
            COMM.Barrier()
            begin = time.perf_counter()
            metrics = dict(evaluation_s=0., packing_s=0., arrival_wait_s=0., reduction_s=0.,
                           accumulation_s=0., error_coordination_s=0., payload_bytes=0,
                           collectives=0, indexed_jobs=0, api_calls=0)
            send = np.zeros((jobs, width), np.complex128) if policy != 'solver_ranges' else None
            owned = []
            if policy != 'solver_ranges':
                for job in range(jobs):
                    _, position = divmod(job, args.count)
                    lo, hi = bounds(args.count, RANK)
                    if (policy == 'flat_cyclic' and job % SIZE == RANK) or (
                            policy == 'topology_split' and lo <= position < hi):
                        owned.append(job)
                def evaluate_owned():
                    for job in owned:
                        topology, position = divmod(job, args.count)
                        index = args.first+position
                        start = time.perf_counter()
                        result = solver.d.compute_self_energy_range(solver.G0, topologies[topology], index, index+1)
                        elapsed = time.perf_counter()-start
                        metrics['evaluation_s'] += elapsed
                        start = time.perf_counter()
                        send[job] = flat(result)
                        metrics['packing_s'] += time.perf_counter()-start
                        metrics['api_calls'] += 1
                        if sample >= 0:
                            per_index.append(dict(sample=sample, policy=policy, rank=RANK,
                                                  topology=topology, index=index, api_s=elapsed,
                                                  signal_norm=float(np.max(np.abs(send[job]), initial=0))))
                start = time.perf_counter()
                collective(evaluate_owned)
                metrics['error_coordination_s'] = time.perf_counter()-start-metrics['evaluation_s']-metrics['packing_s']
                received, wait, reduction = record_reduce(send)
                metrics.update(arrival_wait_s=wait, reduction_s=reduction,
                               payload_bytes=send.nbytes, collectives=1, indexed_jobs=len(owned))
                # Each slot has exactly one owner, so reduction cannot regroup paths.
                start = time.perf_counter()
                indexed = received
                total = np.zeros(width, np.complex128)
                for value in received:
                    total += value
                metrics['accumulation_s'] += time.perf_counter()-start
            else:
                # Actual contiguous range calls and blockwise payload reductions,
                # matching Solver's per-topology grouping. Error coordination uses
                # allgather above rather than the Solver's scalar failure allreduce.
                total = np.zeros(width, np.complex128)
                lo, hi = bounds(args.count, RANK)
                metrics['indexed_jobs'] = len(topologies)*(hi-lo)
                for topology in topologies:
                    def evaluate_range():
                        start = time.perf_counter()
                        value = solver.d.compute_self_energy_range(
                            solver.G0, topology, args.first+lo, args.first+hi)
                        metrics['evaluation_s'] += time.perf_counter()-start
                        return value
                    before = metrics['evaluation_s']
                    start = time.perf_counter()
                    result = collective(evaluate_range)
                    metrics['error_coordination_s'] += time.perf_counter()-start-(metrics['evaluation_s']-before)
                    metrics['api_calls'] += 1
                    offset = 0
                    for _, block in result:
                        start = time.perf_counter()
                        packed = np.ascontiguousarray(block.data).reshape(-1)
                        metrics['packing_s'] += time.perf_counter()-start
                        received, wait, reduction = record_reduce(packed)
                        metrics['arrival_wait_s'] += wait
                        metrics['reduction_s'] += reduction
                        metrics['collectives'] += 1
                        metrics['payload_bytes'] += packed.nbytes
                        start = time.perf_counter()
                        total[offset:offset+len(received)] += received
                        offset += len(received)
                        metrics['accumulation_s'] += time.perf_counter()-start
            elapsed = time.perf_counter()-begin
            if policy != 'solver_ranges':
                compare(indexed, expected)
                if canonical is None:
                    canonical = indexed.copy()
                elif not np.array_equal(canonical, indexed):
                    raise RuntimeError('Ownership changed an indexed result')
            reference_total = np.zeros(width, np.complex128)
            for value in expected:
                reference_total += value
            error, norm = collective(lambda: compare(total, reference_total))
            if sample >= 0:
                phases.append(dict(sample=sample, policy=policy, rank=RANK,
                                   total_s=elapsed, max_abs_error=error, signal_norm=norm, **metrics))
    if profiler is not None and profiler.cudaProfilerStop() != 0:
        raise RuntimeError('cudaProfilerStop failed')
    package = Path(__import__('triqs_xca').__file__).resolve().parent
    extensions = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('_triqs_xca*.so')}
    metadata = COMM.gather(dict(rank=RANK, host=socket.gethostname(), pid=os.getpid(),
                               extensions=extensions, cpu_affinity=sorted(os.sched_getaffinity(0)),
                               setup_s=setup_s, cpu_control_s=control_s,
                               module=str(Path(__import__('triqs_xca').__file__).resolve())), root=0)
    all_phases, all_jobs = COMM.gather(phases, root=0), COMM.gather(per_index, root=0)
    if RANK == 0:
        for name, groups in [('phases.csv', all_phases), ('jobs.csv', all_jobs)]:
            rows = [row for group in groups for row in group]
            with (output/name).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
        (output/'manifest.json').write_text(json.dumps(dict(arguments=vars(args),
            topologies=[t.tolist() for t in topologies], topology_counts=counts,
            indexed_jobs=jobs, output_complex_elements=width, canonical_ownership_exact=True,
            rank_metadata=metadata, sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_name('mpi_fixture.py'))}), indent=2)+'\n')
        print(f'MPI ownership passed: {SIZE} ranks, {jobs} indexed jobs, {args.samples} samples', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--operator-backend', choices=('library', 'ordered'), default='library')
    parser.add_argument('--special-backend', choices=('reference', 'mapped'), default='reference')
    parser.add_argument('--partition', choices=('number', 'occupation', 'mixed'), default='number')
    parser.add_argument('--modes', type=int, choices=(2, 3, 4), default=3)
    parser.add_argument('--profile', action='store_true', help='CUDA profiler capture range after warmup; use separate instrumented trials')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--count', type=int, choices=range(1, 5), default=3)
    parser.add_argument('--samples', type=int, choices=range(1, 4), default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.profile and args.backend != 'cuda':
        parser.error('--profile requires CUDA')
    if not 1 <= args.ranks <= 4 or not 0 <= args.first <= 16:
        parser.error('Ranks must be1..4 and first must be0..16')
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        COMM.Abort(1)

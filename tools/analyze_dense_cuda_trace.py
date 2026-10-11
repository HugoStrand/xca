#!/usr/bin/env python3
"""Audit Nsight SQLite traversal ranges; emit CSV, fail on inner allocations/copies.

GPU activity is selected by timestamps: the evaluator synchronizes its default
stream before ending a range. UVM page migrations are reported separately from
explicit host tensor transfers. Kernel families use demangled NVIDIA names.
"""
import argparse
import collections
import csv
import sqlite3
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--phases", action="store_true", help="report phase ranges contained in traversal (excludes constructor warmup)")
    parser.add_argument("--within", help="only audit traversals inside this exact NVTX range name")
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.trace}?mode=ro", uri=True)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    required = {"NVTX_EVENTS", "CUDA_GPU_MEMORY_USAGE_EVENTS", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    if not required <= tables:
        raise RuntimeError(f"incomplete CUDA/memory trace: missing {required-tables}")
    ranges = list(db.execute("select start,end,text from NVTX_EVENTS where text like 'xca.%traversal' order by start"))
    if not ranges or any(end is None for _, end, _ in ranges):
        raise RuntimeError("missing or incomplete traversal ranges")
    if args.within:
        parents = list(db.execute("select start,end from NVTX_EVENTS where text=? and end is not null", (args.within,)))
        ranges = [row for row in ranges if any(begin <= row[0] and row[1] <= finish for begin, finish in parents)]
        if not ranges:
            raise RuntimeError("no traversal in requested enclosing range")
    if args.phases:
        traversals = ranges
        ranges = [row for row in db.execute("select start,end,text from NVTX_EVENTS where text like 'xca.phase.%' order by start")
                  if row[1] is not None and any(begin <= row[0] and row[1] <= finish for begin, finish, _ in traversals)]
        if not ranges:
            raise RuntimeError("missing traversal phase ranges")
    live = peak = 0
    # Nsight-visible device, array, managed and static allocations. This does
    # not claim to measure unreported driver/context memory or physical UVM residency.
    for size, operation in db.execute("select bytes,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS where memKind in (2,3,4,5,6) order by start"):
        live += size if operation == 0 else -size
        if operation not in (0,1) or live < 0:
            raise RuntimeError("incomplete or unknown allocation history")
        peak = max(peak,live)
    headers = ["trace", "tracked_peak_device_allocation_bytes", "range", "occurrence", "wall_ms", "allocation_events", "deallocation_events", "host_copy_bytes", "max_host_copy_bytes", "uvm_copy_bytes", "max_uvm_copy_bytes", "d2d_copy_bytes", "kernel_launches", "kernel_ms", "gemm_ms", "solve_ms"]
    output = csv.DictWriter(sys.stdout, headers, lineterminator="\n")
    output.writeheader()
    seen = collections.Counter()
    failed = False
    for start, end, name in ranges:
        seen[name] += 1
        allocations = db.execute("select count(*) from CUDA_GPU_MEMORY_USAGE_EVENTS where start>=? and start<? and memoryOperationType=0", (start, end)).fetchone()[0]
        deallocations = db.execute("select count(*) from CUDA_GPU_MEMORY_USAGE_EVENTS where start>=? and start<? and memoryOperationType=1", (start,end)).fetchone()[0]
        copies = list(db.execute("select bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?", (start, end)))
        host = [size for size, kind in copies if kind in (1, 2, 3, 4, 6, 7)]
        uvm = [size for size, kind in copies if kind in (11, 12)]
        if any(kind not in range(1,14) for _, kind in copies):
            raise RuntimeError("unknown memory-copy kind")
        kernels = list(db.execute("select k.end-k.start,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.start<?", (start, end)))
        gemm = sum(t for t, n in kernels if "gemm" in n.lower() or "splitkreduce" in n.lower())
        solve = sum(t for t, n in kernels if any(x in n.lower() for x in ("trsm", "laswp", "copy_info")))
        output.writerow(dict(zip(headers, [args.trace.name, peak, name, seen[name], (end-start)/1e6, allocations, deallocations, sum(host), max(host, default=0), sum(uvm), max(uvm, default=0), sum(size for size, kind in copies if kind==8), len(kernels), sum(t for t,_ in kernels)/1e6, gemm/1e6, solve/1e6])))
        failed |= bool(allocations or deallocations or host)
    # Strict gate for any allocation or explicit host tensor transfer in traversal.
    # UVM is not silently ignored: byte totals/maxima remain in the output and
    # must be interpreted against metadata/table sizes and page granularity.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

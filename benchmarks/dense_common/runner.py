#!/usr/bin/env python3
"""Run one serial benchmark process, preserve raw samples/provenance and summarize.

Extra arguments after -- are passed verbatim to the C++ harness. A failed comparison
or incomplete process never produces a successful summary. Output directories are new.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
KEY = ['fixture','N','r','n','p','beta','lambda','order','topology','method',
       'requested_indices','indices','total_indices','index_list','seed']
PROFILES = {
    'quick': ['--orders','1,3,6,8','--indices','1,8','--samples','5'],
    'throughput': ['--orders','3,6,8','--indices','1,8,32,128','--samples','5'],
    'coverage': ['--orders','1,3,6,8,10,12','--indices','3','--methods','se,pairs,spgf','--samples','3'],
}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def capture(cmd):
    try: return subprocess.check_output(cmd,cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=20).strip()
    except (OSError,subprocess.SubprocessError) as e: return f'unavailable: {e}'

def summarize(raw, destination):
    groups=collections.defaultdict(list)
    for row in csv.DictReader(raw.open()):
        groups[tuple(row[k] for k in ['backend']+KEY)].append(row)
    if not groups: raise RuntimeError('no benchmark rows')
    headers=['backend']+KEY+['samples','first_ms','median_ms','min_ms','max_ms','mad_ms','median_ns_per_index',
                            'setup_ms','context_ms','owned_bytes','result_norm','max_abs_error','max_relative_error','cpu_abs_error','cpu_relative_error','index_cpu_abs_error','index_cpu_relative_error']
    results=[]
    for key,rows in groups.items():
        samples=[r for r in rows if int(r['sample'])>=0]
        first=[r for r in rows if int(r['sample'])==-1]
        ids=sorted(int(r['sample']) for r in samples)
        if len(first)!=1 or ids!=list(range(len(ids))) or not samples:
            raise RuntimeError('missing or duplicate sample IDs')
        times=[float(r['wall_ms']) for r in samples];median=statistics.median(times)
        result=dict(zip(['backend']+KEY,key))
        result.update(samples=len(times),first_ms=float(first[0]['wall_ms']),median_ms=median,
                      min_ms=min(times),max_ms=max(times),mad_ms=statistics.median(abs(x-median) for x in times),
                      median_ns_per_index=median*1e6/int(result['indices']),setup_ms=rows[0]['setup_ms'],
                      context_ms=rows[0]['context_ms'],owned_bytes=max(int(r['owned_bytes']) for r in rows),
                      result_norm=max(float(r['result_norm']) for r in rows),
                      max_abs_error=max(float(r['max_abs_error']) for r in rows),
                      max_relative_error=max(float(r['relative_error']) for r in rows),
                      **{k:max(float(r[k]) for r in rows) for k in ['cpu_abs_error','cpu_relative_error','index_cpu_abs_error','index_cpu_relative_error']})
        results.append(result)
    with destination.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,headers,lineterminator='\n');writer.writeheader();writer.writerows(results)
    return results

def main(default_binary=None, default_extra=(), source_directories=()):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=default_binary,required=default_binary is None)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=PROFILES,default='quick')
    parser.add_argument('--candidate-sources',type=Path,help='Experiment source directory to hash alongside the baseline')
    parser.add_argument('extra',nargs=argparse.REMAINDER)
    args=parser.parse_args();binary=args.binary.resolve();out=args.output.resolve()
    if not binary.is_file(): parser.error(f'binary not found: {binary}')
    out.mkdir(parents=True,exist_ok=False)
    extra=args.extra[1:] if args.extra[:1]==['--'] else args.extra
    cmd=[str(binary)]+PROFILES[args.profile]+list(default_extra)+extra
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sources=[ROOT/'c++/triqs_xca/dense/diagram_evaluator.cpp',ROOT/'c++/triqs_xca/dense/diagram_evaluator.hpp']
    sources += [Path(__file__),ROOT/'benchmarks/dense_common/harness.hpp']
    entry_source=next(iter((ROOT/'benchmarks').glob(f'*/{binary.name}.cpp')), ROOT/'benchmarks'/f'{binary.name}.cpp')
    if entry_source.is_file():
        sources.append(entry_source)
        entry_runner=entry_source.parent/'run.py'
        if entry_runner.is_file() and entry_runner not in sources:
            sources.append(entry_runner)
    for directory in source_directories:
        sources += sorted(p for p in directory.rglob('*') if p.is_file() and
                          (p.suffix in {'.cpp','.hpp','.cu','.cuh','.py'} or p.name=='CMakeLists.txt'))
    if args.candidate_sources:
        directory=args.candidate_sources.resolve()
        if not directory.is_dir() or not directory.is_relative_to(ROOT):
            parser.error('candidate sources must be a directory within the repository')
        sources += sorted(p for p in directory.rglob('*') if p.is_file() and
                          (p.suffix in {'.cpp','.hpp','.cu','.cuh','.py'} or p.name=='CMakeLists.txt'))
    manifest=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,
                  revision=capture(['git','rev-parse','HEAD']),status=capture(['git','status','--short']),
                  binary_sha256=digest(binary),sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                  cpu=platform.processor() or capture(['lscpu']),hostname=platform.node(),platform=platform.platform(),
                  gpu=capture(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
                  libraries=capture(['ldd',str(binary)]),
                  environment={k:env.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TRIQS_ROOT','LD_LIBRARY_PATH',
                                                     'XCA_BATCH_CAPACITY','XCA_BATCH_RANK_MODE','XCA_BATCH_CACHE','XCA_FUSED_MODE','XCA_SOLVE_MODE','XCA_POW2_RANK_LANES','XCA_POW2_SOLVE_LANES','XCA_POW2_SOLVE_GROUP']},
                  profile=args.profile)
    print('Running:', ' '.join(cmd),flush=True)
    with (out/'samples.csv').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    manifest['exit_code']=result.returncode
    manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if result.returncode:
        print((out/'stderr.log').read_text(),file=sys.stderr);return result.returncode
    rows=summarize(out/'samples.csv',out/'summary.csv')
    print(f'Passed {len(rows)} backend/case groups; raw samples, provenance and summary in {out}')
    return 0

if __name__=='__main__': sys.exit(main())

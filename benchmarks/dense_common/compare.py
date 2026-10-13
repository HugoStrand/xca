#!/usr/bin/env python3
"""Compare matching cases from two summary CSVs; speedup = reference/candidate."""
import argparse
import csv
from pathlib import Path
import sys
from runner import KEY

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
p.add_argument('--reference-backend',default='cuda_baseline')
p.add_argument('--candidate-backend',default='cuda_baseline')
a=p.parse_args()
def load(path,backend):
    rows=[r for r in csv.DictReader(path.open()) if r['backend']==backend]
    result={tuple(r[k] for k in KEY):r for r in rows}
    if not rows or len(result)!=len(rows): raise ValueError('missing backend or duplicate cases')
    return result
ref,new=load(a.reference,a.reference_backend),load(a.candidate,a.candidate_backend)
if ref.keys()!=new.keys(): raise ValueError('case sets differ; select identical workloads for comparison')
w=csv.writer(sys.stdout,lineterminator='\n')
w.writerow(['N','r','order','topology','method','indices','reference_ms','candidate_ms','speedup','reference_mad_ms','candidate_mad_ms'])
for key,r in ref.items():
    n=new[key];rt,nt=float(r['median_ms']),float(n['median_ms'])
    w.writerow([r[k] for k in ['N','r','order','topology','method','indices']]+[rt,nt,rt/nt,r['mad_ms'],n['mad_ms']])

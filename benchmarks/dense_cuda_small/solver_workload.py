#!/usr/bin/env python3
"""Serial real Solver workload control; fit, self-consistency, SPGF and density observable.

CUDA backends are timed serially, with fresh Solver objects. GPU speedup is not
assumed for these short workloads. Use the CUDA environment for all three backends.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
from triqs.gfs import Gf
from triqs.operators import n, c, c_dag
import triqs.utility.mpi as mpi
from triqs_xca import Solver
from triqs_xca._backend_config import CUDA_ENABLED, CUDA_SMALL_ENABLED


def make_solver(backend, dynamic):
    H=-.2*(n('0',0)+n('0',1))+.4*n('0',0)*n('0',1)
    S=Solver(H,2.,5.,1e-12,[['0',2]],conserved_operators=[],verbose=False,dense_backend=backend)
    coefficients=np.zeros((2,2,2),complex)
    coefficients[0]=np.eye(2)*.3; coefficients[1]=np.eye(2)*.2
    S.set_hybridization_poles_and_coefficients(np.array([-1.5,.6]),coefficients)
    if dynamic:
        kernel=Gf(mesh=S.mesh_tau,target_shape=[2,2])
        for tau in S.mesh_tau:
            value=float(tau.value)
            kernel[tau]=np.cosh(value-1.)/np.sinh(1.)*np.array([[.005,.001j],[-.001j,.007]])
        S.set_dynamic_interactions([n('0',0),c_dag('0',0)*c('0',1)+c_dag('0',1)*c('0',0)],kernel)
    return S


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=3)
    a=p.parse_args()
    if mpi.size!=1: p.error('serial timing control; use backend integration tests for MPI')
    if a.samples<1: p.error('samples must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    backends=['cpu']+(['cuda'] if CUDA_ENABLED else [])+(['cuda_small'] if CUDA_SMALL_ENABLED else [])
    references={};rows=[]
    with (a.output/'solver.log').open('w') as log,contextlib.redirect_stdout(log):
        for dynamic in [False,True]:
            for backend in backends:
                for sample in range(a.samples):
                    begin=time.perf_counter();S=make_solver(backend,dynamic);setup_ms=(time.perf_counter()-begin)*1000
                    begin=time.perf_counter()
                    S.solve(max_order=2,spgf_max_order=2,maxiter=2,tol=1e-8,verbose=False,hyb_comp=False)
                    solve_ms=(time.perf_counter()-begin)*1000
                    begin=time.perf_counter()
                    chi=S.eval_one_time_correlator(S.G,max_order=2,ops_tau=[n('0',0)],ops_0=[n('0',1)])
                    observable_ms=(time.perf_counter()-begin)*1000
                    values=[S.G[0].data.copy(),S.Sigma[0].data.copy(),S.G_tau['0'].data.copy(),chi.data.copy()]
                    assert all(np.isfinite(x).all() for x in values)
                    assert max(np.max(np.abs(x)) for x in values)>0
                    if backend=='cpu' and sample==0: references[dynamic]=values
                    for actual,expected in zip(values,references[dynamic]):
                        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)
                    rows.append(dict(backend=backend,dynamic=dynamic,sample=sample,setup_ms=setup_ms,solve_ms=solve_ms,
                                     observable_ms=observable_ms,total_ms=setup_ms+solve_ms+observable_ms,
                                     ranks=mpi.size,dlr_rank=len(S.mesh_tau),poles=len(S.hyb.poles),
                                     channels=len(S.dynint_ops) if dynamic else 0))
    with (a.output/'samples.csv').open('w') as f:
        w=csv.DictWriter(f,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    meta=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              status=subprocess.check_output(['git','status','--short'],text=True),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              target='RTX A4000 local control; GH200-class optimization target unqualified',
              numerical='real Solver comparison rtol1e-9 atol1e-12; C++ optimization gate unchanged',
              samples=a.samples,backends=backends)
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    for dynamic in [False,True]:
        for backend in backends:
            times=[r['total_ms'] for r in rows if r['dynamic']==dynamic and r['backend']==backend]
            print(backend,'dynamic' if dynamic else 'ordinary','median total ms',round(statistics.median(times),3))
    return 0

if __name__=='__main__': raise SystemExit(main())

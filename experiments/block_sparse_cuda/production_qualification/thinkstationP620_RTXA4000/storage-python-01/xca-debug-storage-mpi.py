import importlib.util,numpy as np
from triqs_xca.solver import scatter_range_bounds
from triqs.utility import mpi
spec=importlib.util.spec_from_file_location('fixture','/home/hugstr/dev/xca/test/python/block_sparse_cuda.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
R=f.make_solver('cuda','occupation');C=f.make_solver('cuda','occupation',storage_policy='host_inputs')
R.init_diagram_evaluator();C.init_diagram_evaluator();top=np.array([[0,1]],np.int32)
for label,S in [('R',R),('C',C)]:
 print('INPUT',mpi.rank,label,[(str(b),float(g.data.real.sum()))for b,g in S.G0],flush=True)
 count=S.d.get_num_self_energy_backbones(top);first,last=scatter_range_bounds(count)
 local=S.d.compute_self_energy_range(S.G0,top,first,last)
 print('LOCAL',mpi.rank,label,count,first,last,[(str(b),complex(g.data[0,0,0]))for b,g in local],flush=True)
 full=S.d.compute_self_energy_range(S.G0,top,0,count)
 print('FULL',mpi.rank,label,[(str(b),complex(g.data[0,0,0]))for b,g in full],flush=True)
 summed=S._Solver__eval_pseudo_particle_self_energy_topology_loop(S.G0,top)
 print('REDUCED',mpi.rank,label,[(str(b),complex(g.data[0,0,0]))for b,g in summed],flush=True)

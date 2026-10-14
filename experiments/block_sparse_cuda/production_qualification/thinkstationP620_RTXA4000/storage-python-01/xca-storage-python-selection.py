import importlib.util,json
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('storage_fixture','/home/hugstr/dev/xca/test/python/block_sparse_cuda.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
top=np.array([[0,2],[1,3]],np.int32)
records=[]
for partition in ('occupation','number'):
 for dynamic in (False,True):
  bounds={}
  for policy in ('resident','host_inputs','stream_fragments'):
   solver=fixture.make_solver('cuda',partition,dynamic,storage_policy=policy);solver.init_diagram_evaluator()
   bounds[policy]=(solver.d.self_energy_storage_bound(top),solver.d.correlator_storage_bound(2,2))
   records.append(dict(partition=partition,dynamic=dynamic,policy=policy,se_bytes=bounds[policy][0],correlator_bytes=bounds[policy][1]))
  assert bounds['host_inputs'][0]<bounds['resident'][0]
  assert bounds['stream_fragments'][0]==bounds['host_inputs'][0]
  assert bounds['stream_fragments'][1]<bounds['host_inputs'][1]<bounds['resident'][1]
print(json.dumps(records,indent=2))

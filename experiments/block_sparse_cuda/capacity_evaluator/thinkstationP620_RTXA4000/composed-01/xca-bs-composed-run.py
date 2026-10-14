import subprocess,json
from pathlib import Path
exe='cbuild-cuda/benchmarks/experiments/block_sparse_cuda/capacity_evaluator/block_sparse_capacity_evaluator'
cases=[('n1', ['--modes','1','--samples','2'],0),('n4-single',['--modes','4','--samples','2'],0),('n4-all',['--modes','4','--fields','4','--samples','2'],0),('n10-single',['--modes','10','--lambda','4000','--samples','2'],0),('n10-all',['--modes','10','--fields','10','--lambda','4000','--samples','2'],0),('n14-inventory',['--modes','14','--lambda','4000','--dry-run'],0),('host-reject',['--host-budget-mib','1'],2),('device-reject',['--device-budget-mib','1'],2),('large-reject',['--modes','14','--lambda','4000','--device-budget-mib','98304','--host-budget-mib','106496','--allow-large'],1),('suffix-reject',['--modes','4junk'],1)]
records=[]
for name,args,expected in cases:
 r=subprocess.run([exe,*args],capture_output=True,text=True)
 Path('/tmp/xca-bs-composed-'+name+'.csv').write_text(r.stdout)
 Path('/tmp/xca-bs-composed-'+name+'.log').write_text(r.stderr)
 records.append({'case':name,'argv':[exe,*args],'returncode':r.returncode,'expected':expected})
 print(name,r.returncode,flush=True)
 if r.returncode!=expected: raise RuntimeError((name,r.stderr))
Path('/tmp/xca-bs-composed-cases.json').write_text(json.dumps(records,indent=2)+'\n')

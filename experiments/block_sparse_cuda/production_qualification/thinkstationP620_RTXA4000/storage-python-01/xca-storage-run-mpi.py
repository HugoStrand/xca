import json,os,subprocess
from pathlib import Path
config=json.loads(Path('/tmp/xca-storage-python-ctest.json').read_text())
test=next(x for x in config['tests']if x['name']=='Py_block_sparse_cuda')
props={x['name']:x['value']for x in test['properties']}
env=os.environ.copy()
for entry in props.get('ENVIRONMENT',[]):
 key,sep,value=entry.partition('=')
 if sep:env[key]=value
for entry in props.get('ENVIRONMENT_MODIFICATION',[]):
 key,change=entry.split('=',1)
 operation,value=change.split(':',1)
 if operation=='path_list_prepend':env[key]=value+(os.pathsep+env[key]if env.get(key)else'')
 elif operation=='path_list_append':env[key]=(env[key]+os.pathsep if env.get(key)else'')+value
 elif operation=='set':env[key]=value
 elif operation=='unset':env.pop(key,None)
 else:raise RuntimeError('Unimplemented CTest environment operation '+operation)
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=['mpirun','--oversubscribe','-n','2']+test['command']+__import__('sys').argv[1:]
Path('/tmp/xca-storage-python-mpi-command.json').write_text(json.dumps({'command':command,'working_directory':props.get('WORKING_DIRECTORY'),'ctest_environment':props.get('ENVIRONMENT',[]),'ctest_environment_modification':props.get('ENVIRONMENT_MODIFICATION',[]),'resolved_PYTHONPATH':env.get('PYTHONPATH')},indent=2)+'\n')
result=subprocess.run(command,cwd=props.get('WORKING_DIRECTORY'),env=env)
raise SystemExit(result.returncode)

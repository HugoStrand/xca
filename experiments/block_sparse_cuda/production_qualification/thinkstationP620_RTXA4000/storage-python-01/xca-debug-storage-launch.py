import json,os,subprocess
from pathlib import Path
meta=json.loads(Path('/tmp/xca-storage-python-mpi-command.json').read_text());env=os.environ.copy();env['PYTHONPATH']=meta['resolved_PYTHONPATH'];env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
raise SystemExit(subprocess.run(['mpirun','--oversubscribe','-n','2','/usr/bin/python3','/tmp/xca-debug-storage-mpi.py'],env=env).returncode)

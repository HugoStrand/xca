from pathlib import Path
import subprocess,csv,io,json,hashlib,os,statistics
root=Path('/home/hugstr/dev/xca'); dest=root/'experiments/block_sparse_cuda/convolution_rows/thinkstationP620_RTXA4000/evaluator-01';dest.mkdir(parents=True,exist_ok=True)
build=root/'cbuild-cuda/benchmarks'
binaries={'reference':build/'block_sparse_cuda/block_sparse_cuda_benchmark','rows':build/'experiments/block_sparse_cuda/convolution_rows/block_sparse_row_evaluator_benchmark'}
manifest={'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'binaries':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for k,p in binaries.items()},'cases':[],'policy':'sequential GPU runs, one OMP/BLAS thread, warm complete-call fixture, 3 timing samples; allocation observer separately 1 sample'}
allrows=[];summary=[]
cases=[('n4-number-oca',4,'number','se',2,1),('n6-number-oca',6,'number','se',2,4),('n8-number-oca',8,'number','se',2,8),('n6-mixed-spgf',6,'mixed','spgf',2,4)]
for name,modes,partition,observable,order,cap in cases:
    group=[]
    for route in ('reference','rows','rows_sampled'):
        binary=binaries['reference' if route=='reference' else 'rows']
        cmd=[str(binary),'--modes',str(modes),'--partition',partition,'--observable',observable,'--order',str(order),'--first','0','--count','1','--samples','1' if route=='rows_sampled' else '3']
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XCA_EXPERIMENT_CONV_ROWS=str(cap))
        if route=='rows_sampled':env['XCA_EXPERIMENT_SAMPLE_ROWS']='1'
        else:env.pop('XCA_EXPERIMENT_SAMPLE_ROWS',None)
        proc=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=120)
        (dest/f'{name}-{route}.csv').write_text(proc.stdout);(dest/f'{name}-{route}.log').write_text(proc.stderr)
        manifest['cases'].append({'name':name,'route':route,'row_cap':cap,'command':cmd,'returncode':proc.returncode})
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if proc.returncode:raise RuntimeError(f'{name} {route} failed')
        data=list(csv.DictReader(io.StringIO(proc.stdout)))
        assert len(data)==(1 if route=='rows_sampled' else 3)
        for row in data:row.update(case=name,route=route,row_cap=str(cap))
        group+=data;allrows+=data
        summary.append({'case':name,'route':route,'row_cap':cap,'samples':len(data),'median_ms':statistics.median(float(x['call_ms'])for x in data),'bound_bytes':data[0]['device_bound_bytes'],'sampled_peak_delta':max(int(x['device_peak_delta'])for x in data),'max_abs_error':max(float(x['max_abs_error'])for x in data)})
    # Structural counts must agree at these ordinary amplitudes; cutoff audit separate.
    for key in ('proposed','accepted','pruned'):assert len({x[key]for x in group})==1,(name,key)
    print(name,[(s['route'],s['median_ms'],s['bound_bytes'],s['sampled_peak_delta']) for s in summary if s['case']==name],flush=True)
for filename,data in [('samples.csv',allrows),('summary.csv',summary)]:
    with(dest/filename).open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
print('recorded',len(allrows),'rows',flush=True)

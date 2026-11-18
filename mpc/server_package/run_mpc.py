"""Portable, restartable MPC replay. No simulation occurs with --dry-run."""
from pathlib import Path
import os,sys,json,argparse,hashlib,platform,time,datetime,contextlib,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent
def load_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,data):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def task_list(stage,suite):
    prot=load_json(ROOT/'RUN_PROTOCOL.json');ctrls=prot['core_controllers'] if suite=='core' else prot['all_controllers'];out=[]
    for cohort in ('A','B'):
        if stage=='smoke' and cohort=='B':continue
        q=load_json(ROOT/'inputs'/f'scenario_{cohort}.json')
        indices=[0] if stage=='smoke' else [0,5,11] if stage=='timing' else list(range(12))
        for i in indices:
            for ctrl in ctrls:
                sc=q['scenarios'][i];out.append({'key':f'{cohort}_{sc["idx"]}_{ctrl}','cohort':cohort,'scenario':sc,'ctrl':ctrl,
                  'Tp':int(ctrl.rsplit('_',1)[1]) if '_' in ctrl else 0,'disturbance':q['disturbance'],'reference_file':q['reference_file']})
    return out
def worker_init():
    sys.path.insert(0,str(ROOT/'code'))
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.set_default_device('cpu');threadpool_limits(1)
    # Import all numerical code before the per-case clocks begin.
    import mpc_instrumented,y0_turnpike_surrogate,turnpike_surrogate
def work(task,outdir,batch_id):
    import numpy as np,psutil
    import mpc_instrumented as mpc
    from threadpoolctl import threadpool_info
    out=Path(outdir);key=task['key'];prot=load_json(ROOT/'RUN_PROTOCOL.json')
    try:
        with (out/'logs'/(key+'.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            print('START',key,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
            start=time.perf_counter()
            r=mpc.run((task['scenario'],task['ctrl'],task['Tp'],str(ROOT/prot['model']),task['disturbance']))
            wall=time.perf_counter()-start
        original=next(a for a in load_json(ROOT/task['reference_file'])['runs'] if a['scenario']==task['scenario']['idx'] and a['ctrl']==task['ctrl'])
        u=np.asarray(r['u'],dtype=float);ref=np.asarray(original['u'],dtype=float)
        error=float(np.linalg.norm(u-ref)/np.linalg.norm(ref));jerror=abs(r['J']/original['J']-1)
        residuals=r['solver_residuals'];resmax=max(residuals) if residuals else None
        finite=bool(np.isfinite(u).all() and np.isfinite(r['J']) and np.isfinite(r['controller_seconds']).all())
        limits=prot['verification'];passed=(len(u)==2560 and len(r['controller_seconds'])==80 and finite and error<limits['relative_control_difference_max']
            and jerror<limits['relative_cost_difference_max'] and (resmax is None or resmax<limits['solver_residual_max']))
        r.update({'key':key,'cohort':task['cohort'],'batch_id':batch_id,'total_scenario_wall_seconds':wall,
          'relative_control_difference':error,'relative_cost_difference':jerror,'max_solver_residual':resmax,'passed':bool(passed),
          'rss_after_case_bytes':psutil.Process().memory_info().rss,'threadpools':threadpool_info(),
          'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        write_json(out/'runs'/(key+'.json'),r)
        return {'key':key,'passed':bool(passed),'wall_seconds':wall,'relative_control_difference':error,'relative_cost_difference':jerror}
    except Exception:
        failure={'key':key,'passed':False,'traceback':traceback.format_exc(),'batch_id':batch_id}
        write_json(out/'failures'/(key+'.json'),failure);return failure
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['smoke','full','timing'],default='full');ap.add_argument('--suite',choices=['core','all'],default='core')
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--out',type=Path,default=None);ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    if args.stage in ('smoke','timing') and args.workers!=1:ap.error('smoke and timing require --workers 1')
    from check_package import verify_manifest
    nfiles=verify_manifest();tasks=task_list(args.stage,args.suite)
    if args.dry_run:
        print(json.dumps({'simulation_executed':False,'verified_files':nfiles,'stage':args.stage,'suite':args.suite,'workers':args.workers,
                         'trajectories':len(tasks),'controller_calls':len(tasks)*80,'keys':[x['key'] for x in tasks]},indent=2));return
    import psutil,numpy,scipy,torch,threadpoolctl
    out=(args.out or ROOT/'results'/args.stage).resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ('runs','logs','failures','invocations'):(out/name).mkdir(exist_ok=True)
    digest=hashlib.sha256((ROOT/'MANIFEST_SHA256.json').read_bytes()).hexdigest()
    cpu_model=platform.processor()
    if Path('/proc/cpuinfo').is_file():
        cpu_model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),cpu_model)
    hardware={'hostname':platform.node(),'platform':platform.platform(),'processor':cpu_model,'architecture':platform.machine(),'logical_cpus':os.cpu_count(),
       'physical_cpus':psutil.cpu_count(logical=False),'memory_bytes':psutil.virtual_memory().total,'python':platform.python_version(),
       'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'threadpoolctl':threadpoolctl.__version__,
       'cpu_affinity':psutil.Process().cpu_affinity() if hasattr(psutil.Process(),'cpu_affinity') else None,
       'slurm_cpus_per_task':os.environ.get('SLURM_CPUS_PER_TASK')}
    config={'stage':args.stage,'suite':args.suite,'workers':args.workers,'threads_per_worker':1,'manifest_sha256':digest,'hardware':hardware,
       'expected_keys':[x['key'] for x in tasks],'timing_scope':'isolated-task' if args.workers==1 else 'shared-server-parallel'}
    if (out/'run_config.json').exists():
        old=load_json(out/'run_config.json')
        if old!=config:raise RuntimeError('Output folder configuration/hardware differs; use a new --out folder.')
    else:write_json(out/'run_config.json',config)
    done={}
    for t in tasks:
        p=out/'runs'/(t['key']+'.json')
        if p.exists():
            q=load_json(p)
            if q.get('passed'):done[t['key']]=q
    pending=[t for t in tasks if t['key'] not in done]
    batch=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');start=time.perf_counter();results=[]
    print(f'{len(done)} passed cases reused; {len(pending)} pending; {args.workers} worker(s)',flush=True)
    if args.workers==1:
        worker_init()
        for t in pending:
            r=work(t,str(out),batch);results.append(r);print(json.dumps(r),flush=True)
            if not r['passed'] and args.stage=='smoke':break
    else:
        ctx=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=worker_init) as pool:
            futures={pool.submit(work,t,str(out),batch):t['key'] for t in pending}
            for f in as_completed(futures):
                try:r=f.result()
                except Exception:r={'key':futures[f],'passed':False,'traceback':traceback.format_exc()}
                results.append(r);print(json.dumps(r),flush=True)
    receipt={'batch_id':batch,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'elapsed_wall_seconds':time.perf_counter()-start,'reused_passed_cases':len(done),'attempted':results}
    write_json(out/'invocations'/(batch+'.json'),receipt)
    from summarize_results import summarize
    summary=summarize(out);print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))
    if not summary['passed']:sys.exit(2)
if __name__=='__main__':main()

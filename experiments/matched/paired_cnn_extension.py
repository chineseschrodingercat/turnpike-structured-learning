from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,time,json
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import numpy as np,torch
ROOT = output_directory("matched");RAW = RAW_DATA;sys.path.insert(0,str(CODE_DIR))
import baselines_multihorizon as bm,fresh_test_1d as ft
from turnpike_surrogate import grid
from robin_solver import RobinConfig,assemble_operators,assemble_problem_data,solve_state,eval_cost
torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
r=json.loads((ROOT/'paired_cost_accuracy.json').read_text());d=np.load(RAW/'01 - Initial delivery/runs/turnpike/turnpike_data.npz');saved=np.load(ROOT/'paired_cost_accuracy_controls.npz');controls={k:saved[k] for k in saved.files};t,s=grid(8,1/320);one=lambda x,t:np.ones_like(np.asarray(x,float))
for name,maker in [('E19A_CNN',bm.DilatedCNN),('E19A_residual_CNN',ft.ResCNN)]:
 ck=torch.load(ROOT/f'NEW_{name}_seed0.pt',map_location='cpu',weights_only=False);net=maker(d['X'].shape[1]);net.load_state_dict(ck['state']);net.eval();mean=ck['normalizer']['mean'];std=ck['normalizer']['std']
 for row in r['rows']:
  i=row['sample_index'];x=torch.tensor(((d['X'][i]-mean)/std)[None,:],dtype=torch.float32);yd=torch.ones(1,2560);refsec=row['routes']['trained_structured_seed0']['reference_seconds_repeats'];ub=d['u_bar'][i];ts=[]
  with torch.no_grad():
   net(x,yd,t,s)
   for n in range(2):
    start=time.perf_counter();u=ub*net(x,yd,t,s)[0].numpy();ts.append(time.perf_counter()-start)
  truth=d['U_T8'][i];cfg=RobinConfig(T=8.,Nt=2560,**row['parameters']);ops=assemble_operators(cfg);dat=assemble_problem_data(cfg,ops,one);J=eval_cost(cfg,ops,dat,solve_state(ops,dat,u),u)[0];J0=row['routes']['monolithic_KKT']['objective'];times=[a+b for a,b in zip(refsec,ts)];key='trained_'+name
  row['routes'][key]={'seconds_repeats':times,'seconds_median':float(np.median(times)),'cached_query_seconds_repeats':ts,'reference_seconds_repeats':refsec,'relative_error':float(np.linalg.norm(u-truth)/np.linalg.norm(truth)),'objective':J,'relative_objective_difference':float((J-J0)/J0),'identity':f'NEW_{name}_20261116_seed0','precision':'float32 CPU query; weights same as newly retrained CUDA model'};controls[f'row{i}_{key}']=u.copy()
r['aggregate']={name:{'median_seconds':float(np.median([x['routes'][name]['seconds_median'] for x in r['rows']])),'median_relative_error':float(np.median([x['routes'][name]['relative_error'] for x in r['rows']])),'max_relative_error':float(max(x['routes'][name]['relative_error'] for x in r['rows'])),'max_relative_objective_difference':float(max(x['routes'][name]['relative_objective_difference'] for x in r['rows']))} for name in r['rows'][0]['routes']}
np.savez_compressed(ROOT/'paired_cost_accuracy_controls.npz',**controls);(ROOT/'paired_cost_accuracy.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in r['aggregate'].items() if 'CNN' in k}))

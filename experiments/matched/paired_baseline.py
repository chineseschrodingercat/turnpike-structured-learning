from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,time,json,hashlib,platform
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import numpy as np
import scipy,scipy.sparse.linalg as spla
import torch
ROOT = output_directory("matched");RAW = RAW_DATA;CODE=CODE_DIR;sys.path.insert(0,str(CODE))
from hamiltonian_exact import build,control,particular,pencil_modes
from robin_solver import RobinConfig,solve_kkt,assemble_operators,assemble_problem_data,reduced_system,reduced_control,solve_state,eval_cost
from turnpike_surrogate import TurnpikeNet,first_eig,grid
import fresh_test_1d as ft
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
ONE=lambda x,t:np.ones_like(np.asarray(x,float))
def dump(n,o):(ROOT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def measure(f,reps=2):
 times=[];out=None
 for _ in range(reps):
  start=time.perf_counter();out=f();times.append(time.perf_counter()-start)
 return out,times
def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
dpath=RAW/'01 - Initial delivery/runs/turnpike/turnpike_data.npz';d=np.load(dpath);te=d['test_idx'];picks=te[[0,len(te)//2,len(te)-1]]
ck=torch.load(ROOT/'NEW_E19A_structured_seed0.pt',map_location='cpu',weights_only=False);net=TurnpikeNet(r=4,shared=True,use_floor=True,n_in=d['X'].shape[1]);net.load_state_dict({k.removeprefix('net.'):v for k,v in ck['state'].items()});net.eval()
ks=np.load(ROOT/'krr_reconstructed_state.npz');krr=ft.KRR();krr.Xtr=ks['Xtr'];krr.alpha=ks['alpha'];krr.mu=ks['mu'];krr.sd=ks['sd'];krr.ell=float(ks['ell'])
mean=ck['normalizer']['mean'];std=ck['normalizer']['std'];dt=float(d['dt']);t,s=grid(8,dt)
report={'identity':'NEW_PAIRED_1D_CONST_20261116','protocol':{'T':8,'dt':dt,'nx':160,'sample_selection':'first, middle and last entries of immutable original test_idx; fixed before timing','sample_indices':picks.tolist(),'repeats':2,'warmup':'trained net once; no solver warmup; repetitions retain allocations/caches from process','scope':'wall clock; each fresh-mu route includes its own setup and required reference; cached-query costs separate; state propagation for objective QA excluded from solver timing','threads':1,'historical_checkpoint_recovered':False,'learned_identity':ck['identity'],'learned_selection':'6000 epochs; validation T=1 only; original configuration; new single seed0','error':'Euclidean coefficient relative L2; scalar control so equal to consistent mass norm','historical_offline_label_generation_seconds':None},'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'processor':platform.processor(),'platform':platform.platform()},'rows':[]}
controls={};truth_all=d['U_T8']
for i in picks:
 i=int(i);geo={k:float(d[k][i]) for k in ['alpha','beta','xa','xb']};rawx=d['X'][i];truth=truth_all[i];cfg=RobinConfig(T=8.,Nt=2560,**geo)
 row={'sample_index':i,'parameters':geo,'X':rawx.tolist(),'routes':{}}
 def add(name,u,times,**kw):
  row['routes'][name]={'seconds_repeats':times,'seconds_median':float(np.median(times)),'relative_error':rel(u,truth),**kw};controls[f'row{i}_{name}']=u.copy()
 kkt,kt=measure(lambda:solve_kkt(cfg,ONE));add('monolithic_KKT',kkt.U,kt,kkt_residual=kkt.rel_res)
 def setup():
  cc,ops,G=build(geo);return pencil_modes(ops,geo['alpha']),particular(ops,G,geo['alpha'],0.),cc.dt
 modes_parts,st=measure(setup);mo,pa,dd=modes_parts
 u,qt=measure(lambda:control(mo,pa,geo['alpha'],0.,8,dd,True).real)
 add('pencil_exact',u,[a+b for a,b in zip(st,qt)],setup_seconds_repeats=st,cached_query_seconds_repeats=qt)
 u,lt=measure(lambda:control(mo,pa,geo['alpha'],0.,8,dd,False).real);add('ARE_independent_layers',u,[a+b for a,b in zip(st,lt)],setup_seconds_repeats=st,cached_query_seconds_repeats=lt)
 def reduced():
  oo=assemble_operators(cfg);data=assemble_problem_data(cfg,oo,ONE);return reduced_control(reduced_system(oo,data),geo['alpha'])
 u,rt=measure(reduced);add('reduced_Hessian',u,rt)
 def reference():
  cc,oo,G=build(geo);psi=spla.splu(oo.K_beta.tocsc()).solve(oo.e0);return float(psi@G/(psi@(oo.M_obs@psi)+geo['alpha']))
 ub,ref_t=measure(reference);fl,floor_t=measure(lambda:first_eig(geo['beta']))
 for shortT in [1,2]:
  short,short_t=measure(lambda:solve_kkt(RobinConfig(T=float(shortT),Nt=320*shortT,**geo),ONE))
  u=np.full(2560,ub);n=160*shortT;u[:n]=short.U[:n];u[-n:]=short.U[-n:]
  add(f'stitch_T{shortT}',u,[a+b for a,b in zip(ref_t,short_t)],reference_seconds_repeats=ref_t,short_KKT_seconds_repeats=short_t,splice_half_steps=n)
 xn=(rawx-mean)/std;xt=torch.tensor(xn[None,:]);flt=torch.tensor([fl])
 with torch.no_grad():
  net(xt,t,s,flt)
  u,nt=measure(lambda:ub*net(xt,t,s,flt)[0].numpy())
 add('trained_structured_seed0',u,[a+b+c for a,b,c in zip(ref_t,floor_t,nt)],reference_seconds_repeats=ref_t,rate_floor_seconds_repeats=floor_t,cached_query_seconds_repeats=nt,normalization_scope='mean/std fitted train_idx only; negligible cached array arithmetic excluded')
 u,kqt=measure(lambda:ub*krr.predict(xn[None,:],np.array([fl]),8,dt)[0]);add('KRR_reconstructed',u,[a+b+c for a,b,c in zip(ref_t,floor_t,kqt)],reference_seconds_repeats=ref_t,rate_floor_seconds_repeats=floor_t,cached_query_seconds_repeats=kqt)
 for name,route in row['routes'].items():
  u=controls[f'row{i}_{name}'];Y=solve_state(kkt.operators,kkt.data,u);J=eval_cost(cfg,kkt.operators,kkt.data,Y,u)[0];route['objective']=J;route['relative_objective_difference']=(J-kkt.J_total)/kkt.J_total
 controls[f'row{i}_truth']=truth;report['rows'].append(row);dump('paired_cost_accuracy.json',report);print('Paired case complete',i,flush=True)
report['aggregate']={name:{'median_seconds':float(np.median([r['routes'][name]['seconds_median'] for r in report['rows']])),'median_relative_error':float(np.median([r['routes'][name]['relative_error'] for r in report['rows']])),'max_relative_error':float(max(r['routes'][name]['relative_error'] for r in report['rows'])),'max_relative_objective_difference':float(max(r['routes'][name]['relative_objective_difference'] for r in report['rows']))} for name in report['rows'][0]['routes']}
np.savez_compressed(ROOT/'paired_cost_accuracy_controls.npz',**controls,sample_indices=picks,dt=dt)
dump('paired_cost_accuracy.json',report);print(json.dumps(report['aggregate']),flush=True)

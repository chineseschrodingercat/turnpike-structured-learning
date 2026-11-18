from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,json,time,hashlib,platform,contextlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import numpy as np
import torch
ROOT = output_directory("matched"); RAW = RAW_DATA
CODE=CODE_DIR;sys.path.insert(0,str(CODE))
import baselines_multihorizon as bm
import fresh_test_1d as ft
from turnpike_surrogate import first_eig
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
def dump(n,o): (ROOT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def summ(a):return {k:float(f(a)) for k,f in [('median',np.median),('max',np.max)]}
data_path=RAW/'01 - Initial delivery/runs/turnpike/turnpike_data.npz'
fresh_path=RAW/'01 - Initial delivery/runs/fresh1d/fresh_const.npz'
d=np.load(data_path);f=np.load(fresh_path);X=d['X'];tr=d['train_idx'];te=d['test_idx'];dt=float(d['dt'])
mean=X[tr].mean(0);std=X[tr].std(0);xn=(X-mean)/std;xf=(f['X']-mean)/std
floors=np.array([first_eig(b) for b in d['beta']]);floorf=np.array([first_eig(b) for b in f['beta']]);Ts=[int(t) for t in f['T_list']]
info=json.loads((RAW/'01 - Initial delivery/runs/fresh1d/fresh_krr.json').read_text())['krr']['info']
start=time.perf_counter();C=ft.fit_coefs(d['U_T1'][tr]/d['u_bar'][tr,None],floors[tr],dt,info['eps'])
krr=ft.KRR().fit(xn[tr],C,info['ell'],info['lam']);fitsec=time.perf_counter()-start
np.savez_compressed(ROOT/'krr_reconstructed_state.npz',Xtr=krr.Xtr,alpha=krr.alpha,mu=krr.mu,sd=krr.sd,ell=krr.ell,lam=info['lam'],eps=info['eps'],mean=mean,std=std,train_idx=tr)
pred={T:krr.predict(xf,floorf,T,dt)*f['u_bar'][:,None] for T in Ts}
orig={T:krr.predict(xn[te],floors[te],T,dt)*d['u_bar'][te,None] for T in [1,2,4,8]}
np.savez_compressed(ROOT/'fresh_krr_replay_predictions.npz',**{f'U_T{T}':v for T,v in pred.items()},sample_idx=np.arange(len(xf)),X=f['X'],dt=dt,source_sha256=np.array(hashlib.sha256(fresh_path.read_bytes()).hexdigest()))
rep={'provenance':'new deterministic fit with historical selected hyperparameters; not recovered checkpoint','krr':{'hyperparameters':info,'fit_seconds_excludes_hyperparameter_search_and_label_generation':fitsec,'fresh':ft.summarize_rows(ft.window_rows(pred,{T:f[f'U_T{T}'] for T in Ts})),'original_test':{str(T):summ(np.linalg.norm(orig[T]-d[f'U_T{T}'][te],axis=1)/np.linalg.norm(d[f'U_T{T}'][te],axis=1)) for T in orig}},'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'processor':platform.processor(),'cuda_device':torch.cuda.get_device_name(0),'threads':4},'failures':['First inventory attempt used removed NumPy private header reader; replaced with public version-specific header readers and reran successfully.']}
old=json.loads((RAW/'01 - Initial delivery/runs/fresh1d/fresh_krr.json').read_text())['krr']['fresh']
rep['krr']['max_abs_difference_from_retained_summary']=max(abs(rep['krr']['fresh'][str(T)][metric][s]-old[str(T)][metric][s]) for T in Ts for metric in ['total_rel','init_rel','term_rel'] for s in ['median','p90','max'])
dump('reconstruction_results.json',rep);print('KRR reconstructed',flush=True)

# Full new single-seed replay; retain identity rather than claiming historical recovery.
p=bm.load_problem('const',data_path);torch.manual_seed(0)
start=time.perf_counter();model=bm.TurnpikeConst(p,r=4,floor='lam1');model=bm.train(model,p,[1],6000,2e-3,700);training_seconds=time.perf_counter()-start
ck={'state':model.state_dict(),'seed':0,'identity':'NEW_RETRAIN_20261116_E19A_structured_seed0','r':4,'train_T':[1],'epochs':6000,'lr':2e-3,'batch':700,'dtype':'float64','threads':4,'normalizer':{'mean':mean,'std':std},'train_idx':tr,'val_idx':d['val_idx'],'test_idx':te,'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'training_seconds':training_seconds}
torch.save(ck,ROOT/'NEW_E19A_structured_seed0.pt')
fresh=ft.fresh_problem(fresh_path,p,X,tr)
model.eval()
with torch.no_grad():
 pred={T:ft.predict_fresh(model,fresh,T,dt) for T in Ts}
 orig={T:bm.predict(model,p,p['te'],T).numpy()*d['u_bar'][te,None] for T in [1,2,4,8]}
np.savez_compressed(ROOT/'NEW_fresh_structured_seed0_predictions.npz',**{f'U_T{T}':v for T,v in pred.items()},sample_idx=np.arange(len(xf)),X=f['X'],dt=dt)
np.savez_compressed(ROOT/'NEW_original_structured_seed0_predictions.npz',**{f'U_T{T}':v for T,v in orig.items()},test_idx=te,X=X[te],dt=dt)
rep['new_structured']={'identity':ck['identity'],'training_seconds':training_seconds,'fresh':ft.summarize_rows(ft.window_rows(pred,{T:f[f'U_T{T}'] for T in Ts})),'original_test':{str(T):summ(np.linalg.norm(orig[T]-d[f'U_T{T}'][te],axis=1)/np.linalg.norm(d[f'U_T{T}'][te],axis=1)) for T in orig},'historical_checkpoint_recovered':False,'settings':{k:ck[k] for k in ['r','train_T','epochs','lr','batch','dtype','threads']}}
dump('reconstruction_results.json',rep);print('Full structured retrain complete',training_seconds,flush=True)

# Bounded actual forward/backward probes for missing final E04 / E07 implementations.
probes=[]
import box_turnpike_surrogate as bx
bd=np.load(RAW/'02 - Supplement/runs/turnpike_box/box_merged_n3400.npz');ii=bd['train_idx'][:8];xx=bd['X'];btr=bd['train_idx'];ub=bd['u_bar'][ii]
bp={'x':torch.tensor((xx[ii]-xx[btr].mean(0))/xx[btr].std(0)),'lo':torch.tensor(xx[ii,4]),'hi':torch.tensor(xx[ii,5]),'qbar':torch.tensor(bd['q_bar'][ii]/ub),'uc':torch.tensor(bd['u_c'][ii]/ub),'floor':torch.tensor([first_eig(b) for b in bd['beta'][ii]]),'dt':float(bd['dt']),'U1':torch.tensor(bd['U_T1'][ii]/ub[:,None])}
net=bx.BoxTurnpike(r=8);idx=torch.arange(8);start=time.perf_counter();loss=bx.loss_fn(net,bp,idx,1,'u',True);loss.backward();sec=time.perf_counter()-start
probes.append({'family':'E04 final r8 n3400','actual_action':'8 genuine training rows; exact model forward loss and backward, no fit','loss':float(loss.detach()),'seconds':sec,'parameters':sum(v.numel() for v in net.parameters()),'rows':ii.tolist(),'finite_gradients':all(bool(torch.isfinite(v.grad).all()) for v in net.parameters() if v.grad is not None)})
import multi_turnpike_surrogate as mt
sd=np.load(RAW/'02 - Supplement/runs/robin2d_strip/data_strip.npz');ii=sd['train_idx'][:8];xx=sd['X'];str_=sd['train_idx'];ub=sd['u_bar'][ii];ubk=sd['u_bar_k'][ii];rho=np.maximum(np.sqrt(np.mean(ub**2,axis=1)),1e-12);sig=np.maximum(np.sqrt(np.mean(ubk**2,axis=2)),1e-12);geo=sd['geo_cols']
sp={'xg':torch.tensor(((xx[ii]-xx[str_].mean(0))/xx[str_].std(0))[:,geo]),'tau':torch.tensor(sd['coefs'][ii]),'floor':torch.tensor(sd['lam1'][ii]),'ubk_n':torch.tensor(ubk/rho[:,None,None]),'sig_n':torch.tensor(sig/rho[:,None]),'dt':float(sd['dt'])}
net=mt.TauTurnpike(int(sd['m']),K=ubk.shape[1],r=4,n_geo=len(geo));idx=torch.arange(8);y=torch.tensor(sd['U_T1'][ii]/rho[:,None,None]);start=time.perf_counter();pred=net(sp,idx,1);loss=((pred-y).flatten(1).pow(2).sum(1)/y.flatten(1).pow(2).sum(1)).mean();loss.backward();sec=time.perf_counter()-start
probes.append({'family':'E07 final tau-linear r4','actual_action':'8 genuine training rows; exact model forward loss and backward, no fit','loss':float(loss.detach()),'seconds':sec,'parameters':sum(v.numel() for v in net.parameters()),'rows':ii.tolist(),'finite_gradients':all(bool(torch.isfinite(v.grad).all()) for v in net.parameters() if v.grad is not None)})
rep['feasibility_probes']=probes;dump('reconstruction_results.json',rep);print('Feasibility probes complete',flush=True)

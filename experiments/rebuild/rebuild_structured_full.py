from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,time,json,copy,gc,traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import numpy as np,torch
OUT = output_directory("rebuild");RAW = RAW_DATA;CODE=CODE_DIR;sys.path.insert(0,str(CODE))
import baselines_multihorizon as bm,multi_turnpike_surrogate as mt,semilinear_moving_surrogate as mv
from turnpike_surrogate import grid
torch.set_num_threads(4);torch.set_default_device('cuda');torch.set_default_dtype(torch.float64)
report={'models':[],'failures':[],'protocol':'PROTOCOL_ZH.md'}
def dump():(OUT/'structured_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def summ(a,b):
 e=np.linalg.norm((a-b).reshape(len(a),-1),axis=1)/np.linalg.norm(b.reshape(len(b),-1),axis=1)
 return {'median_pct':100*float(np.median(e)),'max_pct':100*float(e.max()),'n':len(e)}
dump()
for family,relative,epochs in [('2D-semi','02 - Supplement/runs/semilinear2d/data_semilinear2d_strong.npz',6000),
                              ('2D-semi-strip','02 - Supplement/runs/semilinear2d_mi/data.npz',3000)]:
 try:
  d=np.load(RAW/relative);x=d['X'];tr=d['train_idx'];va=d['val_idx'];te=d['test_idx'];scalar=family=='2D-semi'
  scale=d['u_bar'] if scalar else np.sqrt(np.mean(d['u_bar']**2,axis=1))
  if scalar:
   p=bm.load_problem('const',RAW/relative);make=lambda:bm.TurnpikeConst(p,r=8)
   predict=lambda net,idx,h:bm.predict(net,p,idx,h)
   lossfn=lambda net,idx:bm.loss_on(net,p,idx,1)
  else:
   p={'x':torch.tensor((x-x[tr].mean(0))/x[tr].std(0)),'floor':torch.tensor(d['lam1']),
    'ubar_n':torch.tensor(d['u_bar']/scale[:,None]),'dt':float(d['dt']),'tr':torch.tensor(tr),'va':torch.tensor(va),'te':torch.tensor(te),
    'Y1':torch.tensor(d['U_T1'].astype(float)/scale[:,None,None])}
   make=lambda:mt.GenericTurnpike(99,n_in=10,r=8,width=256)
   predict=lambda net,idx,h:net(p,idx,h)
   lossfn=lambda net,idx:(mt.rel(net(p,idx,1),p['Y1'][idx])**2).mean()
  row={'family':family,'seed':0,'dataset':str(RAW/relative),'r':8,'epochs':epochs,'batch':len(tr),'lr':.002,
    'precision':'float64','train_n':len(tr),'val_n':len(va),'test_n':len(te),'train_T':[1]}
  torch.manual_seed(0);net=make();opt=torch.optim.Adam(net.parameters(),lr=.002);times=[]
  for step in range(10):
   torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad();loss=lossfn(net,p['tr']);loss.backward();opt.step();torch.cuda.synchronize()
   assert torch.isfinite(loss)
   if step>=2:times.append(time.perf_counter()-start)
  estimate=float(np.median(times))*epochs*1.10;row['estimated_fit_s']=estimate;row['step_median_s']=float(np.median(times))
  del net,opt;gc.collect();torch.cuda.empty_cache();torch.manual_seed(0);net=make()
  executed=epochs;row['executed_epochs']=executed;row['status']='complete_fixed_fit' if executed==epochs else 'pipeline_probe_only'
  torch.cuda.synchronize();start=time.perf_counter()
  net=bm.train(net,p,[1],executed,.002,len(tr)) if scalar else mt.train(net,p,[1],executed,.002,len(tr))
  torch.cuda.synchronize();row['fit_seconds']=time.perf_counter()-start;net.eval()
  state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};stem='SELF_'+family+'_structured_seed0'
  torch.save({'identity':stem,'state':state,'settings':row,'normalizer':{'mean':x[tr].mean(0),'std':x[tr].std(0)},
    'train_idx':tr,'val_idx':va,'test_idx':te},OUT/(stem+'.pt'))
  restored=make();restored.load_state_dict(state);restored.eval();preds={};stats={}
  with torch.no_grad():
   for h in [1,2,4,8]:
    out=[]
    for first in range(0,len(te),10):
     ix=te[first:first+10];pr=predict(net,torch.tensor(ix),h);sc=torch.tensor(scale[ix]).reshape((-1,)+(1,)*(pr.ndim-1));out.append((pr*sc).cpu().numpy())
    pred=np.concatenate(out);truth=d[f'U_T{h}'][te] if scalar or h<4 else d[f'U_T{h}_test']
    preds['U_T'+str(h)]=pred;stats[str(h)]=summ(pred,truth)
   a=predict(net,p['te'][:3],8);b=predict(restored,p['te'][:3],8);replay=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a));assert replay==0
  np.savez_compressed(OUT/(stem+'_predictions.npz'),**preds,test_idx=te,X=x[te],dt=float(d['dt']))
  row.update({'checkpoint':stem+'.pt','predictions':stem+'_predictions.npz','metrics':stats,'checkpoint_replay_relative_difference':replay})
  report['models'].append(row);dump();print(family,row['status'],row['fit_seconds'],flush=True)
  del p,net,restored,preds;gc.collect();torch.cuda.empty_cache()
 except Exception:
  report['failures'].append({'family':family,'traceback':traceback.format_exc()});dump();print(traceback.format_exc(),flush=True)

# The local moving-target CNN has a larger training cost; first run a bounded chain.
try:
 torch.set_default_dtype(torch.float32)
 path=RAW/'01 - Initial delivery/runs/semilinear_moving/data.npz';d=np.load(path);X=d['X'];tr=d['train_idx'];te=d['test_idx'];test_global=190
 assert test_global in te
 ids=np.concatenate([tr[:16],[test_global]]);xx=X[ids];mu=X[:,mv.MU_COLS];mean=mu[tr].mean(0);std=mu[tr].std(0)
 ten=lambda a:torch.tensor(a,dtype=torch.float32)
 p={'mu':ten((mu[ids]-mean)/std),'omega':ten(xx[:,4]),'phi0':ten(xx[:,6]),'a':ten(xx[:,3]),'sigma':ten(10**xx[:,5]),
  'floor':ten(d['lam1'][ids]),'dt':float(d['dt']),'m':99,'row':{},'P':{},'dP':{},'Y':{}}
 nh=int(d['n_harm'].max())+1;uhat=torch.tensor(d['uhat_lin'][ids,:nh],dtype=torch.complex64)
 nvec=torch.arange(nh,dtype=torch.float32);duhat=uhat*(1j*nvec)[None,:,None]
 t,_=grid(1,p['dt']);rho=torch.sqrt((mv.periodic(uhat,p['omega'],p['phi0'],t)**2).mean((1,2)));p['rho']=rho
 for h in [1,8]:
  t,_=grid(h,p['dt']);p['row'][h]=torch.arange(len(ids))
  p['P'][h]=mv.periodic(uhat,p['omega'],p['phi0'],t)/rho[:,None,None]
  p['dP'][h]=mv.periodic(duhat,p['omega'],p['phi0'],t)/(10*rho[:,None,None])
 p['Y'][1]=ten(d['U_T1'][ids])/rho[:,None,None]
 torch.manual_seed(0);net=mv.MovingTurnpikeLocal(99,r=6,K_layer=1,width=48,width_mlp=128)
 opt=torch.optim.Adam(net.parameters(),lr=.002);times=[];losses=[]
 for step in range(20):
  ix=torch.arange((step%4)*4,(step%4+1)*4);torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad()
  loss=(mv.rel(net(p,ix,1),p['Y'][1][ix])**2).mean();loss.backward();opt.step();torch.cuda.synchronize()
  assert torch.isfinite(loss);losses.append(float(loss));times.append(time.perf_counter()-start)
 net.eval();ix=torch.tensor([16]);stem='SELF_2D-semi-move_local_pipeline_seed0'
 state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};restored=mv.MovingTurnpikeLocal(99);restored.load_state_dict(state);restored.eval()
 with torch.no_grad():
  a=net(p,ix,8);b=restored(p,ix,8);replay=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a));assert replay==0
  pred=(a*rho[ix,None,None]).cpu().numpy()
 truth=d['U_T8_test'][np.where(te==test_global)[0]]
 row={'family':'2D-semi-move','status':'pipeline_probe_only','training_steps':20,'training_parameter_n':16,'test_parameter_ids':[190],
 'precision':'float32/complex64','batch':4,'seed':0,'r':6,'step_median_s':float(np.median(times[2:])),
 'estimated_1500epoch_full550_s_at_batch4':float(np.median(times[2:]))*int(np.ceil(550/4))*1500*1.1,
 'checkpoint_replay_relative_difference':replay,'finite_loss_and_predictions':bool(np.isfinite(pred).all()),'diagnostic_T8_error':summ(pred,truth),
 'warning':'Short training subset probe, not a paper-accuracy result or historical checkpoint recovery.'}
 torch.save({'identity':stem,'state':state,'settings':row,'normalizer':{'mean':mean,'std':std},'original_train_idx':tr,'training_subset':ids[:16]},OUT/(stem+'.pt'))
 np.savez_compressed(OUT/(stem+'_predictions.npz'),U_T8=pred,test_idx=np.array([190]),X=X[[190]])
 report['models'].append(row);dump();print('moving pipeline complete',row['step_median_s'],flush=True)
except Exception:
 report['failures'].append({'family':'2D-semi-move','traceback':traceback.format_exc()});dump();print(traceback.format_exc(),flush=True)


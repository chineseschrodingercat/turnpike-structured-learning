from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,json,time,gc
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import numpy as np,torch
ROOT = output_directory("matched");RAW = RAW_DATA;CODE=CODE_DIR;sys.path.insert(0,str(CODE))
import box_turnpike_surrogate as bx,multi_turnpike_surrogate as mt
from endpoint_windows import summ,errs
torch.set_num_threads(4);torch.set_default_device('cuda');torch.set_default_dtype(torch.float32)
report={'protocol_frozen_before_training':True,'historical_hyperparameters_not_fully_recorded':True,'settings':{'epochs':300,'lr':1e-3,'batch':50,'train_T':[1,2],'seed':0,'dtype':'float32','device':'RTX 5080 CUDA','selection':'original train functions, best short-horizon validation median; no long-horizon selection','identity':'NEW 20261116 baseline, architecture from historical code; explicitly fixed missing run flags, not historical recovery'},'models':{},'probes':[]}
def dump():(ROOT/'new_box_strip_cnn_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
dump();start_all=time.perf_counter()
for family,rel in [('E04','02 - Supplement/runs/turnpike_box/box_data.npz'),('E07','02 - Supplement/runs/robin2d_strip/data_strip.npz')]:
 d=np.load(RAW/rel);x=d['X'];tr=d['train_idx'];te=d['test_idx'];scale=d['u_bar'] if family=='E04' else np.maximum(np.sqrt(np.mean(d['u_bar']**2,axis=1)),1e-12)
 p={'x':torch.tensor((x-x[tr].mean(0))/x[tr].std(0),dtype=torch.float32),'tr':torch.tensor(tr),'va':torch.tensor(d['val_idx']),'te':torch.tensor(te),'dt':float(d['dt'])}
 for T in [1,2]:p[f'U{T}' if family=='E04' else f'Y{T}']=torch.tensor(d[f'U_T{T}']/scale[:,None] if family=='E04' else d[f'U_T{T}']/scale[:,None,None],dtype=torch.float32)
 make=(lambda:bx.CNNWrap()) if family=='E04' else (lambda:mt.CNNmulti(int(d['m']),n_in=x.shape[1]))
 def loss(net,idx):return sum(bx.loss_fn(net,p,idx,T,'u',False) if family=='E04' else (mt.rel(net(p,idx,T),p[f'Y{T}'][idx])**2).mean() for T in [1,2])/2
 torch.manual_seed(0);net=make();opt=torch.optim.Adam(net.parameters(),lr=1e-3);ts=[]
 for n in range(12):
  torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad();l=loss(net,p['tr'][:50]);l.backward();opt.step();torch.cuda.synchronize()
  if n>=2:ts.append(time.perf_counter()-start)
 estimate=float(np.median(ts))*int(np.ceil(len(tr)/50))*300*1.10;report['probes'].append({'family':family,'optimizer_step_median_s':float(np.median(ts)),'estimated_full_s':estimate,'finite_loss':bool(torch.isfinite(l))});dump();del net,opt;gc.collect();torch.cuda.empty_cache()
 if estimate>300:report['models'][family]={'status':'deferred by fixed 300s per-model budget','estimate_seconds':estimate};dump();continue
 torch.manual_seed(0);net=make();torch.cuda.synchronize();start=time.perf_counter();net=bx.train(net,p,'u',False,[1,2],300,1e-3,50) if family=='E04' else mt.train(net,p,[1,2],300,1e-3,50);torch.cuda.synchronize();secs=time.perf_counter()-start;net.eval()
 checkpoint=f'NEW_{family}_CNN_seed0.pt';torch.save({'state':{k:v.cpu() for k,v in net.state_dict().items()},'identity':f'NEW_{family}_CNN_20261116_seed0','settings':report['settings'],'normalizer':{'mean':x[tr].mean(0),'std':x[tr].std(0)},'train_idx':tr,'val_idx':d['val_idx'],'test_idx':te,'training_seconds':secs},ROOT/checkpoint)
 metrics={};preds={};windows={}
 with torch.no_grad():
  for T in [1,2,4,8]:
   outs=[]
   for j in range(0,len(te),10):
    ii=te[j:j+10];idx=torch.tensor(ii);pr=bx.outputs(net,p,idx,T,False)[1] if family=='E04' else net(p,idx,T);outs.append((pr*torch.tensor(scale[ii],dtype=torch.float32).reshape((-1,)+(1,)*(pr.ndim-1))).cpu().numpy())
   pr=np.concatenate(outs);preds[T]=pr;truth=d[f'U_T{T}'][te] if family=='E04' or T<4 else d[f'U_T{T}_test'];metrics[str(T)]=summ(np.linalg.norm((pr.astype(float)-truth).reshape(len(te),-1),axis=1)/np.linalg.norm(truth.reshape(len(te),-1),axis=1));windows[T]={k:summ(v) for k,v in errs(pr.astype(float),truth.astype(float),.5).items()}
 np.savez_compressed(ROOT/f'NEW_{family}_CNN_seed0_predictions.npz',**{f'U_T{T}':v for T,v in preds.items()},test_idx=te,X=x[te],dt=float(d['dt']))
 report['models'][family]={'status':'complete new retrain','source_dataset':rel,'n_train':len(tr),'n_test':len(te),'training_seconds':secs,'checkpoint':checkpoint,'test':metrics,'windows':windows};dump();print(family,'CNN complete',secs,flush=True);del p,net,preds;gc.collect();torch.cuda.empty_cache()
report['total_elapsed_seconds']=time.perf_counter()-start_all;dump()

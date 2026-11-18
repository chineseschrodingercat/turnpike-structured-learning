from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,time,json,hashlib
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[v]='4'
import numpy as np,torch
ROOT = output_directory("matched");RAW = RAW_DATA;sys.path.insert(0,str(CODE_DIR))
import box_turnpike_surrogate as bx,fresh_test_1d as ft
torch.set_num_threads(4);torch.set_default_dtype(torch.float64);torch.set_default_device('cuda')
path=RAW/'02 - Supplement/runs/turnpike_box/box_data.npz';d=np.load(path);p=bx.load(path);cfg={'r':8,'shared_rates':True,'base':'qbar','loss':'projected control relative squared loss','train_T':[1],'epochs':6000,'lr':2e-3,'batch':700,'seed':0,'dtype':'float64','device':'RTX5080 CUDA','n_train':700,'n_validation':150,'n_test':150,'selection':'best T1 validation median; fixed 40 checks; no tuning','same_parameter_train_val_test_rows_as_new_CNN':True}
torch.manual_seed(0);net=bx.BoxTurnpike(r=8);opt=torch.optim.Adam(net.parameters(),lr=2e-3);ts=[]
for k in range(12):
 torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad();loss=bx.loss_fn(net,p,p['tr'],1,'u',True);loss.backward();opt.step();torch.cuda.synchronize()
 if k>=2:ts.append(time.perf_counter()-start)
estimate=float(np.median(ts))*6000*1.1;rep={'settings':cfg,'probe_seconds_per_epoch':float(np.median(ts)),'estimate_full_seconds':estimate,'status':'probe complete'}
if estimate>90:
 rep['status']='deferred by90-second rule';(ROOT/'matched_box_results.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep));raise SystemExit()
del net,opt;torch.manual_seed(0);net=bx.BoxTurnpike(r=8);torch.cuda.synchronize();start=time.perf_counter();net=bx.train(net,p,'u',True,[1],6000,2e-3,700);torch.cuda.synchronize();seconds=time.perf_counter()-start;net.eval()
ck={'identity':'NEW_E04_structured_n700_seed0_20261116','state':{k:v.cpu() for k,v in net.state_dict().items()},'settings':cfg,'train_idx':d['train_idx'],'val_idx':d['val_idx'],'test_idx':d['test_idx'],'normalizer':{'mean':d['X'][d['train_idx']].mean(0),'std':d['X'][d['train_idx']].std(0)},'training_seconds':seconds,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
torch.save(ck,ROOT/'NEW_E04_structured_n700_seed0.pt');te=d['test_idx'];ub=d['u_bar'][te];preds={};wins={};metrics={}
with torch.no_grad():
 for T in [1,2,4,8]:
  pr=(bx.outputs(net,p,p['te'],T,True)[1]*torch.tensor(ub)[:,None]).cpu().numpy();preds[T]=pr;truth=d[f'U_T{T}'][te];metrics[str(T)]=ft.summ(np.linalg.norm(pr-truth,axis=1)/np.linalg.norm(truth,axis=1));wins[T]=ft.errs(pr,truth,.5)
np.savez_compressed(ROOT/'NEW_E04_structured_n700_seed0_predictions.npz',**{f'U_T{T}':v for T,v in preds.items()},test_idx=te,X=d['X'][te],dt=float(d['dt']))
reload=bx.BoxTurnpike(r=8);reload.load_state_dict(ck['state']);reload.eval()
with torch.no_grad():diff=max(float(np.linalg.norm((bx.outputs(reload,p,p['te'],T,True)[1]*torch.tensor(ub)[:,None]).cpu().numpy()-preds[T])/np.linalg.norm(preds[T])) for T in [1,8])
cnn=torch.load(ROOT/'NEW_E04_CNN_seed0.pt',map_location='cpu',weights_only=False)
rep.update({'status':'complete new retrain','training_seconds':seconds,'test':metrics,'windows_and_absolute_transfer':ft.summarize_rows(wins),'checkpoint_replay_max_relative_difference':diff,'checkpoint_replay_passed':diff<1e-6,'train_idx_equals_CNN':bool(np.array_equal(ck['train_idx'],cnn['train_idx'])),'val_idx_equals_CNN':bool(np.array_equal(ck['val_idx'],cnn['val_idx'])),'test_idx_equals_CNN':bool(np.array_equal(ck['test_idx'],cnn['test_idx'])),'normalizer_equals_CNN':bool(np.array_equal(ck['normalizer']['mean'],cnn['normalizer']['mean']))})
(ROOT/'matched_box_results.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in rep.items() if k not in ['windows_and_absolute_transfer','test']}))

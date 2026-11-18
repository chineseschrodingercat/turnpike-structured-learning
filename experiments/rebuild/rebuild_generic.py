from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,time,json,copy,gc,hashlib,traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import numpy as np,torch
OUT = output_directory("rebuild");RAW = RAW_DATA;CODE=CODE_DIR;sys.path.insert(0,str(CODE))
import baselines_multihorizon as bm
torch.set_num_threads(4);torch.set_default_device('cuda');torch.set_default_dtype(torch.float64)
families=[('1D-const','const','01 - Initial delivery/runs/turnpike/turnpike_data.npz'),
 ('1D-osc','osc','01 - Initial delivery/runs/turnpike_osc/osc_data.npz'),
 ('2D-scalar','const','02 - Supplement/runs/robin2d/data_2d.npz')]
report={'protocol':'PROTOCOL_ZH.md','identity':'SELF_REBUILD_20261116','models':[],'failures':[],
 'code_sha256':hashlib.sha256((CODE/'baselines_multihorizon.py').read_bytes()).hexdigest()}
def dump():(OUT/'generic_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
dump()
for family,kind,relpath in families:
  path=RAW/relpath;d=np.load(path);tr=d['train_idx'];va=d['val_idx'];te=d['test_idx'];x=d['X'];p=bm.load_problem(kind,path)
  for method in ['deeponet','fno']:
    row={'family':family,'method':method,'seed':0,'dataset':str(path),'train_n':len(tr),'val_n':len(va),'test_n':len(te),
      'train_T':[1,2],'precision':'float64/complex128','epochs':6000 if method=='deeponet' else 300,
      'batch':len(tr) if method=='deeponet' else 50,'lr':.002 if method=='deeponet' else .001}
    try:
      make=(lambda:bm.DeepONet(x.shape[1])) if method=='deeponet' else (lambda:bm.FNO1d(x.shape[1]))
      torch.manual_seed(0);net=make();opt=torch.optim.Adam(net.parameters(),lr=row['lr']);times=[]
      idx=p['tr'][:row['batch']]
      for step in range(12):
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad()
        loss=sum(bm.loss_on(net,p,idx,h) for h in [1,2])/2;loss.backward();opt.step();torch.cuda.synchronize()
        assert torch.isfinite(loss)
        if step>=2:times.append(time.perf_counter()-start)
      estimate=float(np.median(times))*int(np.ceil(len(tr)/row['batch']))*row['epochs']*1.12
      row['step_median_s']=float(np.median(times));row['estimated_fit_s']=estimate
      del opt,net;gc.collect();torch.cuda.empty_cache()
      torch.manual_seed(0);net=make();epochs=row['epochs'] if estimate<=180 else 3
      row['executed_epochs']=epochs;row['status']='complete_fixed_fit' if epochs==row['epochs'] else 'pipeline_probe_only'
      torch.cuda.synchronize();start=time.perf_counter();net=bm.train(net,p,[1,2],epochs,row['lr'],row['batch']);torch.cuda.synchronize()
      row['fit_seconds']=time.perf_counter()-start;net.eval();stem='SELF_'+family+'_'+method+'_seed0'
      state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
      torch.save({'identity':stem,'state':state,'settings':row,'normalizer':{'mean':x[tr].mean(0),'std':x[tr].std(0)},
       'train_idx':tr,'val_idx':va,'test_idx':te},OUT/(stem+'.pt'))
      restored=make();restored.load_state_dict(state);restored.eval();preds={};statistics={}
      with torch.no_grad():
        for h in [1,2,4,8]:
          values=[]
          for first in range(0,len(te),10):
            ii=p['te'][first:first+10];q=bm.predict(net,p,ii,h);values.append((q*p['scale'][ii,None]).cpu().numpy())
          pred=np.concatenate(values);truth=(p['Y'][h][p['te']]*p['scale'][p['te'],None]).cpu().numpy()
          error=np.linalg.norm(pred-truth,axis=1)/np.linalg.norm(truth,axis=1)
          statistics[str(h)]={'median_pct':100*float(np.median(error)),'max_pct':100*float(np.max(error)),'n':len(te)}
          preds['U_T'+str(h)]=pred
        ix=p['te'][:3];a=bm.predict(net,p,ix,8);b=bm.predict(restored,p,ix,8)
        replay=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a));assert replay==0
      np.savez_compressed(OUT/(stem+'_predictions.npz'),**preds,test_idx=te,X=x[te],dt=float(d['dt']))
      row.update({'checkpoint':stem+'.pt','predictions':stem+'_predictions.npz','metrics':statistics,'checkpoint_replay_relative_difference':replay,
        'long_horizon_used_for_selection':False,'normalizer_train_only':True})
      report['models'].append(row);dump();print(family,method,row['status'],round(row['fit_seconds'],2),flush=True)
      del net,restored,preds;gc.collect();torch.cuda.empty_cache()
    except Exception:
      row['traceback']=traceback.format_exc();report['failures'].append(row);dump();print(row['traceback'],flush=True)
  del p;gc.collect();torch.cuda.empty_cache()
dump()

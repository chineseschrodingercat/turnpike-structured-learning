from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os, sys, time, json, math, gc, copy, hashlib
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import numpy as np
import torch
OUT = output_directory("rebuild")
RAW = RAW_DATA
CODE=CODE_DIR
sys.path.insert(0,str(CODE))
import semilinear_moving_surrogate as mv

def prepare(d,device='cuda'):
    X=d['X']; tr=d['train_idx']; mu=X[:,mv.MU_COLS]
    mean=mu[tr].mean(0); std=mu[tr].std(0)
    ten=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    p={'mu':ten((mu-mean)/std),'omega':ten(X[:,4]),'phi0':ten(X[:,6]),
       'a':ten(X[:,3]),'sigma':ten(10**X[:,5]),'floor':ten(d['lam1']),
       'dt':float(d['dt']),'m':int(d['m']),'row':{},'P':{},'dP':{},'Y':{}}
    for k,key in [('tr','train_idx'),('va','val_idx'),('te','test_idx')]:
        p[k]=torch.as_tensor(d[key],dtype=torch.long,device=device)
    nh=int(d['n_harm'].max())+1
    uh=torch.as_tensor(d['uhat_lin'][:,:nh].astype(np.complex128),device=device)
    # Reference preprocessing follows the original double-precision loader;
    # only its final tensors are converted to the model's float32 precision.
    om=torch.as_tensor(X[:,4],dtype=torch.float64,device=device)
    ph=torch.as_tensor(X[:,6],dtype=torch.float64,device=device)
    du=uh*(1j*torch.arange(nh,dtype=torch.float64,device=device))[None,:,None]
    t=(torch.arange(320,dtype=torch.float64,device=device)+1)*p['dt']
    p1=torch.cat([mv.periodic(uh[k:k+25],om[k:k+25],ph[k:k+25],t) for k in range(0,len(X),25)])
    rho=torch.sqrt(p1.square().mean((1,2)))
    p['rho']=rho.float();p['P'][1]=(p1/rho[:,None,None]).float();del p1
    p['dP'][1]=torch.cat([mv.periodic(du[k:k+25],om[k:k+25],ph[k:k+25],t)/(10*rho[k:k+25,None,None]) for k in range(0,len(X),25)]).float()
    p['row'][1]=torch.arange(len(X),device=device)
    p['Y'][1]=(torch.as_tensor(d['U_T1'].astype(np.float64),device=device)/rho[:,None,None]).float()
    return p,{'mean':mean,'std':std},(uh,du,om,ph,rho)

def add_horizon(p,rawref,T,ids):
    uh,du,om,ph,rho=rawref
    device=uh.device;ix=torch.as_tensor(ids,dtype=torch.long,device=device)
    t=(torch.arange(round(T/p['dt']),dtype=torch.float64,device=device)+1)*p['dt']
    p['P'][T]=torch.cat([mv.periodic(uh[z],om[z],ph[z],t)/rho[z,None,None] for z in ix.split(10)]).float()
    p['dP'][T]=torch.cat([mv.periodic(du[z],om[z],ph[z],t)/(10*rho[z,None,None]) for z in ix.split(10)]).float()
    row=torch.full((len(om),),-1,dtype=torch.long,device=device);row[ix]=torch.arange(len(ix),device=device);p['row'][T]=row

def main():
    torch.set_default_device('cuda');torch.set_default_dtype(torch.float32);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    path=RAW/'01 - Initial delivery/runs/semilinear_moving/data.npz'
    d=np.load(path);p,norm,ref=prepare(d);tr=p['tr'];va=p['va'];te=d['test_idx']
    epochs=1500;micro=64
    settings={'family':'2D-semi-move','model':'MovingTurnpikeLocal','seed':0,'r':6,'K_layer':1,'width':48,'width_mlp':128,
      'epochs':epochs,'logical_batch':len(tr),'microbatch':micro,'gradient_accumulation':True,'lr':.002,
      'precision':'float32','reference_preprocessing':'float64/complex128, cast after normalization',
      'train_n':len(tr),'val_n':len(va),'test_n':len(te),'train_T':[1],'validation_T':[1],
      'long_horizon_used_for_selection':False,'normalizer_train_only':True,'dataset':str(path),'protocol':'FULL_REBUILD_PROTOCOL_ZH.md'}
    # Memory/performance probe uses training rows only and is discarded.
    while True:
        torch.manual_seed(0);net=mv.MovingTurnpikeLocal(99);opt=torch.optim.Adam(net.parameters(),lr=.002)
        try:
            torch.cuda.synchronize();st=time.perf_counter()
            loss=mv.rel(net(p,tr[:micro],1),p['Y'][1][tr[:micro]]).square().mean();loss.backward();opt.step();torch.cuda.synchronize()
            settings['probe_seconds']=time.perf_counter()-st;settings['probe_peak_bytes']=torch.cuda.max_memory_allocated()
            print('training-only probe',micro,settings['probe_seconds'],settings['probe_peak_bytes'],flush=True)
            del loss,net,opt;gc.collect();torch.cuda.empty_cache();break
        except torch.OutOfMemoryError:
            del net,opt;gc.collect();torch.cuda.empty_cache();micro//=2
            if micro<1:raise
    settings['microbatch']=micro
    torch.manual_seed(0);net=mv.MovingTurnpikeLocal(99);opt=torch.optim.Adam(net.parameters(),lr=.002)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=2e-6)
    best=float('inf');best_ep=0;best_state=None;history=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for ep in range(1,epochs+1):
        net.train();perm=tr[torch.randperm(len(tr),device=tr.device)];opt.zero_grad();loss_value=0.
        for ix in perm.split(micro):
            loss=mv.rel(net(p,ix,1),p['Y'][1][ix]).square().mean()
            assert torch.isfinite(loss),(ep,float(loss))
            weighted=loss*(len(ix)/len(tr));weighted.backward();loss_value+=float(weighted.detach())
        opt.step();sch.step()
        if ep%37==0 or ep==epochs:
            net.eval()
            with torch.no_grad():
                v=float(torch.cat([mv.rel(net(p,ix,1),p['Y'][1][ix]) for ix in va.split(16)]).median())
            if v<best:
                best=v;best_ep=ep;best_state={k:z.detach().cpu().clone() for k,z in net.state_dict().items()}
            history.append({'epoch':ep,'train_loss':loss_value,'validation_median':v,'elapsed_s':time.perf_counter()-start})
        if ep%25==0 or ep==1:
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            print('epoch',ep,'loss',loss_value,'best_val',best,'elapsed_s',round(elapsed,2),'projected_fit_s',round(elapsed/ep*epochs,1),flush=True)
            (OUT/'moving_progress.json').write_text(json.dumps({'epoch':ep,'epochs':epochs,'elapsed_s':elapsed,'best_validation':best,'settings':settings,'history':history},indent=2))
        if ep%100==0:
            torch.save({'epoch':ep,'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),
                        'best_state':best_state,'best_epoch':best_ep,'best_validation':best,'settings':settings},OUT/'moving_resume.pt')
    torch.cuda.synchronize();fit=time.perf_counter()-start
    net.load_state_dict(best_state);net.eval();settings.update({'status':'complete_fixed_fit','executed_epochs':epochs,'fit_seconds':fit,'selected_epoch':best_ep,'selected_validation_median':best})
    stem='SELF_2D-semi-move_structured_seed0';preds={};stats={}
    restored=mv.MovingTurnpikeLocal(99);restored.load_state_dict(best_state);restored.eval()
    with torch.no_grad():
        for T in [1,2,4,8]:
            if T!=1:add_horizon(p,ref,T,te)
            chunks=[]
            for ids in np.array_split(te,50):
                ix=torch.as_tensor(ids,dtype=torch.long,device='cuda')
                chunks.append((net(p,ix,T)*p['rho'][ix,None,None]).cpu().numpy())
            a=np.concatenate(chunks);b=d[f'U_T{T}'][te] if T<4 else d[f'U_T{T}_test']
            err=np.linalg.norm((a.astype(float)-b).reshape(len(te),-1),axis=1)/np.linalg.norm(b.astype(float).reshape(len(te),-1),axis=1)
            stats[str(T)]={'n':len(te),'median_pct':float(np.median(err)*100),'max_pct':float(err.max()*100)};preds[f'U_T{T}']=a
            print('test',T,stats[str(T)],flush=True)
            if T==8:
                ix=p['te'][:1];a1=net(p,ix,T);a2=restored(p,ix,T)
                replay=float(torch.linalg.vector_norm(a1-a2)/torch.linalg.vector_norm(a1));assert replay==0
            if T!=1:del p['P'][T],p['dP'][T];gc.collect();torch.cuda.empty_cache()
    settings['checkpoint_replay_relative_difference']=replay
    torch.save({'identity':stem,'state':best_state,'settings':settings,'normalizer':norm,'train_idx':d['train_idx'],'val_idx':d['val_idx'],'test_idx':te},OUT/(stem+'.pt'))
    np.savez_compressed(OUT/(stem+'_predictions.npz'),**preds,test_idx=te,X=d['X'][te],dt=float(d['dt']))
    report={'settings':settings,'metrics':stats,'history':history,'checkpoint':stem+'.pt','predictions':stem+'_predictions.npz','failures':[]}
    (OUT/'moving_full_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FULL MOVING REBUILD COMPLETE',fit,flush=True)

if __name__=='__main__':main()

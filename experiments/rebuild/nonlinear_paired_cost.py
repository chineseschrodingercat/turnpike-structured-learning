from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os,sys,time,json,traceback,gc,platform
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
import scipy
from threadpoolctl import threadpool_limits,threadpool_info
OUT = output_directory("rebuild");RAW = RAW_DATA;CODE=CODE_DIR;sys.path.insert(0,str(CODE))
import semilinear2d as sc,semilinear2d_mi as mi,semilinear_moving as mo
import baselines_multihorizon as bm,multi_turnpike_surrogate as mt,semilinear_moving_surrogate as mv
from robin2d import Cfg2D,lambda1
from gen_semilinear2d_mi_data import cfg_of as strip_cfg,M
from gen_semilinear_moving_data import cfg_of as moving_cfg
from moving2d import assemble,periodic_turnpike
torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);torch.set_num_threads(1);threadpool_limits(1)
SPECS=[('2D-semi','02 - Supplement/runs/semilinear2d/data_semilinear2d_strong.npz',[73,52,473]),
       ('2D-semi-strip','02 - Supplement/runs/semilinear2d_mi/data.npz',[568,605,465]),
       ('2D-semi-move','01 - Initial delivery/runs/semilinear_moving/data.npz',[272,190,356])]
REPORT=OUT/'nonlinear_paired_cost.json'
report=json.loads(REPORT.read_text()) if REPORT.exists() else {'protocol':'FULL_REBUILD_PROTOCOL_ZH.md','cpu_threads':1,'samples':[],'failures':[],
    'hardware':{'cpu':'AMD Ryzen 7 9800X3D','platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
    'scope':'Isolated sequential CPU measurements. Model loading reported separately. Fresh query includes reference/floor, input preparation and actual trained model evaluation. Long and short solvers include LQ initialization and nonlinear solve. No plant verification or file serialization in timers.'}
def dump():REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
def relative(a,b):
 a=np.asarray(a);b=np.asarray(b);dtype=np.result_type(a.dtype,b.dtype,np.float64)
 return float(np.linalg.norm(a.astype(dtype)-b.astype(dtype))/np.linalg.norm(b.astype(dtype)))
def save_np(name,**kw):np.savez_compressed(OUT/name,**kw)
def truth(d,f,i,h):return d[f'U_T{h}'][i] if f=='2D-semi' or h<4 else d[f'U_T{h}_test'][np.flatnonzero(d['test_idx']==i)[0]]

def main():
 for family,relpath,indices in SPECS:
  d=np.load(RAW/relpath)
  torch.set_default_dtype(torch.float32 if family=='2D-semi-move' else torch.float64)
  start=time.perf_counter();ck=torch.load(OUT/f'SELF_{family}_structured_seed0.pt',map_location='cpu',weights_only=False);norm=ck['normalizer']
  if family=='2D-semi':net=bm.TurnpikeConst({'x':torch.zeros(1,6),'floor':torch.ones(1)},r=8)
  elif family=='2D-semi-strip':net=mt.GenericTurnpike(99,n_in=10,r=8,width=256)
  else:net=mv.MovingTurnpikeLocal(99)
  net.load_state_dict(ck['state']);net.eval();model_load_s=time.perf_counter()-start
  for i in indices:
   if any(z['family']==family and z['sample_index']==i and z.get('passed') for z in report['samples']):continue
   row={'family':family,'sample_index':i,'checkpoint':f'SELF_{family}_structured_seed0.pt','checkpoint_status':ck['settings']['status'],
        'model_load_seconds':model_load_s,'solves':{}}
   try:
    print('START',family,i,flush=True);x=d['X'][i];tepos=int(np.flatnonzero(d['test_idx']==i)[0]);assert ck['settings']['status']=='complete_fixed_fit'
    if family=='2D-semi':
     geo={k:float(d[k][i]) for k in ('xa','xb','ya','yb','alpha','beta')};gamma=float(d['gamma'][i]);cfg=lambda h:Cfg2D(nh=100,T=float(h),Nt=320*h,**geo)
     reference=lambda:sc.static_ocp(sc.build(cfg(1),gamma))
     lq=lambda h:sc.lq_control(cfg(h),(h,))[h];build=lambda h:sc.build(cfg(h),gamma);solve=sc.solve
    elif family=='2D-semi-strip':
     gamma=10**x[9];cfg=lambda h:strip_cfg(x,h);reference=lambda:mi.static_ocp(mi.build(cfg(1),M,gamma))
     lq=lambda h:mi.lq_control(cfg(h),M);build=lambda h:mi.build(cfg(h),M,gamma);solve=mi.solve
    else:
     gamma=10**x[7];cfg=lambda h:moving_cfg(x,h)
     def reference():
      uh,hi=periodic_turnpike(assemble(cfg(1)));return {'uhat':uh,'info':hi,'lam1_lin':lambda1(100,float(10**x[2]))}
     lq=lambda h:mo.lq_control(cfg(h))[0];build=lambda h:mo.build(cfg(h),gamma);solve=mo.solve
    # Fresh parameter query: no use of the dataset's cached reference/floor.
    start_total=time.perf_counter();start=start_total;st=reference();refsec=time.perf_counter()-start
    start=time.perf_counter()
    if family=='2D-semi':
     scale=st['u_bar'];p={'x':torch.tensor(((x-norm['mean'])/norm['std'])[None,:]),'floor':torch.tensor([st['lam1_lin']]),'dt':1/320,
        'yd':lambda h,idx:torch.ones(len(idx),int(320*h))};net.floor=p['floor'];fn=lambda:scale*bm.predict(net,p,torch.tensor([0]),8)[0].numpy()
    elif family=='2D-semi-strip':
     scale=float(np.sqrt(np.mean(st['u_bar']**2)));p={'x':torch.tensor(((x-norm['mean'])/norm['std'])[None,:]),'floor':torch.tensor([st['lam1_lin']]),'dt':1/320,
        'ubar_n':torch.tensor((st['u_bar']/scale)[None,:])};fn=lambda:scale*net(p,torch.tensor([0]),8)[0].numpy()
    else:
     p={'mu':torch.tensor(((x[mv.MU_COLS]-norm['mean'])/norm['std'])[None,:],dtype=torch.float32),'omega':torch.tensor([x[4]],dtype=torch.float32),
        'phi0':torch.tensor([x[6]],dtype=torch.float32),'a':torch.tensor([x[3]],dtype=torch.float32),'sigma':torch.tensor([10**x[5]],dtype=torch.float32),
        'floor':torch.tensor([st['lam1_lin']],dtype=torch.float32),'dt':1/320,'m':99,'row':{8:torch.tensor([0])},'P':{},'dP':{}}
     uh=torch.tensor(st['uhat'][None,:,:],dtype=torch.complex128);om=torch.tensor([x[4]],dtype=torch.float64);ph=torch.tensor([x[6]],dtype=torch.float64)
     du=uh*(1j*torch.arange(uh.shape[1],dtype=torch.float64))[None,:,None]
     t1=torch.arange(1,321,dtype=torch.float64)/320;tt=torch.arange(1,2561,dtype=torch.float64)/320
     scale=float(mv.periodic(uh,om,ph,t1).square().mean().sqrt());p['P'][8]=(mv.periodic(uh,om,ph,tt)/scale).float();p['dP'][8]=(mv.periodic(du,om,ph,tt)/(10*scale)).float()
     fn=lambda:scale*net(p,torch.tensor([0]),8)[0].numpy()
    prepsec=time.perf_counter()-start
    with torch.no_grad():
     start=time.perf_counter();pred=fn();firstsec=time.perf_counter()-start;totalsec=time.perf_counter()-start_total
     times=[]
     for rep in range(10):
      start=time.perf_counter();cachedpred=fn();times.append(time.perf_counter()-start)
    assert np.isfinite(pred).all() and np.array_equal(pred,cachedpred)
    target=truth(d,family,i,8)
    row['fresh_query']={'reference_and_floor_s':refsec,'input_preparation_s':prepsec,'first_model_call_s':firstsec,'total_s':totalsec,
       'cached_calls_s':times,'cached_median_s':float(np.median(times)),'relative_control_error':relative(pred,target)}
    row['floor_relative_difference']=abs(st['lam1_lin']/float(d['lam1'][i])-1)
    if family!='2D-semi-move':row['reference_relative_difference']=relative(np.atleast_1d(st['u_bar']),np.atleast_1d(d['u_bar'][i]))
    else:row['reference_relative_difference']=relative(st['uhat'],d['uhat_lin'][i,:len(st['uhat'])])
    gpu=np.load(OUT/f'SELF_{family}_structured_seed0_predictions.npz')['U_T8'][tepos]
    row['cpu_fresh_reference_vs_gpu_cached_reference_relative_difference']=relative(pred,gpu)
    save_np(f'SELF_{family}_case{i}_T8_query.npz',U=pred,U_reference=target,X=x)
    print('QUERY',family,i,totalsec,row['fresh_query']['relative_control_error'],flush=True)
    for h in ([1,8] if family!='2D-semi-move' else [8]):
     print('SOLVE START',family,i,h,flush=True);start=time.perf_counter();ulq=lq(h);lqsec=time.perf_counter()-start
     u,info=solve(build(h),ulq,verbose=True);total=time.perf_counter()-start;target=truth(d,family,i,h);err=relative(u,target)
     row['solves'][str(h)]={'total_s':total,'lq_s':lqsec,'control_relative_difference':err,'gradient_residual':info['grad_rel'],'info':info}
     save_np(f'SELF_{family}_case{i}_T{h}_solver.npz',U=u,U_reference=target,X=x)
     assert err<5e-6 and info['grad_rel']<1e-7,(family,i,h,err,info['grad_rel'])
     print('SOLVE PASS',family,i,h,total,err,flush=True)
     if h==1:short=u.copy()
    if family!='2D-semi-move':
     start=time.perf_counter();base=st['u_bar'];stitched=np.broadcast_to(base,(2560,)+np.shape(base)).copy()
     stitched[:160]+=short[:160]-base;stitched[-160:]+=short[-160:]-base;assemblysec=time.perf_counter()-start
     row['stitch_Ts1']={'reference_and_floor_s':refsec,'short_solve_s':row['solves']['1']['total_s'],'assembly_s':assemblysec,
       'total_component_sum_s':refsec+row['solves']['1']['total_s']+assemblysec,'relative_control_error':relative(stitched,truth(d,family,i,8))}
     save_np(f'SELF_{family}_case{i}_T8_stitch.npz',U=stitched,U_reference=truth(d,family,i,8),X=x)
    row['passed']=True;report['samples'].append(row);dump();print('CASE PASS',family,i,flush=True);gc.collect()
   except Exception:
    row['passed']=False;row['traceback']=traceback.format_exc();report['failures'].append(row);dump();print(row['traceback'],flush=True)
  d.close()
 report['threadpools']=threadpool_info();report['passed']=len(report['samples'])==9 and all(x['passed'] for x in report['samples']);dump()
 assert report['passed'],report['failures']
if __name__=='__main__':main()

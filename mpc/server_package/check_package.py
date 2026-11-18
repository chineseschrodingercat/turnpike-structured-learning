from pathlib import Path
import ast,hashlib,json,argparse,sys
ROOT=Path(__file__).resolve().parent
def verify_manifest():
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for item in manifest:
        p=ROOT/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Missing or changed package file: '+item['path'])
    return len(manifest)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--load-model',action='store_true');args=ap.parse_args()
    count=verify_manifest();protocol=json.loads((ROOT/'RUN_PROTOCOL.json').read_text())
    for p in ROOT.rglob('*.py'):
        if 'results' not in p.parts:ast.parse(p.read_text(encoding='utf-8'))
    assert protocol['core_trajectory_count']==96 and protocol['feedback_updates_per_trajectory']==80
    cases=0
    for cohort in ('A','B'):
        q=json.loads((ROOT/'inputs'/f'scenario_{cohort}.json').read_text());r=json.loads((ROOT/q['reference_file']).read_text())
        assert len(q['scenarios'])==12 and len({x['idx'] for x in q['scenarios']})==12
        for sc in q['scenarios']:
            assert sc['seed']==sc['idx']+1000
            for ctrl in protocol['all_controllers']:
                a=[z for z in r['runs'] if z['scenario']==sc['idx'] and z['ctrl']==ctrl];assert len(a)==1
                assert len(a[0]['u'])==2560 and len(a[0]['ymean'])==2560
                cases+=1
    model='not loaded (static-only check)'
    if args.load_model:
        import os
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
        sys.path.insert(0,str(ROOT/'code'))
        import torch,numpy as np
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
        from mpc_semilinear import load_surrogate
        net,norm,floor=load_surrogate(ROOT/protocol['model'])
        assert all(torch.isfinite(t).all() for t in net.state_dict().values())
        assert np.all(np.asarray(norm['x_std'])>0) and np.all(np.asarray(norm['z_scale'])>0)
        model={'parameters':sum(x.numel() for x in net.parameters()),'floor':floor,'forward_calls':0}
    print(json.dumps({'passed':True,'manifest_files':count,'historical_trajectories_checked':cases,'model':model,'mpc_simulations_run':0},indent=2))
if __name__=='__main__':main()

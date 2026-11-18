"""Verify returned records and recompute tables without running an MPC simulation.

Usage: python verify_mpc_return.py --delivery DIR --package DIR --out DIR
Only Python and NumPy are required. Inputs are never modified.
"""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
def close(a,b): assert np.isclose(a,b,rtol=1e-12,atol=2e-14),(a,b)
def stats(a):
    a=np.asarray(a,dtype=float)
    assert a.size and np.isfinite(a).all()
    return {'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())}
def csv_write(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--delivery',type=Path,required=True)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();d,p,o=a.delivery,a.package,a.out;o.mkdir(parents=True,exist_ok=True)
    mf=read(d/'RETURN_MANIFEST.json')
    actual={x.relative_to(d).as_posix() for x in d.rglob('*') if x.is_file()}
    assert actual=={x['path'] for x in mf}|{'RETURN_MANIFEST.json'}
    for x in mf:
        q=d/x['path'];assert q.stat().st_size==x['bytes'] and sha(q)==x['sha256'],q
    for n in ('MANIFEST_SHA256.json','RUN_PROTOCOL.json','SOURCE_PROVENANCE.json'):
        assert (p/n).read_bytes()==(d/'package_identity'/n).read_bytes()
    package_mf=read(p/'MANIFEST_SHA256.json')
    # The original handoff manifest is a list of path/bytes/sha256 records.
    for x in package_mf:
        q=p/x['path'];assert q.stat().st_size==x['bytes'] and sha(q)==x['sha256'],q
    prot=read(p/'RUN_PROTOCOL.json');scens={c:read(p/'inputs'/f'scenario_{c}.json') for c in ('A','B')}
    hist={c:{(r['scenario'],r['ctrl']):r for r in read(p/scens[c]['reference_file'])['runs']} for c in scens}
    ids=[s['idx'] for s in scens['A']['scenarios']]
    assert len(ids)==len(set(ids))==12
    assert ids==[s['idx'] for s in scens['B']['scenarios']]
    allrows=[];stage_records={};stage_checks={}
    for stage,expected_n in [('smoke',4),('full',96),('timing',24)]:
        config=read(d/'results'/stage/'run_config.json')
        assert config['manifest_sha256']==sha(p/'MANIFEST_SHA256.json')
        assert config['threads_per_worker']==1 and config['suite']=='core'
        assert config['workers']==(24 if stage=='full' else 1)
        positions=[0] if stage=='smoke' else [0,5,11] if stage=='timing' else range(12)
        cohorts=['A'] if stage=='smoke' else ['A','B']
        keys={f'{c}_{scens[c]["scenarios"][i]["idx"]}_{ctrl}' for c in cohorts for i in positions for ctrl in prot['core_controllers']}
        assert len(keys)==expected_n and set(config['expected_keys'])==keys
        files=list((d/'results'/stage/'runs').glob('*.json'));assert {f.stem for f in files}==keys
        assert not list((d/'results'/stage/'failures').glob('*.json'))
        records={}
        for f in sorted(files):
            r=read(f);records[r['key']]=r
            assert r['key']==f.stem==f'{r["cohort"]}_{r["scenario"]}_{r["ctrl"]}'
            assert r['Tp']==int(r['ctrl'].split('_')[1])
            u=np.array(r['u']);ym=np.array(r['ymean']);t=np.array(r['controller_seconds'])
            assert u.shape==ym.shape==(2560,) and t.shape==(80,)
            assert np.isfinite(u).all() and np.isfinite(ym).all() and np.isfinite(t).all() and (t>0).all()
            assert np.isfinite(r['J']) and r['J']>0
            assert all(x['num_threads']==1 for x in r['threadpools'])
            h=hist[r['cohort']][(r['scenario'],r['ctrl'])]
            ur=float(np.linalg.norm(u-np.array(h['u']))/np.linalg.norm(h['u']))
            yr=float(np.linalg.norm(ym-np.array(h['ymean']))/np.linalg.norm(h['ymean']))
            jr=abs(r['J']/h['J']-1)
            close(ur,r['relative_control_difference']);close(jr,r['relative_cost_difference'])
            assert ur<prot['verification']['relative_control_difference_max'] and jr<prot['verification']['relative_cost_difference_max'] and yr<1e-10
            residual=np.asarray(r['solver_residuals'])
            if r['ctrl'].startswith('exact'):
                assert residual.shape==(80,) and np.isfinite(residual).all() and (residual>=0).all()
                assert residual.max()<prot['verification']['solver_residual_max']
                close(float(residual.max()),r['max_solver_residual'])
            else:
                assert residual.size==0 and r['max_solver_residual'] is None
            close(float(np.median(t)),r['solve_s_median']);close(float(t.sum()),r['solve_s_total'])
            close(r['setup_seconds']+t[0],r['initialization_to_first_control_seconds'])
            assert 0<=r['reference_seconds']<=r['setup_seconds']
            assert r['total_scenario_wall_seconds']>=r['setup_seconds']+t.sum()
            assert r['passed'] is True
            allrows.append({'stage':stage,'cohort':r['cohort'],'parameter_index':r['scenario'],'controller':r['ctrl'],
                'J':r['J'],'control_replay_relative':ur,'state_mean_replay_relative':yr,'cost_replay_relative':jr,
                'max_reported_solver_residual':float(residual.max()) if residual.size else None,
                'call_median_s':float(np.median(t)),'first_call_s':float(t[0]),'warm_call_median_s':float(np.median(t[1:])),
                'setup_s':r['setup_seconds'],'reference_s':r['reference_seconds'],
                'setup_to_first_s':r['initialization_to_first_control_seconds'],
                'all_80_calls_s':float(t.sum()),'trajectory_wall_s':r['total_scenario_wall_seconds']})
        ss=read(d/'results'/stage/'summary.json')
        assert ss['expected']==ss['results']==expected_n and ss['passed'] and not ss['failed'] and not ss['missing']
        sr=[r for r in allrows if r['stage']==stage]
        close(ss['max_control_replay_difference'],max(r['control_replay_relative'] for r in sr))
        close(ss['max_cost_replay_difference'],max(r['cost_replay_relative'] for r in sr))
        stage_records[stage]=records
        stage_checks[stage]={'trajectories':expected_n,'controller_calls':expected_n*80,'max_control_replay_relative':max(r['control_replay_relative'] for r in sr),
            'max_cost_replay_relative':max(r['cost_replay_relative'] for r in sr),
            'max_state_mean_replay_relative':max(r['state_mean_replay_relative'] for r in sr),
            'max_reported_solver_residual':max(r['max_reported_solver_residual'] or 0 for r in sr)}
    cross=[]
    for k,r in stage_records['timing'].items():
        f=stage_records['full'][k]
        err=float(np.linalg.norm(np.array(r['u'])-f['u'])/np.linalg.norm(f['u']))
        assert err<1e-10;close(r['J'],f['J'])
        assert r['newton_iters_median']==f['newton_iters_median']
        cross.append({'key':k,'control_relative':err,'cost_relative':abs(r['J']/f['J']-1)})
    accuracy=[]
    for r in allrows:
        base=stage_records[r['stage']][f'{r["cohort"]}_{r["parameter_index"]}_exact_4']['J']
        r['cost_difference_pct']=100*(r['J']/base-1)
    for c in ('A','B'):
        for ctrl in ('tp_1','tp_4','exact_1','exact_4'):
            rs=[r for r in allrows if r['stage']=='full' and r['cohort']==c and r['controller']==ctrl]
            accuracy.append({'scenario':c,'controller':ctrl,'n_parameters':len(rs),**stats([r['cost_difference_pct'] for r in rs])})
    timing=[]
    for ctrl in ('tp_1','tp_4','exact_1','exact_4'):
        rs=[r for r in allrows if r['stage']=='timing' and r['controller']==ctrl]
        row={'controller':ctrl,'n_parameter_scenarios':len(rs),'n_distinct_parameters':3}
        for k in ('call_median_s','first_call_s','warm_call_median_s','setup_s','reference_s','setup_to_first_s','all_80_calls_s','trajectory_wall_s'):
            row[k]=stats([r[k] for r in rs])
        pooled=np.concatenate([stage_records['timing'][f'{r["cohort"]}_{r["parameter_index"]}_{ctrl}']['controller_seconds'] for r in rs])
        row['pooled_call_s']=stats(pooled)
        timing.append(row)
    paired=[]
    for horizon in (1,4):
        rs=[]
        for c in ('A','B'):
            for i in (0,5,11):
                idx=ids[i];s=stage_records['timing'][f'{c}_{idx}_tp_{horizon}'];e=stage_records['timing'][f'{c}_{idx}_exact_{horizon}']
                rs.append({'cohort':c,'parameter_index':idx,'horizon':horizon,
                    'controller_median_speedup':e['solve_s_median']/s['solve_s_median'],
                    'trajectory_wall_speedup':e['total_scenario_wall_seconds']/s['total_scenario_wall_seconds']})
        paired.append({'horizon':horizon,'n_parameter_scenarios':6,
            'controller_median_speedup':stats([r['controller_median_speedup'] for r in rs]),
            'trajectory_wall_speedup':stats([r['trajectory_wall_speedup'] for r in rs]),'cases':rs})
    csv_write(o/'per_trajectory_recomputed.csv',allrows);csv_write(o/'closed_loop_costs.csv',accuracy)
    save(o/'timing_statistics.json',timing);save(o/'paired_speedups.json',paired)
    save(o/'cross_stage_numerical_comparison.json',cross)
    result={'passed':True,'return_manifest_files_verified':len(mf),'source_package_files_verified':len(package_mf),
        'stages':stage_checks,'unique_parameter_count':12,'disturbance_scenarios_per_parameter':2,
        'full_trajectories':96,'total_execution_records':124,'total_controller_calls':9920,
        'timing_parameter_indices':[ids[i] for i in (0,5,11)],'timing_aggregation':'median across six trajectory-level statistics (three parameters, two scenarios)',
        'locally_executed_MPC_simulations':0,
        'verification_scope':'Independent record/hash/identity/arithmetic verification; residual values are checked from the solver records. State trajectories and objective integrations are not re-simulated here.',
        'source_zip_package_manifest_sha256':sha(p/'MANIFEST_SHA256.json')}
    save(o/'verification.json',result)
    print(json.dumps({'verification':result,'accuracy':accuracy,'timing':timing,'paired_speedups':paired},indent=2))

if __name__=='__main__':main()

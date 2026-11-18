from pathlib import Path
import json,csv,statistics,math,argparse
def summarize(out):
    out=Path(out);config=json.loads((out/'run_config.json').read_text());expected=config['expected_keys'];runs=[];missing=[];failed=[]
    for key in expected:
        p=out/'runs'/(key+'.json')
        if not p.is_file():missing.append(key);continue
        r=json.loads(p.read_text())
        if not r.get('passed'):failed.append(key)
        runs.append(r)
    exact={(r['cohort'],r['scenario']):r['J'] for r in runs if r['ctrl']=='exact_4' and r['passed']}
    rows=[]
    for r in runs:
        baseline=exact.get((r['cohort'],r['scenario']))
        rows.append({'key':r['key'],'cohort':r['cohort'],'scenario':r['scenario'],'controller':r['ctrl'],'passed':r['passed'],
           'J':r['J'],'signed_cost_difference_pct_vs_fresh_exact4':100*(r['J']/baseline-1) if baseline is not None else None,
           'median_call_s':statistics.median(r['controller_seconds']),
           'median_calls_after_first_s':statistics.median(r['controller_seconds'][1:]),'first_call_s':r['controller_seconds'][0],
           'reference_s':r['reference_seconds'],'setup_s':r['setup_seconds'],'setup_to_first_control_s':r['initialization_to_first_control_seconds'],
           'total_controller_s':sum(r['controller_seconds']),'scenario_wall_s':r['total_scenario_wall_seconds'],
           'control_relative_difference_vs_historical':r['relative_control_difference'],'cost_relative_difference_vs_historical':r['relative_cost_difference'],
           'max_solver_residual':r['max_solver_residual'],'timing_scope':config['timing_scope']})
    groups=[]
    for cohort,ctrl in sorted({(r['cohort'],r['controller']) for r in rows}):
        z=[r for r in rows if r['cohort']==cohort and r['controller']==ctrl and r['passed']];item={'cohort':cohort,'controller':ctrl,'n':len(z)}
        for key in ('median_call_s','setup_to_first_control_s','signed_cost_difference_pct_vs_fresh_exact4'):
            a=[r[key] for r in z if r[key] is not None]
            item[key]={'median':statistics.median(a),'min':min(a),'max':max(a)} if a else None
        groups.append(item)
    if rows:
        with (out/'per_case_summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    completed_invocations=[json.loads(p.read_text()) for p in (out/'invocations').glob('*.json')]
    result={'expected':len(expected),'results':len(runs),'missing':missing,'failed':failed,'passed':not missing and not failed,
      'timing_scope':config['timing_scope'],'recorded_completed_invocation_wall_s':sum(x['elapsed_wall_seconds'] for x in completed_invocations),
      'wall_note':'Interrupted invocations may have partial completed cases but no final wall-clock receipt; their wall time is not inferred.',
      'max_control_replay_difference':max((r['relative_control_difference'] for r in runs),default=None),
      'max_cost_replay_difference':max((r['relative_cost_difference'] for r in runs),default=None),'groups':groups}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('result_directory',type=Path);args=ap.parse_args();print(json.dumps(summarize(args.result_directory),indent=2))

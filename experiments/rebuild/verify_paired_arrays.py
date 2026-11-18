"""Independently match saved timed controls to dataset rows and reported errors."""
from experiments.paths import CODE_DIR, RAW_DATA, output_directory
import os, json
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
R = output_directory("rebuild")
RAW = RAW_DATA
report = json.loads((R / 'nonlinear_paired_cost.json').read_text())
assert report['passed']
paths = {'2D-semi': '02 - Supplement/runs/semilinear2d/data_semilinear2d_strong.npz',
         '2D-semi-strip': '02 - Supplement/runs/semilinear2d_mi/data.npz',
         '2D-semi-move': '01 - Initial delivery/runs/semilinear_moving/data.npz'}
fixed = {'2D-semi': [73, 52, 473], '2D-semi-strip': [568, 605, 465], '2D-semi-move': [272, 190, 356]}
rel = lambda a,b: float(np.linalg.norm(a.astype(float)-b.astype(float))/np.linalg.norm(b.astype(float)))
checks = []
for family, path in paths.items():
    rows = [r for r in report['samples'] if r['family'] == family]
    assert [r['sample_index'] for r in rows] == fixed[family]
    with np.load(RAW / path) as d:
        for r in rows:
            i = r['sample_index']
            j = np.flatnonzero(d['test_idx'] == i)
            assert len(j) == 1
            def truth(h):
                return d[f'U_T{h}'][i] if family == '2D-semi' or h == 1 else d[f'U_T{h}_test'][j[0]]
            for h, entry in r['solves'].items():
                with np.load(R / f'SELF_{family}_case{i}_T{h}_solver.npz') as p:
                    assert np.array_equal(p['X'], d['X'][i])
                    assert np.array_equal(p['U_reference'], truth(int(h)))
                    error = rel(p['U'], truth(int(h)))
                    assert np.isclose(error, entry['control_relative_difference'], rtol=1e-10, atol=1e-15)
                    assert error < 5e-6 and entry['gradient_residual'] < 1e-7
                    assert entry['total_s'] >= entry['lq_s'] > 0
            q = r['fresh_query']
            with np.load(R / f'SELF_{family}_case{i}_T8_query.npz') as p:
                assert np.array_equal(p['X'], d['X'][i])
                assert np.array_equal(p['U_reference'], truth(8))
                assert np.isclose(rel(p['U'], truth(8)), q['relative_control_error'], rtol=1e-10, atol=1e-15)
            assert q['total_s'] >= q['reference_and_floor_s'] + q['input_preparation_s'] + q['first_model_call_s']
            assert len(q['cached_calls_s']) == 10 and np.all(np.array(q['cached_calls_s']) > 0)
            assert np.median(q['cached_calls_s']) == q['cached_median_s']
            if family != '2D-semi-move':
                with np.load(R / f'SELF_{family}_case{i}_T1_solver.npz') as p:
                    short = p['U']
                base = d['u_bar'][i]
                stitched = np.broadcast_to(base, (2560,) + np.shape(base)).copy()
                stitched[:160] += short[:160] - base
                stitched[-160:] += short[-160:] - base
                with np.load(R / f'SELF_{family}_case{i}_T8_stitch.npz') as p:
                    # Freshly solved references can differ from cached dataset references at roundoff.
                    assert rel(stitched, p['U']) < 1e-10
                    assert np.isclose(rel(p['U'], truth(8)), r['stitch_Ts1']['relative_control_error'], rtol=1e-10, atol=1e-15)
            checks.append({'family': family, 'sample_index': i, 'dataset_identity_and_errors_match': True})
out = {'passed': len(checks) == 9, 'cases': checks, 'long_solves': 9, 'short_solves': 6,
       'scope': 'Independent array identity, control errors, stitching and timing-component consistency; no timings remeasured.'}
(R / 'verification/paired_arrays.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))

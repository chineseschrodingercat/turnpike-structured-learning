#!/usr/bin/env python3
"""Check of the identifiability lemma on the 1D constant-target data: the per-sample end-window transfer ratio
||e||_{W}(T = 8) / ||e||_{W}(T = 1) of TurnpikeNet (trained on T = 1, 5 seeds) against the open-loop rate lambda_1(beta).
The lemma bounds the initial/terminal ambiguity of a single training horizon by ~ e^{-lambda_1 T_train / 2}, so large
ratios should sit at small lambda_1."""

import glob
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "exp/robin")
from endpoint_windows import errs  # noqa: E402
from turnpike_surrogate import first_eig  # noqa: E402

d = np.load("runs/turnpike/turnpike_data.npz")
lam1 = {}
rows = []
for f in sorted(glob.glob("runs/unified/pred_const1d_tp_T1/pred_turnpike_seed*.npz")):
    z = np.load(f)
    idx = z["test_idx"]
    for i in idx:
        if i not in lam1:
            lam1[i] = first_eig(d["beta"][i])
    e1 = errs(z["U_T1"].astype(float), d["U_T1"][idx].astype(float), 0.5)
    e8 = errs(z["U_T8"].astype(float), d["U_T8"][idx].astype(float), 0.5)
    for j, i in enumerate(idx):
        rows.append((lam1[i], e8["init_abs"][j] / e1["init_abs"][j], e8["term_abs"][j] / e1["term_abs"][j]))
R = np.array(rows)
out = {"spearman_init": float(spearmanr(R[:, 0], R[:, 1]).statistic), "spearman_term": float(spearmanr(R[:, 0], R[:, 2]).statistic)}
for lo, hi in ((0, 4), (4, 6), (6, 100)):
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    out[f"lam1_{lo}_{hi}"] = {"n": int(m.sum()), "ambiguity_bound_e^-lam1/2": float(np.exp(-0.5 * R[m, 0].mean())),
                              "init_ratio_median": float(np.median(R[m, 1])), "init_ratio_p90": float(np.percentile(R[m, 1], 90)),
                              "term_ratio_median": float(np.median(R[m, 2])), "term_ratio_p90": float(np.percentile(R[m, 2], 90))}
print(json.dumps(out, indent=1))

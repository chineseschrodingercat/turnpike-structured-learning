#!/usr/bin/env python3
"""Where do the worst errors of the box-constrained turnpike surrogate come from?

Trains tp_q_uloss (seed 0, T = 1) and, per test sample, reports the error at T = 1 and T = 8,
the T=1 interaction floor (error of clip(q_bar + l0 + l1) with exact T-invariant layers from T = 8),
and the sample's parameters / active fractions. If the worst T=8 errors already appear at T=1, they
are learning errors on hard samples, not extrapolation failures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from box_turnpike_surrogate import SPECS, load, outputs, rel_err, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="tp_q_uloss")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    p = load(args.data)
    d = np.load(args.data)
    te = p["te"]
    tei = te.numpy()
    make, target, project = SPECS[args.model]
    torch.manual_seed(0)
    model = train(make(), p, target, project, [1], 6000, 2e-3, 700)

    with torch.no_grad():
        err = {T: rel_err(outputs(model, p, te, T, project)[1], p[f"U{T}"][te]).numpy() for T in (1, 8)}
    # interaction floor at T=1 from exact T=8 layers
    qbar, lo, hi = d["q_bar"][tei], d["lo"][tei], d["hi"][tei]
    Q8 = d["Q_T8"][tei]
    half = Q8.shape[1] // 2
    L0, L1 = Q8[:, :half] - qbar[:, None], Q8[:, ::-1][:, :half] - qbar[:, None]
    k = np.arange(320)
    uh = np.clip(qbar[:, None] + L0[:, k] + L1[:, 319 - k], lo[:, None], hi[:, None])
    U1 = d["U_T1"][tei]
    floor1 = np.linalg.norm(uh - U1, axis=1) / np.linalg.norm(U1, axis=1)

    X = d["X"][tei]
    order = np.argsort(err[8])[::-1]
    rows = []
    for i in order[:12]:
        rows.append({"err_T8": float(err[8][i]), "err_T1": float(err[1][i]), "interaction_floor_T1": float(floor1[i]),
                     "theta_a": float(X[i, 4]), "theta_b": float(X[i, 5]), "log10_alpha": float(X[i, 2]),
                     "log10_beta": float(X[i, 3]), "xa": float(d["xa"][tei][i]),
                     "frac_upper_T1": float(d["frac_upper_T1"][tei][i]), "frac_lower_T1": float(d["frac_lower_T1"][tei][i])})
    rep = {
        "model": args.model,
        "spearman_err_T8_vs_err_T1": float(spearmanr(err[8], err[1]).statistic),
        "spearman_err_T8_vs_floor_T1": float(spearmanr(err[8], floor1).statistic),
        "spearman_err_T8_vs": {name: float(spearmanr(err[8], col).statistic) for name, col in
                               (("theta_a", X[:, 4]), ("theta_b", X[:, 5]), ("log10_alpha", X[:, 2]), ("log10_beta", X[:, 3]),
                                ("frac_upper_T1", d["frac_upper_T1"][tei]))},
        "ratio_err_T8_over_err_T1": {"median": float(np.median(err[8] / err[1])), "max": float(np.max(err[8] / err[1]))},
        "worst12_T8": rows,
    }
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

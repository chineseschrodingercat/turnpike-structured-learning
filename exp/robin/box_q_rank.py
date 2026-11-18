#!/usr/bin/env python3
"""Is the constrained control 'a projection of something low-rank'? For each box pilot: q = B^T p(u*) / (alpha m)
(so u* = clip(q)), singular values of u*, q and the unconstrained control, and the representation floor of
  (i) POD of u* itself:      ||u*_r - u*|| / ||u*||
  (ii) project a rank-r q:   ||clip(q_r) - u*|| / ||u*||."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import Sweeps, assemble, target_loads  # noqa: E402
from plot_bistable_box_gt import load  # noqa: E402

RANKS = (5, 10, 15, 20, 30, 40)


def trunc(Z, r):
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def rk(Z, tol):
    sv = np.linalg.svd(Z, compute_uv=False)
    return int(np.sum(sv > tol * sv[0]))


def main():
    d = pathlib.Path(sys.argv[1])
    out = {}
    for name in ("box_pos", "box_pos40", "box_sym30", "box_a1e-4_pos40", "box_a1e-5_pos40"):
        z = load(d / f"{name}.npz")
        cf, C, Cu = z["cf"], z["C"].astype(float), z["C_ref"].astype(float)
        ua, ub = z["meta"]["ua"], z["meta"]["ub"]
        o = assemble(cf)
        GY, _ = target_loads(o)
        sw = Sweeps(o, GY)
        am = cf.alpha * np.asarray(o.Mu.sum(axis=1)).ravel()[None, :]
        q = (sw.adjoint(None, True) + sw.adjoint(sw.forward(C), False)) / am
        nC = np.linalg.norm(C)
        r = {"alpha": cf.alpha, "kkt_res": float(np.linalg.norm(C - np.clip(q, ua, ub)) / nC),
             "rank_u": [rk(C, 1e-3), rk(C, 1e-4)], "rank_q": [rk(q, 1e-3), rk(q, 1e-4)], "rank_unc": [rk(Cu, 1e-3), rk(Cu, 1e-4)],
             "pod_u": {k: float(np.linalg.norm(trunc(C, k) - C) / nC) for k in RANKS},
             "clip_q_r": {k: float(np.linalg.norm(np.clip(trunc(q, k), ua, ub) - C) / nC) for k in RANKS}}
        out[name] = r
        print(name, json.dumps(r))
    (d / "box_q_rank.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

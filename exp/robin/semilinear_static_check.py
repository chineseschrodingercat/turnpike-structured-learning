#!/usr/bin/env python3
"""Sparse vs dense static nonlinear OCP on every test sample: agreement with the stored (dense) u_bar / lam1 and
single-thread wall clock (the online cost of the surrogate's plateau and rate floor)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig  # noqa: E402
from semilinear1d import build, static_ocp, static_ocp_sparse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    rows = []
    for i in d["test_idx"]:
        geo = dict(alpha=float(d["alpha"][i]), beta=float(d["beta"][i]), xa=float(d["xa"][i]), xb=float(d["xb"][i]))
        s = build(RobinConfig(nx=160, Nt=320, **geo), float(d["gamma"][i]), float(d["T_ref"][i]))
        t0 = time.perf_counter()
        dn = static_ocp(s)
        t1 = time.perf_counter()
        sp_ = static_ocp_sparse(s)
        t2 = time.perf_counter()
        rows.append({"dense_s": t1 - t0, "sparse_s": t2 - t1, "sparse_iters": sp_["iters"],
                     "ubar_diff": abs(sp_["u_bar"] - d["u_bar"][i]) / abs(d["u_bar"][i]),
                     "lam1_diff": abs(sp_["lam1_lin"] - d["lam1"][i]) / d["lam1"][i]})
    rep = {k: {"median": float(np.median([r[k] for r in rows])), "max": float(np.max([r[k] for r in rows]))} for k in rows[0]}
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()

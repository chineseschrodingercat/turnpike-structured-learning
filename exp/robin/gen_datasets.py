#!/usr/bin/env python3
"""Regenerate the four report configurations (report Tables 2-3) with robin_solver.py.

Geometry: w in [0.1, 0.5], c_rel in [0, 1], c = w/2 + c_rel (1 - w), (x_a, x_b) = (c - w/2, c + w/2).
    p3_neumann     : (w, c_rel, T_ref, log10 alpha), beta = 0, y_d = T_ref
    robin_beta1    : (w, c_rel, T_ref, log10 alpha), beta = 1, y_d = T_ref
    robin_betavar  : (w, c_rel, T_ref, log10 alpha, log10 beta), y_d = T_ref
    robin_osc      : (w, c_rel, log10 alpha, log10 beta, A, omega), y_d = A sin(omega t) + 1.25
T_ref in [0.5, 2], log10 alpha in [-3, 0], log10 beta in [-1, 1], A in [0.25, 0.75], omega in [pi, 4 pi].
L = T = kappa = 1, f = 0, y0 = 0, nx = 160, Nt = 320, N = 2000, split 1400/300/300.

All stored controls use the physical sign u = p(0,t)/alpha (for p3_neumann this is -u_P3).
Our Latin hypercube draws are not the upstream samples, so only distributional
statistics (e.g. POD ranks) are comparable with the report.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robin_solver import RobinConfig, reduced_control, reduced_system, solve_kkt  # noqa: E402

CONFIGS = {
    "p3_neumann": ["w", "c_rel", "T_ref", "log10_alpha"],
    "robin_beta1": ["w", "c_rel", "T_ref", "log10_alpha"],
    "robin_betavar": ["w", "c_rel", "T_ref", "log10_alpha", "log10_beta"],
    "robin_osc": ["w", "c_rel", "log10_alpha", "log10_beta", "A", "omega"],
}
RANGES = {
    "w": (0.1, 0.5),
    "c_rel": (0.0, 1.0),
    "T_ref": (0.5, 2.0),
    "log10_alpha": (-3.0, 0.0),
    "log10_beta": (-1.0, 1.0),
    "A": (0.25, 0.75),
    "omega": (np.pi, 4 * np.pi),
}


def sample_params(name: str, n: int, seed: int) -> np.ndarray:
    cols = CONFIGS[name]
    unit = qmc.LatinHypercube(d=len(cols), seed=seed).random(n)
    lo = np.array([RANGES[c][0] for c in cols])
    hi = np.array([RANGES[c][1] for c in cols])
    return lo + unit * (hi - lo)


def problem_from_row(name: str, row: np.ndarray) -> tuple[RobinConfig, object, dict]:
    v = dict(zip(CONFIGS[name], row))
    w, c_rel = v["w"], v["c_rel"]
    c = w / 2 + c_rel * (1 - w)
    xa, xb = c - w / 2, c + w / 2
    beta = {"p3_neumann": 0.0, "robin_beta1": 1.0}.get(name, None)
    if beta is None:
        beta = 10.0 ** v["log10_beta"]
    alpha = 10.0 ** v["log10_alpha"]
    if name == "robin_osc":
        A, om = v["A"], v["omega"]
        yd = lambda x, t: np.full_like(np.asarray(x, float), A * np.sin(om * t) + 1.25)
    else:
        T_ref = v["T_ref"]
        yd = lambda x, t: np.full_like(np.asarray(x, float), T_ref)
    cfg = RobinConfig(alpha=alpha, beta=beta, xa=xa, xb=xb, nx=160, Nt=320)
    return cfg, yd, {"xa": xa, "xb": xb, "beta": beta, "alpha": alpha}


def solve_one(args: tuple[str, np.ndarray]) -> dict:
    name, row = args
    cfg, yd, meta = problem_from_row(name, row)
    res = solve_kkt(cfg, yd_func=yd)
    red = reduced_system(res.operators, res.data)
    u_red = reduced_control(red, cfg.alpha)
    return {
        "U": res.U,
        "rel_res": res.rel_res,
        "costs": (res.J_total, res.J_track, res.J_ctrl),
        "reduced_vs_kkt": float(np.linalg.norm(u_red - res.U) / np.linalg.norm(res.U)),
        **meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lhs-seed", type=int, default=20260617)
    ap.add_argument("--split-seed", type=int, default=20260616)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool

    for name in args.configs:
        t0 = time.perf_counter()
        X = sample_params(name, args.n, args.lhs_seed)
        with Pool(args.workers) as pool:
            rows = pool.map(solve_one, [(name, x) for x in X], chunksize=4)
        U = np.array([r["U"] for r in rows])
        perm = np.random.default_rng(args.split_seed).permutation(args.n)
        n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
        meta = {
            "config": name,
            "input_columns": CONFIGS[name],
            "ranges": {c: RANGES[c] for c in CONFIGS[name]},
            "L": 1.0, "T": 1.0, "kappa": 1.0, "f": "0", "y0": "0", "nx": 160, "Nt": 320,
            "control_sign": "physical, alpha u = p(0,t)",
            "lhs_seed": args.lhs_seed, "split_seed": args.split_seed,
            "generator": "exp/robin/gen_datasets.py (re-implementation from the report)",
        }
        path = args.out_dir / f"{name}_{args.n}_nx160_nt320.npz"
        np.savez_compressed(
            path,
            X=X,
            U=U,
            xa=np.array([r["xa"] for r in rows]),
            xb=np.array([r["xb"] for r in rows]),
            beta=np.array([r["beta"] for r in rows]),
            alpha=np.array([r["alpha"] for r in rows]),
            kkt_rel_res=np.array([r["rel_res"] for r in rows]),
            costs=np.array([r["costs"] for r in rows]),
            reduced_vs_kkt=np.array([r["reduced_vs_kkt"] for r in rows]),
            t=np.arange(1, 321) / 320.0,
            train_idx=np.sort(perm[:n_tr]),
            val_idx=np.sort(perm[n_tr : n_tr + n_va]),
            test_idx=np.sort(perm[n_tr + n_va :]),
            metadata_json=json.dumps(meta),
        )
        rr = np.array([r["rel_res"] for r in rows])
        dv = np.array([r["reduced_vs_kkt"] for r in rows])
        print(
            f"{name}: {args.n} samples in {time.perf_counter() - t0:.0f}s, "
            f"max KKT rel res {rr.max():.3e}, max |u_reduced - u_kkt|/|u_kkt| {dv.max():.3e} -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

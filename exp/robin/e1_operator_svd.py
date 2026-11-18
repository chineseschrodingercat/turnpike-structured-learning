#!/usr/bin/env python3
"""E1: singular values of the control-to-observation operator S(x_a, x_b, beta).

S maps u in L2(0,T) to y restricted to omega_o in L2(0,T; L2(omega_o)). Its
squared singular values are the eigenvalues of H = S^T C S (robin_solver.reduced_system).
For a regularization alpha the optimal control is u = sum_j sigma_j/(sigma_j^2 + alpha) <r, w_j> v_j,
so the filter f_j = sigma_j/(sigma_j^2 + alpha) sets how many right singular vectors
v_j a control can need, for any target.

Prediction under test: sigma_j decays fast when the subzone is away from the
controlled boundary (x_a > 0) and slows down as x_a -> 0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robin_solver import RobinConfig, assemble_operators, assemble_problem_data, reduced_system  # noqa: E402

W_LIST = [0.1, 0.3, 0.5]
CREL_LIST = [0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
BETA_LIST = [0.0, 0.1, 1.0, 10.0]
ALPHA_LIST = [1e-3, 1e-2, 1e-1, 1.0]
EPS_S = [1e-2, 1e-4, 1e-6]
EPS_F = [1e-2, 1e-3, 1e-4]


def spectrum(args: tuple[float, float, float]) -> dict:
    w, c_rel, beta = args
    c = w / 2 + c_rel * (1 - w)
    xa, xb = c - w / 2, c + w / 2
    cfg = RobinConfig(alpha=1.0, beta=beta, xa=xa, xb=xb)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, lambda x, t: np.zeros_like(np.asarray(x, float)))
    s2 = np.sort(np.linalg.eigvalsh(reduced_system(ops, data).H))[::-1]
    sigma = np.sqrt(np.clip(s2, 0.0, None))
    rel = sigma / sigma[0]
    out = {
        "w": w, "c_rel": c_rel, "xa": xa, "xb": xb, "beta": beta,
        "sigma": sigma.tolist(),
        "rank_S": {f"{e:g}": int(np.sum(rel > e)) for e in EPS_S},
        "rank_control_operator": {},
    }
    for alpha in ALPHA_LIST:
        f = sigma / (sigma**2 + alpha)
        fr = f / f.max()
        out["rank_control_operator"][f"{alpha:g}"] = {f"{e:g}": int(np.sum(fr > e)) for e in EPS_F}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool

    grid = [(w, c, b) for w in W_LIST for c in CREL_LIST for b in BETA_LIST]
    with Pool(args.workers) as pool:
        rows = pool.map(spectrum, grid)
    (args.out_dir / "e1_spectra.json").write_text(json.dumps(rows), encoding="utf-8")

    def pick(w, c, b):
        return next(r for r in rows if r["w"] == w and r["c_rel"] == c and r["beta"] == b)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for c in CREL_LIST:
        r = pick(0.3, c, 1.0)
        s = np.array(r["sigma"])
        ax.semilogy(np.arange(1, 61), s[:60] / s[0], label=f"x_a={r['xa']:.3f}")
    ax.set_title("sigma_j/sigma_1, w=0.3, beta=1 (vary subzone position)")
    ax.set_xlabel("j"); ax.set_ylim(1e-10, 2); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for b in BETA_LIST:
        r = pick(0.3, 0.3, b)
        s = np.array(r["sigma"])
        ax.semilogy(np.arange(1, 61), s[:60] / s[0], label=f"beta={b:g}")
    ax.set_title("sigma_j/sigma_1, w=0.3, c_rel=0.3 (vary beta)")
    ax.set_xlabel("j"); ax.set_ylim(1e-10, 2); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    for w in W_LIST:
        for alpha, ls in [(1e-3, "-"), (1e-1, "--")]:
            rs = [pick(w, c, 1.0) for c in CREL_LIST]
            ax.plot([r["xa"] for r in rs], [r["rank_control_operator"][f"{alpha:g}"]["0.001"] for r in rs],
                    ls, marker="o", label=f"w={w}, alpha={alpha:g}")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_title("# filter factors > 1e-3 * max (beta=1)")
    ax.set_xlabel("x_a (distance of subzone from controlled boundary)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "e1_spectra.png", dpi=150)

    print("w     c_rel  x_a     beta   rankS(1e-4) rankS(1e-6)  ctrl-rank@1e-3 for alpha=1e-3,1e-2,1e-1,1")
    for r in rows:
        cr = r["rank_control_operator"]
        print(f"{r['w']:.1f}  {r['c_rel']:<5}  {r['xa']:.4f}  {r['beta']:<5g}  {r['rank_S']['0.0001']:>11d} {r['rank_S']['1e-06']:>11d}  "
              + "  ".join(f"{cr[f'{a:g}']['0.001']:>3d}" for a in ALPHA_LIST))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()

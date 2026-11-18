#!/usr/bin/env python3
"""E2: rank-k closed-form control from the operator SVD vs the KKT control, and data POD.

For every sample, with its own S (geometry, beta) and target:
    u_k = V_k diag(1/(sigma^2 + alpha)) V_k^T g           (operator truncation, data-free)
    proj_k = V_k V_k^T u_kkt                               (is u_kkt in span of top-k v_j?)
For each dataset, a train-only POD (centered by the train mean, as in the report):
    ranks at 99 / 99.9 / 99.99 % energy (compare report Table 4)
    rank-k projection error of the test controls (the ROM-MLP floor)
All errors are per-sample relative L2(0,T) errors (dt-weighted, equal to Euclidean).
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
from scipy.stats import spearmanr  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_datasets import problem_from_row  # noqa: E402
from robin_solver import assemble_operators, assemble_problem_data, reduced_system  # noqa: E402

K_LIST = list(range(1, 17)) + [20, 24, 32, 48]
TOLS = [1e-2, 1e-3]


def per_sample(args: tuple[str, np.ndarray, np.ndarray]) -> dict:
    name, row, u_kkt = args
    cfg, yd, meta = problem_from_row(name, row)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, yd)
    red = reduced_system(ops, data)
    s2, V = np.linalg.eigh(red.H)
    order = np.argsort(s2)[::-1]
    s2, V = np.clip(s2[order], 0.0, None), V[:, order]
    coef = V.T @ red.g
    nrm = np.linalg.norm(u_kkt)
    closed, proj = [], []
    for k in K_LIST:
        uk = V[:, :k] @ (coef[:k] / (s2[:k] + cfg.alpha))
        closed.append(float(np.linalg.norm(uk - u_kkt) / nrm))
        pk = V[:, :k] @ (V[:, :k].T @ u_kkt)
        proj.append(float(np.linalg.norm(pk - u_kkt) / nrm))
    u_full = V @ (coef / (s2 + cfg.alpha))
    return {"closed": closed, "proj": proj, "full": float(np.linalg.norm(u_full - u_kkt) / nrm), **meta}


def pod_analysis(U: np.ndarray, tr: np.ndarray, te: np.ndarray, dt: float) -> dict:
    mean = U[tr].mean(axis=0)
    Z = (U[tr] - mean) * np.sqrt(dt)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    ranks = {f"{q}": int(np.searchsorted(energy, q) + 1) for q in (0.99, 0.999, 0.9999)}
    errs = []
    for k in K_LIST:
        Phi = Vt[:k]
        C = U[te] - mean
        R = C - (C @ Phi.T) @ Phi
        errs.append(np.linalg.norm(R, axis=1) / np.linalg.norm(U[te], axis=1))
    return {"energy_ranks": ranks, "sigma": s[:32].tolist(), "test_proj_err": np.array(errs)}  # (len(K), n_test)


def summarize(e: np.ndarray) -> dict:
    return {"median": float(np.median(e)), "mean": float(np.mean(e)), "p90": float(np.percentile(e, 90)), "max": float(np.max(e))}


def k_needed(errs: np.ndarray, tol: float) -> np.ndarray:
    """errs: (n_samples, len(K_LIST)); smallest k in K_LIST reaching tol, else 320."""
    out = np.full(errs.shape[0], 320)
    for i in range(errs.shape[0]):
        hit = np.nonzero(errs[i] <= tol)[0]
        if hit.size:
            out[i] = K_LIST[hit[0]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool

    report = {}
    files = sorted(args.data_dir.glob("*_nx160_nt320.npz"))
    fig, axes = plt.subplots(2, len(files), figsize=(4.6 * len(files), 8.5), squeeze=False)
    for col, path in enumerate(files):
        d = np.load(path)
        name = json.loads(str(d["metadata_json"]))["config"]
        X, U = d["X"], d["U"]
        tr, te = d["train_idx"], d["test_idx"]
        with Pool(args.workers) as pool:
            rows = pool.map(per_sample, [(name, X[i], U[i]) for i in range(len(X))], chunksize=4)
        closed = np.array([r["closed"] for r in rows])
        proj = np.array([r["proj"] for r in rows])
        full = np.array([r["full"] for r in rows])
        xa = np.array([r["xa"] for r in rows])
        alpha = np.array([r["alpha"] for r in rows])
        pod = pod_analysis(U, tr, te, 1.0 / 320)

        res = {
            "n": int(len(X)),
            "pod_energy_ranks_train": pod["energy_ranks"],
            "pod_sigma_train": pod["sigma"],
            "full_rank_closed_form_vs_kkt_max": float(full.max()),
            "k_list": K_LIST,
            "test": {},
            "k_needed_all_samples": {},
        }
        for j, k in enumerate(K_LIST):
            res["test"][str(k)] = {
                "operator_closed_form": summarize(closed[te, j]),
                "operator_projection": summarize(proj[te, j]),
                "pod_projection": summarize(pod["test_proj_err"][j]),
            }
        for tol in TOLS:
            kn = k_needed(closed, tol)
            res["k_needed_all_samples"][f"{tol:g}"] = {
                "median": float(np.median(kn)), "p90": float(np.percentile(kn, 90)), "max": int(kn.max()),
                "spearman_vs_xa": float(spearmanr(kn, xa).statistic),
                "spearman_vs_log_alpha": float(spearmanr(kn, np.log10(alpha)).statistic),
                "median_by_xa_quartile": [float(np.median(kn[(xa >= lo) & (xa <= hi)]))
                                          for lo, hi in zip(np.quantile(xa, [0, .25, .5, .75]), np.quantile(xa, [.25, .5, .75, 1]))],
                "median_by_log_alpha_quartile": [float(np.median(kn[(np.log10(alpha) >= lo) & (np.log10(alpha) <= hi)]))
                                                 for lo, hi in zip(np.quantile(np.log10(alpha), [0, .25, .5, .75]),
                                                                   np.quantile(np.log10(alpha), [.25, .5, .75, 1]))],
            }
        report[name] = res

        ax = axes[0, col]
        for label, arr in [("operator closed form (median)", closed[te]), ("operator projection (median)", proj[te])]:
            ax.semilogy(K_LIST, np.median(arr, axis=0), marker="o", ms=3, label=label)
        ax.semilogy(K_LIST, np.median(pod["test_proj_err"], axis=1), marker="s", ms=3, label="train-POD projection (median)")
        ax.semilogy(K_LIST, np.max(closed[te], axis=0), "--", color="C0", alpha=0.6, label="operator closed form (max)")
        ax.semilogy(K_LIST, np.max(pod["test_proj_err"], axis=1), "--", color="C2", alpha=0.6, label="POD projection (max)")
        ax.set_title(f"{name}: test rel. L2 vs rank k"); ax.set_xlabel("k"); ax.grid(alpha=0.3)
        ax.set_xlim(0, 33); ax.legend(fontsize=7)

        ax = axes[1, col]
        kn = k_needed(closed, 1e-3)
        sc = ax.scatter(xa, kn, c=np.log10(alpha), s=6, cmap="viridis")
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_title(f"{name}: k for 0.1% error (operator)"); ax.set_xlabel("x_a"); ax.set_ylabel("k")
        fig.colorbar(sc, ax=ax, label="log10 alpha")
        print(f"{name}: POD ranks {pod['energy_ranks']}, full-rank closed form vs KKT max {full.max():.2e}", flush=True)
        for k in (4, 6, 8, 12, 16):
            t = res["test"][str(k)]
            print(f"  k={k:>2}: operator closed form median {t['operator_closed_form']['median']:.2e} max {t['operator_closed_form']['max']:.2e} | "
                  f"POD projection median {t['pod_projection']['median']:.2e} max {t['pod_projection']['max']:.2e}", flush=True)
        print(f"  k needed for 1e-3: {res['k_needed_all_samples']['0.001']}", flush=True)

    fig.tight_layout()
    fig.savefig(args.out_dir / "e2_closed_form.png", dpi=140)
    (args.out_dir / "e2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()

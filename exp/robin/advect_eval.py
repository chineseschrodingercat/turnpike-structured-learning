#!/usr/bin/env python3
"""Evaluation for the advection dataset: dataset analysis (ranks, layer lengths, T = 2 vs T = 4 layer consistency),
optimality gaps (J(u_pred) - J*) / J* of saved predictions at T = 2 and T = 4, single-thread timing (exact vs
surrogates), and an example figure (one T = 4 test sample)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import advect_moving as am  # noqa: E402
import moving2d  # noqa: E402
from gen_advect_data import STEPS_PER_UNIT, cfg_of  # noqa: E402

MODELS = ["tp_transport", "tp_transport_mirror", "tp_transport_w48", "tp_transport_mirror_w48", "cnn_per", "cnn", "deeponet", "fno", "per_only"]


def summ(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def periodic_np(uhat, nh, omega, phi0, t):
    return moving2d.eval_periodic(uhat[: nh + 1].astype(np.complex128), omega, phi0, t)


def analysis(d):
    te = d["test_idx"]
    out = {"pcg_iters_T2": summ(d["iters_T2"]), "exact_seconds_T2_loaded": summ(d["hb_seconds"] + d["pcg_seconds_T2"]),
           "hb_seconds_loaded": summ(d["hb_seconds"]), "n_harm": summ(d["n_harm"])}
    rk, per_only, cons, mid = {2: [], 4: []}, {2: [], 4: []}, [], []
    for j, i in enumerate(te):
        for T in (2, 4):
            U = d["U_T2"][i] if T == 2 else d["U_T4_test"][j]
            t = np.arange(1, U.shape[0] + 1) / STEPS_PER_UNIT
            P = periodic_np(d["uhat"][i], int(d["n_harm"][i]), 2 * np.pi, float(d["X"][i, 2]), t)
            sv = np.linalg.svd(U.astype(float), compute_uv=False)
            rk[T].append(int(np.sum(sv > 1e-3 * sv[0])))
            per_only[T].append(float(np.linalg.norm(U - P) / np.linalg.norm(U)))
            if T == 4:
                k = int(0.8 * STEPS_PER_UNIT)
                U2 = d["U_T2"][i]
                cons.append(float(np.linalg.norm(U2[:k] - U[:k]) / np.linalg.norm(U[:k] - P[:k])))
                sl = slice(int(1.5 * STEPS_PER_UNIT), int(2.5 * STEPS_PER_UNIT))
                mid.append(float(np.linalg.norm((U - P)[sl]) / np.linalg.norm(P[sl])))
    out.update({"rank_T2": summ(rk[2]), "rank_T4": summ(rk[4]), "periodic_only_T2": summ(per_only[2]), "periodic_only_T4": summ(per_only[4]),
                "init_layer_T2_vs_T4_on_[0,0.8]": summ(cons), "T4_mid_[1.5,2.5]_deviation_from_periodic": summ(mid)})
    return out


def gap_one(task):
    i, mu, T, Jstar, preds = task
    c = cfg_of(mu, T)
    o = am.assemble(c)
    GY, csq = moving2d.target_loads(o)
    return i, T, {k: (moving2d.cost(o, GY, csq, U.astype(float)) - Jstar) / Jstar for k, U in preds.items()}


def gap(d, pred_dir, workers):
    te = d["test_idx"]
    P = {m: np.load(pred_dir / f"pred_{m}_seed0.npz") for m in MODELS if (pred_dir / f"pred_{m}_seed0.npz").exists()}
    tasks = []
    for j, i in enumerate(te):
        for T in (2, 4):
            Jstar = float(d["J_T2"][i] if T == 2 else d["J_T4_test"][j])
            tasks.append((int(i), d["X"][i], T, Jstar, {m: P[m][f"U_T{T}"][j] for m in P}))
    from multiprocessing import Pool

    with Pool(workers) as pool:
        res = pool.map(gap_one, tasks)
    out = {}
    for T in (2, 4):
        for m in P:
            out[f"{m}_T{T}"] = summ([r[2][m] for r in res if r[1] == T])
    return out


def timing(d, k_samples, out_path):
    """Single thread: exact (assemble + harmonic balance + PCG from the periodic warm start) vs the harmonic balance
    alone (what tp_transport / cnn_per / per_only need at inference) and the network forwards on CPU."""
    import torch

    import advect_surrogate as asg

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)  # an imported helper module switches the default to float64
    te = d["test_idx"]
    rows = []
    for i in te[:k_samples]:
        mu = d["X"][i]
        r = {}
        for T in (2, 4):
            t0 = time.perf_counter()
            c = cfg_of(mu, T)
            o = am.assemble(c)
            U, _ = moving2d.periodic_turnpike(o, nmax=79)
            t_hb = time.perf_counter() - t0
            GY, _ = moving2d.target_loads(o)
            t = np.arange(1, c.Nt + 1) * c.dt
            C, info = am.pcg_solve(o, GY, C0=moving2d.eval_periodic(U, c.omega, c.phi0, t), tol=1e-9)
            r[f"exact_T{T}"] = time.perf_counter() - t0
            r[f"hb_T{T}"] = t_hb
            r[f"pcg_iters_T{T}"] = info["iters"]
        rows.append(r)
        out_path.write_text(json.dumps({"exact_rows": rows}, indent=1))  # keep the expensive part if the rest fails
    out = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    # network forwards (weights do not matter for timing): build a one-sample problem dict on the CPU
    path = pathlib.Path(d["_path"])
    p = asg.load(path, "cpu")
    idx = p["te"][:1]
    for name, mk in (("tp_transport", lambda: asg.TransportTP(p["m"])),
                     ("tp_transport_mirror_w48", lambda: asg.TransportTP(p["m"], width=48, mirror=True)), ("cnn_per", lambda: asg.CNNt(p["m"], True)),
                     ("cnn", lambda: asg.CNNt(p["m"], False)), ("deeponet", asg.DeepONet), ("fno", asg.FNO2d)):
        model = mk()
        with torch.no_grad():
            for T in (2, 4):
                model(p, idx, T)
                t0 = time.perf_counter()
                for _ in range(3):
                    model(p, idx, T)
                out[f"net_{name}_T{T}"] = (time.perf_counter() - t0) / 3
    return out


def example_figure(d, pred_dir, out, which="median"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    te = d["test_idx"]
    P = {m: np.load(pred_dir / f"pred_{m}_seed0.npz") for m in MODELS if (pred_dir / f"pred_{m}_seed0.npz").exists()}
    E = np.array([np.linalg.norm(P["tp_transport"]["U_T4"][j] - d["U_T4_test"][j]) / np.linalg.norm(d["U_T4_test"][j]) for j in range(len(te))])
    j = int(np.argsort(E)[len(E) // 2]) if which == "median" else int(np.argmax(E))
    i = int(te[j])
    U = d["U_T4_test"][j].astype(float)
    t = np.arange(1, U.shape[0] + 1) / STEPS_PER_UNIT
    s = np.arange(1, U.shape[1] + 1) / (U.shape[1] + 1)
    v = np.quantile(np.abs(U), 0.995)
    names = ["exact"] + [m for m in MODELS if m in P]
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 8.4), squeeze=False)
    for c, nm in enumerate(names):
        Z = U if nm == "exact" else P[nm]["U_T4"][j].astype(float)
        ax = axes[0, c]
        pc = ax.pcolormesh(t, s, Z.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        err = 0.0 if nm == "exact" else np.linalg.norm(Z - U) / np.linalg.norm(U)
        ax.set_title(f"{nm}" + ("" if nm == "exact" else f": rel err {100 * err:.2f}%"), fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        ax.axvline(2.0, color="k", lw=0.6, ls=":")
        fig.colorbar(pc, ax=ax)
        ax = axes[1, c]
        if nm == "exact":
            ax.semilogy(t, np.linalg.norm(U, axis=1), color="k", label="|u(t)|")
            Pp = periodic_np(d["uhat"][i], int(d["n_harm"][i]), 2 * np.pi, float(d["X"][i, 2]), t)
            ax.semilogy(t, np.linalg.norm(U - Pp, axis=1), color="C3", label="|u - u_periodic|")
            ax.legend(fontsize=8)
            ax.set_title("exact: deviation from the periodic turnpike", fontsize=10)
        else:
            ax.semilogy(t, np.linalg.norm(Z - U, axis=1) / np.linalg.norm(U, axis=1).max(), color="C0")
            ax.set_ylim(1e-5, 2)
            ax.set_title(f"{nm}: |error(t)| / max|u|", fontsize=10)
        ax.axvline(2.0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    mu = d["X"][i]
    fig.suptitle(f"Advection, observation x in [0.6, 0.8]: T = 4 test sample #{i} ({which} tp_transport error); trained on T = 2 only "
                 f"(dotted: end of the training horizon). gamma = {mu[0]:.2f}, eps = {10 ** mu[1]:.1e}, a = {mu[3]:.2f}, "
                 f"sigma = {10 ** mu[4]:.3f}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=100)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--pred", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--what", nargs="+", default=["analysis", "gap", "timing", "figure"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timing-samples", type=int, default=3)
    args = ap.parse_args()
    d = dict(np.load(args.data))
    d["_path"] = str(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    if "analysis" in args.what:
        r = analysis(d)
        (args.out / "analysis.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    if "gap" in args.what:
        r = gap(d, args.pred, args.workers)
        (args.out / "gap.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    if "timing" in args.what:
        r = timing(d, args.timing_samples, args.out / "timing.json")
        (args.out / "timing.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    if "figure" in args.what:
        example_figure(d, args.pred, args.out / "example.png", "median")
        example_figure(d, args.pred, args.out / "example_worst.png", "worst")


if __name__ == "__main__":
    main()

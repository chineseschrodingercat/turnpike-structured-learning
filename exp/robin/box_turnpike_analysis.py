#!/usr/bin/env python3
"""Does the turnpike survive box constraints? Analysis of box_data.npz.

Checks, with layers taken from the longest horizon (T = 8):
  plateau   : u_8(4) vs the static constrained optimum u_c = clip(u_bar, u_a, u_b); q_8(4) vs q_bar
  q-superposition : q_T(t) ~ q_bar + l0(t) + l1(T - t)            (adjoint trace / alpha)
  u-superposition : u_T(t) ~ clip(q_bar + l0(t) + l1(T - t), u_a, u_b)
  clip baseline   : clip(u_unconstrained_T, u_a, u_b) vs u_T     (do constraints change more than the clipping?)
  late decay rate of the q layers vs lambda_1(beta) (prediction: saturated plateau -> open-loop rate)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators  # noqa: E402


def first_eig(beta):
    ops = assemble_operators(RobinConfig(beta=beta))
    return float(sla.eigh(ops.K_beta.toarray(), ops.M.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0])


def late_rate(layer: np.ndarray, dt: float, scale: float) -> float:
    """Decay rate fitted on the part of the layer between 1e-3 and 1e-6 of scale."""
    a = np.abs(layer)
    idx = np.nonzero((a < 1e-3 * scale) & (a > 1e-6 * scale))[0]
    if idx.size < 10:
        return np.nan
    k = idx[(idx >= idx[0]) & (idx <= idx[0] + 400)]
    slope = np.polyfit(k * dt, np.log(a[k]), 1)[0]
    return float(-slope)


def summ(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x)), "n": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data)
    dt = float(d["dt"])
    ubar, uc, qbar, lo, hi = d["u_bar"], d["u_c"], d["q_bar"], d["lo"], d["hi"]
    th_b = d["X"][:, 5]
    sat = th_b < 1.0
    Q8, U8 = d["Q_T8"], d["U_T8"]
    half = Q8.shape[1] // 2
    rel = lambda a, b: np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)

    plateau_u = np.abs(U8[:, half - 1] - uc) / ubar
    plateau_q = np.abs(Q8[:, half - 1] - qbar) / np.abs(qbar)
    L0 = Q8[:, :half] - qbar[:, None]
    L1 = Q8[:, ::-1][:, :half] - qbar[:, None]

    rep = {"n": int(len(ubar)), "frac_plateau_saturated": float(sat.mean()),
           "plateau_u_err_T8": summ(plateau_u), "plateau_q_err_T8": summ(plateau_q),
           "max_fixed_point_residual": float(max(d[f"res_T{T}"].max() for T in (1, 2, 4, 8))),
           "max_pdas_iters": int(max(d[f"iters_T{T}"].max() for T in (1, 2, 4, 8))),
           "active_fraction_T1": {"upper": summ(d["frac_upper_T1"]), "lower": summ(d["frac_lower_T1"])},
           "q_superposition": {}, "u_superposition": {}, "clip_of_unconstrained": {}}
    for T in (1, 2, 4, 8):
        Q, U = d[f"Q_T{T}"], d[f"U_T{T}"]
        N = Q.shape[1]
        if T < 8:
            k = np.arange(N)
            qh = qbar[:, None] + L0[:, k] + L1[:, N - 1 - k]
            uh = np.clip(qh, lo[:, None], hi[:, None])
            eq, eu = rel(qh, Q), rel(uh, U)
            rep["q_superposition"][str(T)] = {"all": summ(eq), "saturated": summ(eq[sat]), "free": summ(eq[~sat])}
            rep["u_superposition"][str(T)] = {"all": summ(eu), "saturated": summ(eu[sat]), "free": summ(eu[~sat])}
        ec = rel(np.clip(d[f"Uunc_T{T}"], lo[:, None], hi[:, None]), U)
        rep["clip_of_unconstrained"][str(T)] = {"all": summ(ec), "saturated": summ(ec[sat]), "free": summ(ec[~sat])}

    lam1 = np.array([first_eig(b) for b in d["beta"]])
    r0 = np.array([late_rate(L0[i], dt, abs(qbar[i])) for i in range(len(qbar))])
    r1 = np.array([late_rate(L1[i], dt, abs(qbar[i])) for i in range(len(qbar))])
    rep["late_rate_over_lambda1"] = {
        "initial_saturated": summ(r0[sat] / lam1[sat]), "initial_free": summ(r0[~sat] / lam1[~sat]),
        "terminal_saturated": summ(r1[sat] / lam1[sat]), "terminal_free": summ(r1[~sat] / lam1[~sat]),
    }
    (args.out_dir / "box_turnpike_analysis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, mask, title in ((axes[0], ~sat, "free plateau (theta_b > 1)"), (axes[1], sat, "saturated plateau (theta_b < 1)")):
        ids = np.nonzero(mask)[0]
        for j, color in zip(ids[np.linspace(0, len(ids) - 1, 3).astype(int)], ["C0", "C1", "C2"]):
            for T, ls in ((8, "-"), (2, "--"), (1, ":")):
                tt = np.arange(1, d[f"U_T{T}"].shape[1] + 1) * dt
                ax.plot(tt, d[f"U_T{T}"][j] / ubar[j], ls, color=color, lw=1.4 if T == 8 else 1.0)
            ax.plot(np.arange(1, Q8.shape[1] + 1) * dt, Q8[j] / ubar[j], color=color, alpha=0.3, lw=0.8)
        ax.set_xlim(0, 3.2); ax.set_title(f"u/u_bar (solid T=8, -- T=2, : T=1), faint = q; {title}")
        ax.set_xlabel("t"); ax.grid(alpha=0.3)
    ax = axes[2]
    for T in (1, 2, 4):
        e = rel(np.clip(qbar[:, None] + L0[:, np.arange(d[f'Q_T{T}'].shape[1])] + L1[:, d[f'Q_T{T}'].shape[1] - 1 - np.arange(d[f'Q_T{T}'].shape[1])],
                        lo[:, None], hi[:, None]), d[f"U_T{T}"])
        ax.scatter(th_b, e, s=5, label=f"T={T}")
    ax.set_yscale("log"); ax.axvline(1.0, color="k", lw=0.6); ax.set_xlabel("theta_b (u_b / u_bar)")
    ax.set_title("u-superposition error (T-invariant q-layers + clip)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "box_turnpike_analysis.png", dpi=140)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

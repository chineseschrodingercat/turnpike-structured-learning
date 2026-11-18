#!/usr/bin/env python3
"""One strongly nonlinear test sample of the nonlinear moving-target problem at T = 8: target, exact u(s, t),
TurnpikeNet (trained on T = 1), error, and profiles / time traces against CNN, the LQ control and the linear periodic
turnpike."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_semilinear_moving_data import cfg_of  # noqa: E402
from moving2d import eval_periodic, target  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--tp", type=pathlib.Path, required=True)
    ap.add_argument("--cnn", type=pathlib.Path, required=True)
    ap.add_argument("--q", type=float, default=0.85, help="quantile of |u - u_LQ| / |u_LQ| used to pick the sample")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    te = d["test_idx"]
    U = d["U_T8_test"].astype(np.float64)
    Ulq = d["U_lq_T8_test"].astype(np.float64)
    nl = np.array([rel(Ulq[r], U[r]) for r in range(len(te))])
    r = int(np.argmin(np.abs(nl - np.quantile(nl, args.q))))
    i = int(te[r])
    cf = cfg_of(d["X"][i], 8)
    gamma = float(10 ** d["X"][i, 7])
    t = np.arange(1, cf.Nt + 1) * cf.dt
    s = np.arange(1, 100) / 100
    u, up, uc, ul = U[r], np.load(args.tp)["U_T8"][r].astype(np.float64), np.load(args.cnn)["U_T8"][r].astype(np.float64), Ulq[r]
    uh = eval_periodic(d["uhat_lin"][i].astype(np.complex128), cf.omega, cf.phi0, t)
    Yd = target(cf, s[None, :], cf.omega * t[:, None] + cf.phi0)
    v = np.abs(u).max()
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.6))

    def heat(ax, Z, lo, hi, ttl, **kw):
        sel = (t >= lo) & (t <= hi)
        pc = ax.pcolormesh(t[sel], s, Z[sel].T, shading="auto", **kw)
        ax.set_xlabel("t")
        ax.set_ylabel("s (along the edge)")
        ax.set_title(ttl, fontsize=9.5)
        fig.colorbar(pc, ax=ax)

    heat(axes[0, 0], Yd, 0, 2, f"target y_d(s, t), t in [0, 2]: a = {cf.a:.2f}, omega = {cf.omega / np.pi:.2f} pi, sigma = {cf.sigma:.3f}", cmap="viridis")
    heat(axes[0, 1], u, 0, 2, f"exact nonlinear u(s, t) (Newton-CG), gamma_eff = {gamma:.0f}", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(axes[0, 2], up, 0, 2, f"TurnpikeNet, trained on T = 1 only ({100 * rel(up, u):.3f}% on [0, 8])", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(axes[0, 3], np.maximum(np.abs(up - u) / v, 1e-8), 0, 8,
         f"TurnpikeNet |error| / max|u|; CNN {100 * rel(uc, u):.2f}%, LQ {100 * rel(ul, u):.1f}%", cmap="magma", norm=LogNorm(1e-6, 1e-1))
    heat(axes[1, 0], u - ul, 0, 2, "exact nonlinear minus LQ control (the nonlinear effect)", cmap="RdBu_r",
         vmin=-np.abs(u - ul).max(), vmax=np.abs(u - ul).max())
    ax = axes[1, 1]
    k = int(round(4.0 / cf.dt))
    for lab, z, st in (("exact", u, dict(color="k", lw=2.2)), ("TurnpikeNet", up, dict(color="C3", lw=1.2, ls="--")),
                       ("CNN", uc, dict(color="C0", lw=1.0, ls=":")), ("LQ control", ul, dict(color="C2", lw=1.0, ls="-.")),
                       ("linear periodic turnpike", uh, dict(color="C4", lw=0.8, ls="-."))):
        ax.plot(s, z[k], label=lab, **st)
    ax.set_xlabel("s")
    ax.set_title("profile at t = 4 (periodic regime)", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    j = int(np.argmin(np.abs(s - 0.5)))
    for c, (lo, hi, ttl) in enumerate([(0, 1.0, "time trace at s = 0.5, t in [0, 1]"), (7.0, 8.0, "time trace at s = 0.5, t in [7, 8]")]):
        ax = axes[1, 2 + c]
        sel = (t >= lo) & (t <= hi)
        ax.plot(t[sel], u[sel, j], "k", lw=2.0, label="exact")
        ax.plot(t[sel], up[sel, j], "r--", lw=1.2, label="TurnpikeNet")
        ax.plot(t[sel], uc[sel, j], "C0:", lw=1.0, label="CNN")
        ax.plot(t[sel], ul[sel, j], "C2-.", lw=0.9, label="LQ control")
        ax.set_xlabel("t")
        ax.set_title(ttl, fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
    sv = np.linalg.svd(u, compute_uv=False)
    fig.suptitle(f"Nonlinear (gamma y^3) moving hot-spot target, distributed edge control (n = 9900, m = 99), T = 8, test sample #{i}: "
                 f"w = {cf.w:.2f}, alpha = {cf.alpha:.1e}, beta = {cf.beta:.2f};  rank@1e-3 = {int(np.sum(sv > 1e-3 * sv[0]))}, "
                 f"s2/s1 = {sv[1] / sv[0]:.2f}, |u - u_LQ|/|u_LQ| = {100 * nl[r]:.0f}%", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("sample", i, "tp", rel(up, u), "cnn", rel(uc, u), "lq", rel(ul, u), "lin_per", rel(uh, u))


if __name__ == "__main__":
    main()

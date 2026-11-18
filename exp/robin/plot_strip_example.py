#!/usr/bin/env python3
"""One test sample of the edge-strip problem (section 18b): target y_d(y) = 1 + sum_{k<=6} tau_k sin(k pi y) on
omega = [0, w] x [0.05, 0.95], distributed control u(s, t) on x = 0 (m = 99), T = 8.
Exact (PCG) vs tau-linear TurnpikeNet trained on T = 1: target vs plateau, heatmaps, error, profiles, time traces,
plateau state. The sample is the one with median TurnpikeNet error at T = 8 (or --pick worst)."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_2d_strip_data import cfg_of  # noqa: E402
from plot_2d_examples import field  # noqa: E402
from robin2d_multi import assemble_strip  # noqa: E402

T = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--pred", type=pathlib.Path, required=True)
    ap.add_argument("--cnn", type=pathlib.Path, default=None)
    ap.add_argument("--pick", choices=["median", "worst", "small_alpha"], default="median",
                    help="small_alpha: median-error sample among the 20%% of test samples with the smallest alpha (cheap control)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d, P = np.load(args.data), np.load(args.pred)
    te, dt, m = d["test_idx"], float(d["dt"]), int(d["m"])
    U = d["U_T8_test"].astype(np.float64)
    Up = P["U_T8"].astype(np.float64)
    err = np.linalg.norm((Up - U).reshape(len(te), -1), axis=1) / np.linalg.norm(U.reshape(len(te), -1), axis=1)
    if args.pick == "median":
        r = int(np.argsort(err)[len(err) // 2])
    elif args.pick == "worst":
        r = int(np.argmax(err))
    else:
        la_te = d["X"][te, 1]
        cand = np.nonzero(la_te <= np.quantile(la_te, 0.2))[0]
        r = int(cand[np.argsort(err[cand])[len(cand) // 2]])
    i = int(te[r])
    row = d["X"][i]
    w, la, lb, tau = float(row[0]), float(row[1]), float(row[2]), row[3:9]
    u, up = U[r], Up[r]
    ub = d["u_bar"][i]
    e_cnn = None
    if args.cnn is not None:
        Uc = np.load(args.cnn)["U_T8"].astype(np.float64)
        e_cnn = float(np.linalg.norm(Uc[r] - u) / np.linalg.norm(u))
    t = np.arange(1, u.shape[0] + 1) * dt
    s = np.arange(1, m + 1) / (m + 1)
    sv = np.linalg.svd(u, compute_uv=False)
    svl = np.linalg.svd(u - ub[None, :], compute_uv=False)

    mi = assemble_strip(cfg_of(row, T), m)
    o = mi.base
    lu = spla.splu(o.A.tocsc())
    Mdt = (o.M / dt).tocsr()
    y = np.zeros(o.n)
    for k in range(int(round(4.0 / dt))):
        y = lu.solve(Mdt @ y + mi.B @ u[k])

    fig = plt.figure(figsize=(20, 9.8))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    yy = np.linspace(0, 1, 400)
    yd = 1 + sum(tau[k] * np.sin((k + 1) * np.pi * yy) for k in range(6))
    ax.plot(yy, yd, "C1", lw=1.8, label="target y_d(y) on the strip")
    ax.axvspan(0, 0.05, color="gray", alpha=0.15)
    ax.axvspan(0.95, 1, color="gray", alpha=0.15)
    ax.set_xlabel("y  (= s along the edge)")
    ax.set_ylabel("target", color="C1")
    ax2 = ax.twinx()
    ax2.plot(s, ub, "k", lw=1.8, label="plateau u_bar(s) (exact static optimum)")
    ax2.plot(s, up[1279], "r--", lw=1.0, label="TurnpikeNet u(s, t=4)")
    ax2.set_ylabel("control")
    ax.set_title(f"target vs plateau control; tau = [{', '.join(f'{v:+.2f}' for v in tau)}]", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)

    vmin, vmax = min(u.min(), up.min()), max(u.max(), up.max())
    for c, (Z, ttl) in enumerate([(u, "exact u(s, t) (PCG)"), (up, "tau-linear TurnpikeNet, trained on T = 1 only")]):
        axh = fig.add_subplot(gs[0, 1 + c])
        pc = axh.pcolormesh(t, s, Z.T, shading="auto", cmap="RdBu_r", vmin=-max(abs(vmin), abs(vmax)), vmax=max(abs(vmin), abs(vmax)))
        axh.set_xlabel("t")
        axh.set_ylabel("s (position along the edge x = 0)")
        axh.set_title(ttl, fontsize=9.5)
        fig.colorbar(pc, ax=axh)
    axe = fig.add_subplot(gs[0, 3])
    E = np.maximum(np.abs(up - u) / np.max(np.abs(ub)), 1e-8)
    pc = axe.pcolormesh(t, s, E.T, shading="auto", cmap="magma", norm=LogNorm(1e-6, 1e-1))
    axe.set_xlabel("t")
    axe.set_ylabel("s")
    axe.set_title(f"|error| / max|u_bar|   (rel. L2 {100 * err[r]:.3f}%" + (f"; CNN {100 * e_cnn:.2f}%)" if e_cnn is not None else ")"), fontsize=9.5)
    fig.colorbar(pc, ax=axe)

    ax = fig.add_subplot(gs[1, 0])
    for tt, col in zip([0.006, 0.03, 0.1, 4.0, 7.9, 7.98], ["C3", "C1", "C2", "k", "C0", "C4"]):
        k = int(round(tt / dt)) - 1
        ax.plot(s, u[k], color=col, lw=1.8, label=f"t = {tt}")
        ax.plot(s, up[k], color=col, lw=1.0, ls="--")
    ax.set_xlabel("s")
    ax.set_ylabel("u(s, t)")
    ax.set_title("profiles along the edge (solid exact, dashed TurnpikeNet)", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)

    inner = np.nonzero((s > 0.15) & (s < 0.85))[0]
    j_max, j_min = int(inner[np.argmax(ub[inner])]), int(inner[np.argmin(ub[inner])])
    j_far = int(inner[np.argmax(np.minimum(np.abs(s[inner] - s[j_max]), np.abs(s[inner] - s[j_min])))])
    j_pick = [j_max, j_min, j_far]
    for c, (lo, hi, ttl) in enumerate([(0, 0.3, "time traces, initial layer"), (T - 0.3, T, "time traces, terminal layer")]):
        ax = fig.add_subplot(gs[1, 1 + c])
        for j, col in zip(j_pick, ["C3", "C0", "C2"]):
            ax.plot(t, u[:, j], color=col, lw=1.8, label=f"s = {s[j]:.2f}")
            ax.plot(t, up[:, j], color=col, lw=1.0, ls="--")
            ax.axhline(ub[j], color=col, lw=0.6, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("t")
        ax.set_title(ttl + " (dotted = u_bar(s))", fontsize=9.5)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[1, 3])
    F = field(y)
    im = ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-np.abs(F).max(), vmax=np.abs(F).max(), interpolation="bilinear")
    ax.add_patch(Rectangle((0, 0.05), w, 0.9, fill=False, ec="k", lw=1.4, ls="--"))
    ax.plot([0, 0], [0, 1], color="lime", lw=5, solid_capstyle="butt")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("state at t = 4 (exact control); dashed = strip", fontsize=9.5)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"Edge-strip target, distributed edge control u(s, t), m = 99, n = 9900, T = 8 ({args.pick}-error test sample #{i}): "
                 f"w = {w:.2f}, alpha = {10**la:.2g}, beta = {10**lb:.2g};  singular values of u(s,t): s2/s1 = {sv[1] / sv[0]:.1e}; "
                 f"of u - u_bar: s2/s1 = {svl[1] / svl[0]:.2f}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=125)
    print("sample", i, "err", float(err[r]), "cnn", e_cnn, "sv", sv[:4] / sv[0], "layer sv", svl[:5] / svl[0])


if __name__ == "__main__":
    main()

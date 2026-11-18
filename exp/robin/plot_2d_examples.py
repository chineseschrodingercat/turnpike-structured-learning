#!/usr/bin/env python3
"""What the 2D controls look like (observation rectangle inside the unit square, control on the edge x = 0), T = 8.

Figure 1 (scalar control, robin2d): state snapshots y(x, y, t) under the exact control, the control u(t)
exact vs TurnpikeNet (trained on T = 1), and the mean state in the rectangle.
Figure 2 (multi-input control u(s, t), robin2d_mi, m = 99): exact and learned u(s, t), error, s-profiles at
several times, time traces at several s, and the plateau state.
The test sample is the one with median TurnpikeNet error at T = 8.
"""

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
from robin2d import Cfg2D, assemble  # noqa: E402
from robin2d_mi import assemble_mi  # noqa: E402

NH, T = 100, 8


def field(y):
    """Free-node vector -> (nh+1) x (nh+1) array on the full grid (Dirichlet zeros added), rows = y, cols = x."""
    F = np.zeros((NH + 1, NH + 1))
    F[1:NH, 0:NH] = y.reshape(NH - 1, NH)
    return F


def draw_domain(ax, g, F, vmax, title):
    im = ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")
    ax.add_patch(Rectangle((g["xa"], g["ya"]), g["xb"] - g["xa"], g["yb"] - g["ya"], fill=False, ec="cyan", lw=1.6, ls="--"))
    ax.plot([0, 0], [0, 1], color="lime", lw=5, solid_capstyle="butt")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def forward_snapshots(ops, rhs_of_k, times, dt):
    lu = spla.splu(ops.A.tocsc())
    Mdt = (ops.M / dt).tocsr()
    want = {int(round(t / dt)) - 1: t for t in times}
    y, out, mean = np.zeros(ops.n), {}, []
    x = (np.arange(NH * (NH - 1)) % NH) / NH
    yy = (np.arange(NH * (NH - 1)) // NH + 1) / NH
    c = ops.cfg
    mask = (x >= c.xa) & (x <= c.xb) & (yy >= c.ya) & (yy <= c.yb)
    for k in range(max(want) + 1):
        y = lu.solve(Mdt @ y + rhs_of_k(k))
        mean.append(y[mask].mean())
        if k in want:
            out[want[k]] = y.copy()
    return out, np.array(mean)


def geo(d, i):
    return {k: float(d[k][i]) for k in ("xa", "xb", "ya", "yb", "alpha", "beta")}


def fig_scalar(data, pred, out):
    d, P = np.load(data), np.load(pred)
    te, dt = d["test_idx"], float(d["dt"])
    U, Up = d["U_T8"][te], P["U_T8"]
    err = np.linalg.norm(Up - U, axis=1) / np.linalg.norm(U, axis=1)
    r = int(np.argsort(err)[len(err) // 2])
    i = int(te[r])
    g = geo(d, i)
    ub = float(d["u_bar"][i])
    ops = assemble(Cfg2D(nh=NH, T=float(T), Nt=320 * T, **g))
    times = [0.03, 0.3, 4.0, 7.97]
    snaps, mean = forward_snapshots(ops, lambda k: ops.b * U[r][k], times, dt)
    t = np.arange(1, U.shape[1] + 1) * dt
    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], height_ratios=[1, 0.95])
    vmax = max(np.max(v) for v in snaps.values())
    for c, tt in enumerate(times):
        ax = fig.add_subplot(gs[0, c])
        im = draw_domain(ax, g, field(snaps[tt]), vmax, f"state y(x,y,t) at t = {tt}")
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 4]))
    for c, (lo, hi, ttl) in enumerate([(0, T, "control u(t) on the edge x = 0 (uniform in y)"), (0, 0.5, "initial layer (zoom)"),
                                       (T - 0.5, T, "terminal layer (zoom)")]):
        ax = fig.add_subplot(gs[1, c])
        ax.plot(t, U[r], "k", lw=2.2, label="exact (reduced exact method)")
        ax.plot(t, Up[r], "r--", lw=1.4, label=f"TurnpikeNet, trained on T=1 only ({100 * err[r]:.3f}%)")
        ax.axhline(ub, color="gray", ls="--", lw=0.8, label="plateau u_bar (static optimum)")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("t")
        ax.set_title(ttl, fontsize=9.5)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel("u(t)")
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(np.arange(1, len(mean) + 1) * dt, mean, "C0")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="target y_d = 1")
    ax.set_xlabel("t")
    ax.set_title("mean state inside the rectangle", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"2D, scalar boundary control (n = 9900), test sample with median error: rectangle [{g['xa']:.2f},{g['xb']:.2f}] x "
                 f"[{g['ya']:.2f},{g['yb']:.2f}], alpha = {g['alpha']:.2g}, beta = {g['beta']:.2g}, T = 8   "
                 f"(green = control edge, cyan = observation rectangle)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=125)
    print("scalar sample", i, g, "err", err[r])


def fig_mi(data, pred, out):
    d, P = np.load(data), np.load(pred)
    te, dt, m = d["test_idx"], float(d["dt"]), int(d["m"])
    U = d["U_T8_test"].astype(np.float64)
    Up = P["U_T8"].astype(np.float64)
    err = np.linalg.norm((Up - U).reshape(len(te), -1), axis=1) / np.linalg.norm(U.reshape(len(te), -1), axis=1)
    r = int(np.argsort(err)[len(err) // 2])
    i = int(te[r])
    g = geo(d, i)
    ub = d["u_bar"][i]
    u, up = U[r], Up[r]  # (N, m)
    t = np.arange(1, u.shape[0] + 1) * dt
    s = np.arange(1, m + 1) / (m + 1)
    sv = np.linalg.svd(u, compute_uv=False)
    mi = assemble_mi(Cfg2D(nh=NH, T=float(T), Nt=320 * T, **g), m)
    snaps, mean = forward_snapshots(mi.base, lambda k: mi.B @ u[k], [0.03, 4.0], dt)

    fig = plt.figure(figsize=(20, 9.6))
    gs = fig.add_gridspec(2, 4)
    vmin, vmax = min(u.min(), up.min()), max(u.max(), up.max())

    def heat(ax, Z, tl, th, ttl, **kw):
        k0, k1 = int(tl / dt), int(round(th / dt))
        pc = ax.pcolormesh(t[k0:k1], s, Z[k0:k1].T, shading="auto", **kw)
        for yv in (g["ya"], g["yb"]):
            ax.axhline(yv, color="w", ls="--", lw=1)
        ax.set_xlabel("t")
        ax.set_ylabel("s (position along the edge x = 0)")
        ax.set_title(ttl, fontsize=9.5)
        return pc

    pc = heat(fig.add_subplot(gs[0, 0]), u, 0, T, "exact u(s,t), whole horizon", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=fig.axes[-1])
    pc = heat(fig.add_subplot(gs[0, 1]), u, 0, 0.3, "exact u(s,t), initial layer t in [0, 0.3]", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=fig.axes[-1])
    pc = heat(fig.add_subplot(gs[0, 2]), up, 0, 0.3, "TurnpikeNet (trained on T=1), same window", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=fig.axes[-1])
    E = np.abs(up - u) / np.max(np.abs(ub))
    pc = heat(fig.add_subplot(gs[0, 3]), np.maximum(E, 1e-7), 0, T, f"|error| / max|u_bar|  (rel. L2 {100 * err[r]:.3f}%)",
              cmap="magma", norm=LogNorm(1e-6, 1e-1))
    fig.colorbar(pc, ax=fig.axes[-1])

    ax = fig.add_subplot(gs[1, 0])
    ax.axvspan(g["ya"], g["yb"], color="cyan", alpha=0.12, label="y-range of the rectangle")
    for tt, col in zip([0.006, 0.03, 0.1, 4.0, 7.95, 7.99], ["C3", "C1", "C2", "k", "C0", "C4"]):
        k = int(round(tt / dt)) - 1
        ax.plot(s, u[k], color=col, lw=1.8, label=f"t = {tt}")
        ax.plot(s, up[k], color=col, lw=1.0, ls="--")
    ax.set_xlabel("s")
    ax.set_ylabel("u(s, t)")
    ax.set_title("profiles along the edge (solid exact, dashed TurnpikeNet)", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)

    picks = [("rectangle centre", 0.5 * (g["ya"] + g["yb"])), ("rectangle edge", g["yb"]), ("far from it", 0.08 if g["ya"] > 0.5 else 0.92)]
    for c, (lo, hi, ttl) in enumerate([(0, 0.4, "time traces, initial layer"), (T - 0.4, T, "time traces, terminal layer")]):
        ax = fig.add_subplot(gs[1, 1 + c])
        for (lab, sv_), col in zip(picks, ["C3", "C2", "C0"]):
            j = int(np.argmin(np.abs(s - sv_)))
            ax.plot(t, u[:, j], color=col, lw=1.8, label=f"{lab}, s = {s[j]:.2f}")
            ax.plot(t, up[:, j], color=col, lw=1.0, ls="--")
            ax.axhline(ub[j], color=col, lw=0.6, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("t")
        ax.set_title(ttl + " (dotted = u_bar(s))", fontsize=9.5)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7.5)
    ax = fig.add_subplot(gs[1, 3])
    F = field(snaps[4.0])
    im = draw_domain(ax, g, F, F.max(), "state at t = 4 (plateau) under the exact u(s,t)")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"2D, distributed edge control u(s,t), m = 99 (n = 9900), median-error test sample: rectangle "
                 f"[{g['xa']:.2f},{g['xb']:.2f}] x [{g['ya']:.2f},{g['yb']:.2f}], alpha = {g['alpha']:.2g}, beta = {g['beta']:.2g}, "
                 f"T = 8;  singular values of u(s,t): s2/s1 = {sv[1] / sv[0]:.1e}, s3/s1 = {sv[2] / sv[0]:.1e}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=125)
    print("mi sample", i, g, "err", err[r], "sv ratios", sv[1] / sv[0], sv[2] / sv[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar-data", type=pathlib.Path, required=True)
    ap.add_argument("--scalar-pred", type=pathlib.Path, required=True)
    ap.add_argument("--mi-data", type=pathlib.Path, required=True)
    ap.add_argument("--mi-pred", type=pathlib.Path, default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_scalar(args.scalar_data, args.scalar_pred, args.out_dir / "example_2d_scalar.png")
    if args.mi_pred is not None:
        fig_mi(args.mi_data, args.mi_pred, args.out_dir / "example_2d_multiinput.png")


if __name__ == "__main__":
    main()

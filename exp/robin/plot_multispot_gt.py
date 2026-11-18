#!/usr/bin/env python3
"""Ground-truth controls of the multi-spot (quasi-periodic) pilot, runs/multispot/<case>.npz:
fig 1 (overview, t in [0, 8]): target, exact linear u, exact semilinear u, |linear - quasi-periodic reference|, time traces;
fig 2 (zoom, t in [0, 2.5]): target and semilinear control, plus the nonlinear effect u_nl - u_lin."""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

D = pathlib.Path("runs/multispot")
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else D
S = json.loads((D / "summary.json").read_text())
cases = list(S)
s = np.arange(1, 100) / 100
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})


def heat(fig, ax, t, Z, title, **kw):
    pc = ax.pcolormesh(t, s, Z.T, shading="auto", rasterized=True, **kw)
    ax.set_title(title)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$s$")
    fig.colorbar(pc, ax=ax, pad=0.02)


fig, axes = plt.subplots(len(cases), 5, figsize=(24, 3.9 * len(cases)), gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1.2, 1]}, squeeze=False)
for r, c in enumerate(cases):
    z, d = np.load(D / f"{c}.npz"), S[c]
    Cl, Cn, R, Yd = (z[k].astype(float) for k in ("C_lin_T8", "C_nl_T8", "R_T8", "Yd_T8"))
    t = np.arange(1, Cl.shape[0] + 1) / 320
    v = max(np.abs(Cl).max(), np.abs(Cn).max())
    heat(fig, axes[r, 0], t, Yd, f"{c}: target $y_d(s,t)$ (rank {d['rank_target']})", cmap="viridis")
    heat(fig, axes[r, 1], t, Cl, f"exact linear control (rank {d['rank_lin']}, $\\sigma_2/\\sigma_1$ = {d['s2s1_lin']:.2f})", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(fig, axes[r, 2], t, Cn, f"exact semilinear, $\\gamma$ = {d['params']['gamma']:g} (rank {d['rank_nl']})", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(fig, axes[r, 3], t, np.maximum(np.abs(Cl - R) / np.abs(Cl).max(), 1e-8),
         f"|linear - quasi-periodic reference| / max|u|  (interior {100 * d['lin_vs_ref_interior']:.1e}%)", cmap="magma", norm=LogNorm(1e-7, 1))
    ax = axes[r, 4]
    j = int(np.argmax(np.abs(Cn).mean(axis=0)))
    ax.plot(t, Cl[:, j], "C0", lw=1.0, label="exact linear")
    ax.plot(t, R[:, j], "k:", lw=0.9, label="quasi-periodic reference (linear)")
    ax.plot(t, Cn[:, j], "C3", lw=1.0, label=f"exact semilinear ({100 * d['nl_vs_lin_interior']:.0f}% from linear)")
    ax.set_xlabel("$t$")
    ax.set_title(f"time traces at $s$ = {s[j]:.2f}")
    ax.legend(fontsize=7, loc="lower center")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "multispot_gt_overview.png", dpi=110)

fig, axes = plt.subplots(len(cases), 4, figsize=(19, 3.9 * len(cases)), squeeze=False)
for r, c in enumerate(cases):
    z, d = np.load(D / f"{c}.npz"), S[c]
    Cl, Cn, Yd, Cl2, Cn2 = (z[k].astype(float) for k in ("C_lin_T8", "C_nl_T8", "Yd_T8", "C_lin_T2", "C_nl_T2"))
    t = np.arange(1, Cl.shape[0] + 1) / 320
    w = t <= 2.5
    v = np.abs(Cn).max()
    heat(fig, axes[r, 0], t[w], Yd[w], f"{c}: target, $t\\in[0,2.5]$", cmap="viridis")
    heat(fig, axes[r, 1], t[w], Cn[w], "exact semilinear control, $T=8$", cmap="RdBu_r", vmin=-v, vmax=v)
    dv = np.abs(Cn - Cl)[w].max()
    heat(fig, axes[r, 2], t[w], (Cn - Cl)[w], "nonlinear effect $u_{\\rm nl}-u_{\\rm lin}$", cmap="PuOr_r", vmin=-dv, vmax=dv)
    ax = axes[r, 3]
    j = int(np.argmax(np.abs(Cn).mean(axis=0)))
    t2 = np.arange(1, Cn2.shape[0] + 1) / 320
    ax.plot(t, Cn[:, j], "C3", lw=1.2, label="semilinear, $T=8$")
    ax.plot(t2, Cn2[:, j], "k--", lw=1.0, label="semilinear, $T=2$")
    ax.set_xlim(0, 2.5)
    ax.set_xlabel("$t$")
    ax.set_title(f"$s$ = {s[j]:.2f}: start of $T=2$ vs $T=8$ (window [0,0.5] differs by {100 * d['init_window_T2_vs_T8_nl']:.2g}%)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "multispot_gt_zoom.png", dpi=110)
print("wrote", out)

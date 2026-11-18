#!/usr/bin/env python3
"""Compact publication versions of the three example figures (no sample ids or debugging info in titles):
fig_ex_strip (2D semilinear, strip target), fig_ex_moving (2D semilinear, moving target), fig_ex_advect (advection).
Same samples as plot_semilinear2d_mi_example.py (q = 0.9), plot_semilinear_moving_example.py (q = 0.85) and
plot_advect_gt.py (median and worst)."""

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
from moving2d import target  # noqa: E402

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7.5, "axes.grid": False})
EXACT = dict(color="k", lw=1.8)
OURS = dict(color="C3", lw=1.1, ls="--")


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def heat(fig, ax, t, s, Z, title, **kw):
    pc = ax.pcolormesh(t, s, Z.T, shading="auto", rasterized=True, **kw)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$s$")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, pad=0.02)


def errmap(fig, ax, t, s, E, title):
    heat(fig, ax, t, s, np.maximum(E, 1e-7), title, cmap="magma", norm=LogNorm(1e-6, 1e-1))


# ---------------- 2D semilinear, strip target, T = 8 ----------------
d = np.load("runs/semilinear2d_mi/data.npz")
P = "runs/semilinear2d_mi/models/pred"
te, dt = d["test_idx"], float(d["dt"])
nl = np.linalg.norm(d["u_bar"][te] - d["u_bar_lq"][te], axis=1) / np.linalg.norm(d["u_bar_lq"][te], axis=1)
r = int(np.argmin(np.abs(nl - np.quantile(nl, 0.9))))
i = int(te[r])
row = d["X"][i]
u = d["U_T8_test"][r].astype(float)
up = np.load(f"{P}/pred_turnpike_r8_seed0.npz")["U_T8"][r].astype(float)
ut = np.load(f"{P}/pred_tau_turnpike_r8_seed0.npz")["U_T8"][r].astype(float)
ub, ubl = d["u_bar"][i], d["u_bar_lq"][i]
t = np.arange(1, u.shape[0] + 1) * dt
s = np.arange(1, 100) / 100
fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
a = ax[0, 0]
yy = np.linspace(0, 1, 400)
a.plot(yy, 1 + sum(row[3 + k] * np.sin((k + 1) * np.pi * yy) for k in range(6)), color="C1", lw=1.4, label=r"target $y_d(s)$")
a2 = a.twinx()
a2.plot(s, ub, "k", lw=1.6, label=r"reference $\bar u(s)$ (nonlinear)")
a2.plot(s, ubl, "C2:", lw=1.4, label=r"linear superposition")
h1, l1 = a.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=7)
a.set_xlabel("$s$")
a.set_ylabel("target")
a2.set_ylabel("control")
a.set_title("(a) target and steady controls")
v = np.abs(u).max()
heat(fig, ax[0, 1], t, s, u, "(b) exact control $u^*_T(s,t)$", cmap="RdBu_r", vmin=-v, vmax=v)
errmap(fig, ax[0, 2], t, s, np.abs(up - u) / v, f"(c) ours: pointwise error (rel. $L^2$ {100 * rel(up, u):.2f}%)")
a = ax[1, 0]
for tt, col in zip([0.006, 0.05, 4.0, 7.95], ["C0", "C1", "C2", "C4"]):
    k = int(round(tt / dt)) - 1
    a.plot(s, u[k], color=col, lw=1.6, label=f"$t$ = {tt}")
    a.plot(s, up[k], color=col, lw=1.0, ls="--")
a.set_xlabel("$s$")
a.set_title("(d) profiles (solid exact, dashed ours)")
a.legend(fontsize=7)
inner = np.nonzero((s > 0.15) & (s < 0.85))[0]
jj = [int(inner[np.argmax(ub[inner])]), int(inner[np.argmin(ub[inner])])]
for c, (lo, hi, ttl) in enumerate([(0, 0.3, "(e) near $t=0$"), (7.7, 8.0, "(f) near $t=T$")]):
    a = ax[1, 1 + c]
    for j, col in zip(jj, ["C0", "C3"]):
        a.plot(t, u[:, j], color=col, lw=1.8, label=f"exact, $s$ = {s[j]:.2f}")
        a.plot(t, up[:, j], color="k", lw=0.9, ls="--", label="ours" if j == jj[0] else None)
        a.plot(t, ut[:, j], color=col, lw=0.9, ls=":", label="linear superposition" if j == jj[0] else None)
    a.set_xlim(lo, hi)
    a.set_xlabel("$t$")
    a.set_title(ttl)
    if c == 0:
        a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "fig_ex_strip.pdf", dpi=200)
print("strip sample", i, "nl", float(nl[r]), "ours", rel(up, u), "lin", rel(ut, u))

# ---------------- 2D semilinear, moving target, T = 8 ----------------
d = np.load("runs/semilinear_moving/data.npz")
P = "runs/semilinear_moving/models/pred"
te = d["test_idx"]
U = d["U_T8_test"].astype(float)
Ulq = d["U_lq_T8_test"].astype(float)
nl = np.array([rel(Ulq[k], U[k]) for k in range(len(te))])
r = int(np.argmin(np.abs(nl - np.quantile(nl, 0.85))))
i = int(te[r])
cf = cfg_of(d["X"][i], 8)
t = np.arange(1, cf.Nt + 1) * cf.dt
u, ul = U[r], Ulq[r]
up = np.load(f"{P}/pred_tp_local_seed0.npz")["U_T8"][r].astype(float)
uc = np.load(f"{P}/pred_cnn_seed0.npz")["U_T8"][r].astype(float)
Yd = target(cf, s[None, :], cf.omega * t[:, None] + cf.phi0)
v = np.abs(u).max()
w2 = t <= 2.0
fig = plt.figure(figsize=(10, 5.6))
gs = fig.add_gridspec(2, 3)
heat(fig, fig.add_subplot(gs[0, 0]), t[w2], s, Yd[w2], "(a) target $y_d(s,t)$", cmap="viridis")
heat(fig, fig.add_subplot(gs[0, 1]), t[w2], s, u[w2], "(b) exact control $u^*_T(s,t)$", cmap="RdBu_r", vmin=-v, vmax=v)
errmap(fig, fig.add_subplot(gs[0, 2]), t, s, np.abs(up - u) / v, f"(c) ours: pointwise error (rel. $L^2$ {100 * rel(up, u):.2f}%)")
dv = np.abs(u - ul).max()
heat(fig, fig.add_subplot(gs[1, 0]), t[w2], s, (u - ul)[w2], "(d) nonlinear effect $u^* - u^*_{\\rm LQ}$", cmap="RdBu_r", vmin=-dv, vmax=dv)
a = fig.add_subplot(gs[1, 1])
k = int(round(4.0 / cf.dt))
a.plot(s, u[k], label="exact", **EXACT)
a.plot(s, up[k], label="ours", **OURS)
a.plot(s, uc[k], "C0:", lw=1.0, label="CNN")
a.plot(s, ul[k], "C2-.", lw=1.0, label="LQ control")
a.set_xlabel("$s$")
a.set_title("(e) profile at $t=4$")
a.legend(fontsize=7)
sub = gs[1, 2].subgridspec(1, 2, wspace=0.08)
j = int(np.argmin(np.abs(s - 0.5)))
for c, (lo, hi) in enumerate([(0, 1.0), (7.0, 8.0)]):
    a = fig.add_subplot(sub[0, c])
    sel = (t >= lo) & (t <= hi)
    a.plot(t[sel], u[sel, j], **EXACT)
    a.plot(t[sel], up[sel, j], **OURS)
    a.plot(t[sel], uc[sel, j], "C0:", lw=1.0)
    a.plot(t[sel], ul[sel, j], "C2-.", lw=1.0)
    a.set_xlabel("$t$")
    if c == 1:
        a.set_yticklabels([])
    a.set_title("(f) $s=0.5$, $t\\in[0,1]$" if c == 0 else "$t\\in[7,8]$")
fig.tight_layout()
fig.savefig(out / "fig_ex_moving.pdf", dpi=200)
print("moving sample", i, "ours", rel(up, u), "cnn", rel(uc, u), "lq", rel(ul, u))

# ---------------- advection, T = 4 (trained on T = 2) ----------------
d = np.load("runs/advect_data/advect_data.npz")
U4, te = d["U_T4_test"].astype(float), d["test_idx"]
Up = np.load("runs/advect_data/pred/pred_tp_transport_mirror_w48_seed0.npz")["U_T4"].astype(float)
err = np.linalg.norm((Up - U4).reshape(len(te), -1), axis=1) / np.linalg.norm(U4.reshape(len(te), -1), axis=1)
order = np.argsort(err)
t = np.arange(1, U4.shape[1] + 1) / 320.0
fig, ax = plt.subplots(2, 4, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1]})
for rr, (lab, jx) in enumerate([("median", order[len(order) // 2]), ("worst", order[-1])]):
    u, up = U4[jx], Up[jx]
    v = np.quantile(np.abs(u), 0.995)
    heat(fig, ax[rr, 0], t, s, u, f"exact ({lab} sample)", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(fig, ax[rr, 1], t, s, up, f"ours (rel. $L^2$ error {100 * err[jx]:.2f}%)", cmap="RdBu_r", vmin=-v, vmax=v)
    ve = np.quantile(np.abs(up - u), 0.999)
    heat(fig, ax[rr, 2], t, s, up - u, f"ours $-$ exact (scale {100 * ve / v:.1f}% of $u$)", cmap="PuOr_r", vmin=-ve, vmax=ve)
    a = ax[rr, 3]
    a.semilogy(t, np.linalg.norm(u, axis=1), "k", lw=1.2, label=r"$\|u^*(\cdot,t)\|$")
    a.semilogy(t, np.linalg.norm(up - u, axis=1), "C3", lw=1.1, label=r"$\|\hat u(\cdot,t)-u^*(\cdot,t)\|$")
    a.set_ylim(1e-6, None)
    a.set_xlabel("$t$")
    a.set_title("error over time")
    a.legend(fontsize=7, loc="lower center")
    for c in range(4):
        for tw in (1.0, 3.0):
            ax[rr, c].axvline(tw, color="gray", lw=0.6, ls="--")
fig.tight_layout()
fig.savefig(out / "fig_ex_advect.pdf", dpi=200)
print("advect median", float(np.median(err)), "max", float(err.max()))

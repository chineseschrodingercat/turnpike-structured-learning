#!/usr/bin/env python3
"""Figures for the manuscript draft: fig_turnpike (1D decomposition), fig_transfer (frozen fresh 1D test),
fig_cost (accuracy vs online time). Numbers for fig_cost are the single-thread medians recorded in the log (§28-29)."""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# ---------------- fig_turnpike ----------------
d = np.load("runs/turnpike/turnpike_data.npz")
te = d["test_idx"]
la, lb = np.log10(d["alpha"][te]), np.log10(d["beta"][te])
i = int(te[np.argmin((la + 2) ** 2 + lb**2 + (d["xb"][te] - d["xa"][te] - 0.3) ** 2)])
ub = d["u_bar"][i]
cols = {1: "C0", 2: "C1", 4: "C2", 8: "C3"}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for T in (1, 2, 4, 8):
    u = d[f"U_T{T}"][i]
    t = np.arange(1, len(u) + 1) / 320
    ax[0].plot(t, u / ub, color=cols[T], label=f"T = {T}")
    ax[1].semilogy(t, np.abs(u / ub - 1), color=cols[T])
    ax[2].semilogy(T - t, np.abs(u / ub - 1), color=cols[T])
ax[0].axhline(1, color="k", ls="--", lw=0.8, label=r"reference $\bar u$")
ax[0].set_xlabel("t")
ax[0].set_ylabel(r"$u_T(t)/\bar u$")
ax[0].set_title("(a) optimal controls for four horizons", fontsize=10)
ax[0].legend(fontsize=8)
for a, lab, ttl in ((ax[1], "time from the start  t", "(b) aligned at the start"), (ax[2], "time to the end  T - t", "(c) aligned at the end")):
    a.set_xlim(0, 1.5)
    a.set_ylim(1e-6, 2)
    a.set_xlabel(lab)
    a.set_ylabel(r"$|u_T/\bar u - 1|$")
    a.set_title(ttl, fontsize=10)
fig.tight_layout()
fig.savefig(out / "fig_turnpike.pdf")
print("fig_turnpike sample", i, "alpha", d["alpha"][i], "beta", d["beta"][i])

# ---------------- fig_transfer ----------------
R = json.loads(pathlib.Path("runs/fresh1d/fresh_summary.json").read_text())
Ts = [1, 2, 3, 4, 6, 8, 12]
rows = [("turnpike", "ours (trained on T = 1)", "C0", "o-"), ("krr", "kernel ridge, same structure", "C2", "s-"),
        ("cnn", r"CNN (T $\in$ {1, 2})", "C1", "^-"), ("cnn_res_T1", "residual CNN, same reference (T = 1)", "C4", "v-"),
        ("short_solve_T2", r"exact $T_s=2$ solve at the ends", "C8", "d--"), ("reference_only", r"reference $\bar u$ only", "0.5", "x:")]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for key, lab, c, st in rows:
    fr = R[key]["fresh"]
    TT = [T for T in Ts if str(T) in fr]
    ax[0].semilogy(TT, [100 * fr[str(T)]["total_rel"]["median"] for T in TT], st, color=c, label=lab, ms=4)
    if key in ("turnpike", "krr", "cnn"):
        for w, ls in (("init_rel", "-"), ("term_rel", "--")):
            ax[1].semilogy(TT, [100 * fr[str(T)][w]["median"] for T in TT], ls, marker=st[0], color=c, ms=4,
                           label=f"{lab.split(' (')[0].split(',')[0]}, {'initial' if w == 'init_rel' else 'terminal'} window")
ax[0].set_title("(a) whole-horizon error, fresh parameters", fontsize=10)
ax[1].set_title("(b) error inside the two end windows (width 0.5)", fontsize=10)
for a in ax[:2]:
    a.set_xscale("log")
    a.set_xticks(Ts)
    a.set_xticklabels([str(T) for T in Ts])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("horizon T (training used T = 1; T = 3, 6, 12 never seen)")
    a.set_ylabel("median relative error (%)")
ax[0].legend(fontsize=7)
ax[1].legend(fontsize=7, ncol=1)
bins = ["< 4", "4-6", "> 6"]
pred = [0.21, 0.086, 0.028]
med = [1.44, 1.12, 1.00]
p90 = [4.11, 1.70, 1.17]
x = np.arange(3)
ax[2].bar(x - 0.2, med, 0.4, label="median", color="C0")
ax[2].bar(x + 0.2, p90, 0.4, label="90th percentile", color="C9")
ax2 = ax[2].twinx()
ax2.plot(x, pred, "ko--", label=r"predicted ambiguity $e^{-\lambda_1 T_{tr}/2}$")
ax2.set_ylabel(r"$e^{-\lambda_1/2}$")
ax2.grid(False)
ax[2].axhline(1, color="k", lw=0.8)
ax[2].set_xticks(x)
ax[2].set_xticklabels([r"$\lambda_1$ " + b for b in bins])
ax[2].set_ylabel("initial-window transfer ratio  T=8 / T=1")
ax[2].set_title("(c) where transfer degrades: slow end layers", fontsize=10)
h1, l1 = ax[2].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(out / "fig_transfer.pdf")

# ---------------- fig_cost ----------------
# (online seconds per query, median relative error % at the long horizon); exact = reference (error 0, drawn at 1e-5)
S = {
    "1D-const, T = 8": {"exact": 14.1, "exact_struct": (0.19, "Riccati"), "short1": (1.22, 0.82), "ours": (0.0051, 0.11)},
    "2D-strip, T = 8": {"exact": 56.0, "short1": (7.25, 0.001), "ours": (0.23, 0.024)},
    "2D-semi, T = 8": {"exact": 500.0, "short1": (67.0, 0.0032), "ours": (2.4, 0.040)},
    "2D-semi-strip, T = 8": {"exact": 671.0, "short1": (71.6, 0.0003), "ours": (1.6, 0.134)},
    "2D-move, T = 8": {"exact": 60.0, "ours": (4.3, 0.100)},
    "2D-semi-move, T = 8": {"exact": 757.0, "ours": (4.4, 0.237), "rescnn": (4.4, 0.316)},
    "2D-adv, T = 4": {"exact": 760.0, "short1": (272.0, 0.0006), "ours": (6.0, 1.15), "rescnn": (6.0, 17.8)},
}
markers = ["o", "s", "D", "^", "v", "P", "X"]
fig, ax = plt.subplots(figsize=(8.6, 5.2))
for (name, r), mk in zip(S.items(), markers):
    ax.scatter([r["exact"]], [1e-5], marker=mk, color="k", s=40)
    if "exact_struct" in r:
        ax.scatter([r["exact_struct"][0]], [1e-5], marker=mk, color="0.5", s=40)
    if "short1" in r:
        ax.scatter([r["short1"][0]], [r["short1"][1]], marker=mk, color="C2", s=45)
    if "rescnn" in r:
        ax.scatter([r["rescnn"][0]], [r["rescnn"][1]], marker=mk, color="C4", s=45)
    ax.scatter([r["ours"][0]], [r["ours"][1]], marker=mk, color="C0", s=55, label=name)
    xs = [r["ours"][0], r.get("short1", (None,))[0], r["exact"]]
    ys = [r["ours"][1], r.get("short1", (None, None))[1], 1e-5]
    pts = [(a, b) for a, b in zip(xs, ys) if a is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="0.7", lw=0.8, zorder=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("online time per query, single thread (s)")
ax.set_ylabel("median relative error at the long horizon (%)")
ax.axhline(1e-5, color="k", lw=0.5, ls=":")
ax.text(25, 1.6e-5, "exact solve (error 0)", fontsize=8)
from matplotlib.lines import Line2D

leg2 = [Line2D([], [], color=c, marker="o", ls="", label=l) for c, l in
        (("C0", "ours"), ("C2", "short solve at the ends (exact)"), ("C4", "residual CNN, same reference"), ("k", "exact full solve"),
         ("0.5", "Riccati-based solver (1D-const)"))]
l1 = ax.legend(fontsize=7.5, loc="upper right", title="problem (marker)", title_fontsize=8)
ax.add_artist(l1)
ax.legend(handles=leg2, fontsize=7.5, loc="center left", title="method (colour)", title_fontsize=8)
fig.tight_layout()
fig.savefig(out / "fig_cost.pdf")
print("wrote figures to", out)

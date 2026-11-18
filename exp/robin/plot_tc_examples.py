#!/usr/bin/env python3
"""Figures of how the convolutional implementation (cnn_res_tc) predicts, from saved predictions (log sections 31.4, 32).

    e20    one test sample of the semilinear two-spot problem (E20): the reference P on the extended grid, the exact
           control and the cnn_res_tc (W = 127, L = 0.1, seed 0) prediction at one boundary node on T = 1 and T = 8,
           the learned correction u - P over all nodes, and the node-RMS error on T = 8 against the T = 1 error placed
           at the same distance to the nearest end.
    tc1d   the 1D tests (E01, E08): controls of a slow and a fast E01 sample on T = 8 for every model, and the per-sample
           T = 8 error against the slowest decay rate.

Run with numpy + matplotlib (.env/bin/python), from ~/ctrl1d_lowrank.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, GRAY, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#b5332e", "#8a5cc2"
SPOT = 4
WIN = 0.1 + 127 / 320  # L + W dt of the final E20 model
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8})


def reference(u0, uhat, om, ph, t):
    """Quasi-periodic reference of E20 (numpy port of multispot_surrogate.reference): (m,), (2, K, m), (2,), (2,), (N,) -> (N, m)."""
    out = np.repeat(u0[None, :], len(t), 0).astype(np.float64)
    k = np.arange(uhat.shape[1])
    for j in range(2):
        E = np.exp(1j * (om[j] * t[:, None] + ph[j]) * k[None, :])
        E[:, 0] = 0
        out += 2 * np.real(E @ uhat[j])
    return out


def node_rms(a):
    return np.sqrt((a ** 2).mean(-1))


def e20(args):
    d = np.load(args.data)
    pr = np.load(args.pred)
    te = pr["test_idx"]
    U1p, U8p = pr["U_T1"].astype(np.float64), pr["U_T8"].astype(np.float64)
    U1, U8 = d["U_nl_T1"][te].astype(np.float64), d["U_nl_T8_test"].astype(np.float64)
    err8 = np.linalg.norm((U8p - U8).reshape(len(te), -1), axis=1) / np.linalg.norm(U8.reshape(len(te), -1), axis=1)
    i = int(np.argsort(err8)[len(te) // 2])  # median-error test sample
    n = te[i]
    X, dt = d["X"], float(d["dt"])
    om, ph = X[n, [SPOT + 3, SPOT + 9]], X[n, [SPOT + 5, SPOT + 11]]
    K = int(d["nharm"].max()) + 1
    u0, uhat = d["u0"][n].astype(np.float64), d["uhat"][n, :, :K].astype(np.complex128)
    W = 127
    t1 = np.arange(1, 321) * dt
    t8 = np.arange(1, 2561) * dt
    te1 = np.arange(1 - W, 320 + W + 1) * dt
    te8 = np.arange(1 - W, 2560 + W + 1) * dt
    P1, P8 = reference(u0, uhat, om, ph, te1), reference(u0, uhat, om, ph, te8)
    P8in = P8[W:-W]
    j = int(np.argmax(np.abs(U8[i, 0] - P8in[0])))  # node with the largest initial layer
    s_j = (j + 1) / (d["U_nl_T1"].shape[2] + 1)
    print(f"E20 sample test #{i} (data row {n}), T=8 error {100 * err8[i]:.3f}% (median {100 * np.median(err8):.3f}%), node {j}, s={s_j:.3f}")

    fig = plt.figure(figsize=(10.5, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_b1, ax_b2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_c1, ax_c2 = fig.add_subplot(gs[2, 0], sharex=ax_b1), fig.add_subplot(gs[2, 1], sharex=ax_b2)

    def shade(ax, T, lo, hi):
        ax.axvspan(lo, 0, color="0.85", lw=0, zorder=0)
        ax.axvspan(T, hi, color="0.85", lw=0, zorder=0)
        ax.axvspan(0, min(WIN, T), color=BLUE, alpha=0.08, lw=0, zorder=0)
        ax.axvspan(max(T - WIN, 0), T, color=ORANGE, alpha=0.09, lw=0, zorder=0)
        if T > 2 * WIN:
            ax.axvspan(WIN, T - WIN, color=AQUA, alpha=0.08, lw=0, zorder=0)

    def trace(ax, t, u, T, lw=2.0):
        for msk, c in ((t <= WIN, BLUE), ((t > WIN) & (t < T - WIN), AQUA), (t >= T - WIN, ORANGE)):
            ax.plot(t[msk], u[msk], color=c, lw=lw, zorder=3)

    # (a) T = 1, node j
    shade(ax_a, 1.0, te1[0], te1[-1])
    ax_a.plot(te1, P1[:, j], "--", color=GRAY, lw=1.1, label="reference $P$ (analytic, extended grid)")
    ax_a.plot(t1, U1[i, :, j], color="k", lw=1.0, label="exact $u_1$")
    trace(ax_a, t1, U1p[i, :, j], 1.0)
    ax_a.set_xlim(te1[0], te1[-1])
    ax_a.set_title(f"(a) training horizon $T=1$, node $s={s_j:.2f}$")
    ax_a.set_xlabel("$t$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Line2D([], [], color=GRAY, ls="--", lw=1.1, label="reference $P$ (analytic, also on the extended grid)"),
          Line2D([], [], color="k", lw=1.0, label="exact control"),
          Line2D([], [], color=BLUE, lw=2, label="prediction: start region"), Line2D([], [], color=AQUA, lw=2, label="interior"),
          Line2D([], [], color=ORANGE, lw=2, label="end region"), Patch(color="0.85", label="extended grid ($t<0$, $t>T$)")]
    fig.legend(handles=hd, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.995), fontsize=7.5)
    # learned correction u_hat - P on T = 1 over all nodes
    corr = (U1p[i] - P1[W:-W]).T
    v = np.abs(corr).max()
    im = ax_h.imshow(corr, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=[0, 1, 0, 1])
    ax_h.axvline(WIN, color="k", lw=0.8, ls=":")
    ax_h.axvline(1 - WIN, color="k", lw=0.8, ls=":")
    ax_h.axhline(s_j, color="k", lw=0.6, ls="--")
    ax_h.set_title(r"(b) learned correction $\hat u_1-P$ on $T=1$, all 99 nodes")
    ax_h.set_xlabel("$t$")
    ax_h.set_ylabel("boundary node $s$")
    fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02)
    # (b) T = 8, broken axis
    for ax, (lo, hi) in ((ax_b1, (te8[0], 1.6)), (ax_b2, (6.4, te8[-1]))):
        shade(ax, 8.0, te8[0], te8[-1])
        ax.plot(te8, P8[:, j], "--", color=GRAY, lw=1.1)
        ax.plot(t8, U8[i, :, j], color="k", lw=1.0)
        trace(ax, t8, U8p[i, :, j], 8.0)
        ax.set_xlim(lo, hi)
    ax_b2.spines["left"].set_visible(False)
    ax_b2.tick_params(left=False, labelleft=False)
    ax_b2.set_ylim(ax_b1.get_ylim())
    ax_b1.set_title("(c) prediction on $T=8$ with the same weights (same node)", loc="left")
    # (c) node-RMS error on T = 8 and the T = 1 error at the same distance to the nearest end
    e8 = node_rms(U8p[i] - U8[i]) / node_rms(U8[i]).mean()
    e1 = node_rms(U1p[i] - U1[i]) / node_rms(U8[i]).mean()
    for ax, (lo, hi) in ((ax_c1, (te8[0], 1.6)), (ax_c2, (6.4, te8[-1]))):
        ax.axvspan(lo, 0, color="0.85", lw=0)
        ax.axvspan(8, hi, color="0.85", lw=0)
        ax.semilogy(t8, e8, color=BLUE, lw=1.6, label="error on $T=8$")
        ax.set_xlabel("$t$")
    h = 160
    ax_c1.semilogy(t1[:h], e1[:h], color=YELLOW, lw=1.2, ls="--", label="error on $T=1$, same $t$")
    ax_c2.semilogy(8 - (1 - t1[h:]), e1[h:], color=YELLOW, lw=1.2, ls="--", label="error on $T=1$, same $T-t$")
    ax_c2.set_ylim(ax_c1.get_ylim())
    ax_c2.spines["left"].set_visible(False)
    ax_c2.tick_params(left=False, labelleft=False)
    ax_c1.set_ylabel("node-RMS error / $\\|u\\|$")
    ax_c1.set_title("(d) node-RMS error on $T=8$ vs. the $T=1$ error at the same distance to the end", loc="left")
    ax_c1.legend(loc="upper right", fontsize=7.5)
    ax_c2.legend(loc="upper left", fontsize=7.5)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png")


def load_run(path):
    """One result JSON, or the single-seed files <stem>_s<k>.json merged."""
    path = pathlib.Path(path)
    files = [path] if path.exists() else sorted(path.parent.glob(f"{path.stem}_s*.json"))
    if not files:
        return None, None
    r = json.loads(files[0].read_text())
    runs = [run for g in files for run in json.loads(g.read_text())["runs"]]
    per = np.array([run["test"]["8"]["per_sample"] for run in runs])  # (seeds, n_test)
    return r, per


def tc1d(args):
    root = pathlib.Path(args.root)
    d1 = np.load(args.e01_data)
    te = d1["test_idx"]
    U8 = d1["U_T8"][te] / d1["u_bar"][te, None]
    ub = d1["u_bar"][te]
    models = [("tp_T1", "TurnpikeNet (exp. layers), $T_{tr}=1$", BLUE, "-"), ("hyb_T1", "hybrid, joint training, $T_{tr}=1$", PURPLE, ":"),
              ("hyb2_T1", "hybrid, two-stage, $T_{tr}=1$", PURPLE, "-"), ("tc_T1", "conv. (cnn_res_tc), $T_{tr}=1$", ORANGE, "-"),
              ("tc_T2", "conv., $T_{tr}=2$", YELLOW, "-"), ("tc_T4", "conv., $T_{tr}=4$", AQUA, "-")]
    r_or, _ = load_run(root / "e01" / "oracle_T1.json")
    lam = np.array(r_or["lam_open_test"])
    preds = {k: np.load(root / "e01" / "pred" / f"pred_{k}_seed0.npz")["U_T8"] / ub[:, None] for k, *_ in models if (root / "e01" / "pred" / f"pred_{k}_seed0.npz").exists()}
    t8 = np.arange(1, 2561) / 320
    slow, fast = int(np.argmin(lam)), int(np.argmax(lam))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, i, tag in ((axes[0, 0], slow, "slow"), (axes[0, 1], fast, "fast")):
        sl = t8 <= 3.0
        ax.plot(t8[sl], U8[i, sl], color="k", lw=2.4, alpha=0.35, label="exact $u_8/\\bar u$")
        for k, lab, c, ls in models:
            if k in preds:
                ax.plot(t8[sl], preds[k][i, sl], color=c, lw=1.3, ls=ls, label=lab)
        for w, c in ((0.497, ORANGE), (0.997, YELLOW), (1.997, AQUA)):
            ax.axvline(w, color=c, lw=0.8, ls=":")
        ax.set_title(f"({'a' if tag == 'slow' else 'b'}) E01, {tag}est test sample: $\\lambda_1(\\beta)={lam[i]:.2f}$, start of $T=8$")
        ax.set_xlabel("$t$")
    axes[0, 0].set_ylabel("$u/\\bar u$")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for ax, ex, key, xlab in ((axes[1, 0], "e01", "lam_open_test", r"$\lambda_1(\beta)$ (open-loop rate)"),
                              (axes[1, 1], "e08", "lam_floor_test", r"$\lambda_{\mathrm{lin}}$ (linearized rate)")):
        for k, lab, c, ls in models:
            r, per = load_run(root / ex / f"{k}.json")
            if r is None:
                continue
            x = np.array(r[key])
            ax.scatter(x, 100 * per.mean(0), s=10, color=c, alpha=0.8, label=lab, lw=0, marker="^" if ls == ":" else "o")
        ro, _ = load_run(root / ex / "oracle_T1.json")
        x = np.array(ro[key])
        o = np.array(ro["runs"][0]["test"]["8"]["per_sample"])
        ax.scatter(x, 100 * o, s=14, facecolors="none", edgecolors=GRAY, lw=0.6, label="local oracle, $T_{tr}=1$")
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_title(f"({'c' if ex == 'e01' else 'd'}) {ex.upper()}: per-sample error at $T=8$ (seed mean)")
    axes[1, 0].set_ylabel("relative $L^2$ error at $T=8$ (%)")
    axes[1, 1].legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png; slow sample {slow} lam {lam[slow]:.2f}, fast {fast} lam {lam[fast]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["e20", "tc1d"])
    ap.add_argument("--data", default="runs/multispot_data/data.npz")
    ap.add_argument("--pred", default="runs/multispot_data/models/pred/pred_nl_cnn_res_tc_W127_seed0.npz")
    ap.add_argument("--root", default="runs/tc1d")
    ap.add_argument("--e01-data", default="runs/turnpike/turnpike_data.npz")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    {"e20": e20, "tc1d": tc1d}[args.what](args)


if __name__ == "__main__":
    main()

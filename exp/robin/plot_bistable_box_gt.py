#!/usr/bin/env python3
"""Ground-truth figures for the moving hot spot with (a) box constraints on u and (b) a bistable (Nagumo) reaction.
Inputs are bistable_box_pilot.py outputs in --dir. Writes box_gt.png, nagumo_gt.png, nagumo_snapshots.png."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bistable_box_pilot import front_target  # noqa: E402
from moving2d import CfgMove, target  # noqa: E402

NH = 100


def target_of(d):
    m = d["meta"]
    return front_target(m["delta"], m.get("c0", 0.5)) if m.get("target") == "front" else target


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files if k not in ("prm", "meta", "info")}
    d["cf"] = CfgMove(**json.loads(str(z["prm"])))
    d["meta"], d["info"] = json.loads(str(z["meta"])), json.loads(str(z["info"]))
    return d


def axes_ts(cf):
    return np.arange(1, cf.Nt + 1) * cf.dt, np.arange(1, NH) / NH


def show_u(fig, ax, cf, Z, title, v):
    t, s = axes_ts(cf)
    pc = ax.pcolormesh(t, s, Z.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t")
    ax.set_ylabel("s (along the edge)")
    fig.colorbar(pc, ax=ax)


def show_state(fig, ax, cf, Z, title, vmax=1.2, c0=0.5):
    t, s = axes_ts(cf)
    pc = ax.pcolormesh(t, s, Z.T, shading="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.plot(t, c0 + cf.a * np.sin(cf.omega * t + cf.phi0), color="cyan", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t")
    ax.set_ylabel("s")
    fig.colorbar(pc, ax=ax)


def show_profiles(ax, cf, Z, Zr, title):
    t, s = axes_ts(cf)
    t0 = 0.5 * t[-1]
    for tt, col in zip(np.linspace(t0, t0 + 2 * np.pi / cf.omega, 4, endpoint=False), ["C0", "C1", "C2", "C3"]):
        k = int(np.argmin(np.abs(t - tt)))
        ax.plot(s, Z[k], color=col, lw=1.8, label=f"t = {t[k]:.2f}")
        ax.plot(s, Zr[k], color=col, lw=0.9, ls="--")
    ax.set_xlabel("s")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def show_sv(ax, items):
    for Z, st, nm in items:
        sv = np.linalg.svd(Z.astype(float), compute_uv=False)
        ax.semilogy(np.arange(1, 21), sv[:20] / sv[0], st, ms=4, label=f"{nm} (rank@1e-3 = {int(np.sum(sv > 1e-3 * sv[0]))})")
    ax.axhline(1e-3, color="gray", lw=0.7, ls=":")
    ax.set_ylim(1e-6, 1.5)
    ax.set_title("singular values of u(s, t)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def rel_track(cf, strip, tf=target):
    t, s = axes_ts(cf)
    Yd = tf(cf, s[None, :], cf.omega * t[:, None] + cf.phi0)
    k0 = cf.Nt // 4
    return np.linalg.norm(strip[k0:] - Yd[k0:]) / np.linalg.norm(Yd[k0:])


def box_figure(d, cases, out):
    """One row per case, each with its own unconstrained reference (same alpha)."""
    rows = [load(d / f"{c}.npz") for c, _ in cases]
    fig, axes = plt.subplots(len(rows), 6, figsize=(32, 4.4 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1.3, 1.1, 1.2, 1, 0.8]})
    for r, (row, (_, lab)) in enumerate(zip(rows, cases)):
        cf, C, Cu, m, inf = row["cf"], row["C"].astype(float), row["C_ref"].astype(float), row["meta"], row["info"]
        t, s = axes_ts(cf)
        v = np.abs(Cu).max()
        J0 = inf["J_unc"]
        ua, ub = m["ua"], m["ub"]
        show_u(fig, axes[r, 0], cf, Cu, f"alpha = {cf.alpha:g}: UNCONSTRAINED exact u(s, t)\n(max|u| = {v:.1f}; "
               f"strip tracking err {100 * rel_track(cf, row['strip_ref']):.0f}%)", v)
        bnd = f"[{ua:.2g}, {ub:.2g}]".replace("inf", "∞")
        show_u(fig, axes[r, 1], cf, C, f"{lab}: exact CONSTRAINED u, u in {bnd}\nJ/J_unc = {inf['J'] / J0:.3f}  "
               f"(clip of u_unc: {inf['J_clip_unc'] / J0:.3f})", v)
        tol = 1e-5 * v
        act = np.where(C >= ub - tol, 1.0, np.where(C <= ua + tol, -1.0, 0.0))
        ax = axes[r, 2]
        ax.pcolormesh(t, s, act.T, shading="auto", cmap="coolwarm", vmin=-1.3, vmax=1.3)
        ax.set_title(f"active set: red u = u_b ({100 * inf['frac_upper']:.0f}%), blue u = u_a ({100 * inf['frac_lower']:.0f}%)", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        show_state(fig, axes[r, 3], cf, row["strip"], f"strip-averaged state, constrained (tracking err {100 * rel_track(cf, row['strip']):.0f}%)")
        show_profiles(axes[r, 4], cf, C, Cu, f"one period: solid constrained, dashed unconstr.\n"
                      f"(|u - u_unc|/|u_unc| = {100 * np.linalg.norm(C - Cu) / np.linalg.norm(Cu):.0f}%)")
        show_sv(axes[r, 5], [(C, "o-", "constrained"), (Cu, "x--", "unconstrained")])
    fig.suptitle("Ground truth with control constraints: moving hot spot (cyan: centre), linear heat equation, edge control u(s, t) "
                 f"(n = 9900, m = 99, T = {rows[0]['cf'].T:g})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=105)
    print("wrote", out)


def best_of(d, tag):
    """Lower-J of the LQ-initialised and zero-initialised Newton runs, plus both J values."""
    runs = {i: load(d / f"nag_{tag}_{i}.npz") for i in ("lq", "zero") if (d / f"nag_{tag}_{i}.npz").exists()}
    best = min(runs, key=lambda i: runs[i]["info"]["J"])
    return runs[best], {i: r["info"]["J"] for i, r in runs.items()}


def nagumo_figure(d, tags, out):
    fig, axes = plt.subplots(len(tags), 7, figsize=(36, 4.4 * len(tags)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.1, 1.3, 1.3, 1.2, 1.2, 1, 0.8]})
    for r, tag in enumerate(tags):
        row, Js = best_of(d, tag)
        cf, C, Cl, m, inf = row["cf"], row["C"].astype(float), row["C_ref"].astype(float), row["meta"], row["info"]
        tf, c0 = target_of(row), m.get("c0", 0.5)
        t, s = axes_ts(cf)
        show_state(fig, axes[r, 0], cf, tf(cf, s[None, :], cf.omega * t[:, None] + cf.phi0),
                   f"target ({m.get('target', 'spot')}), alpha = {cf.alpha:g}", c0=c0)
        v = max(np.abs(C).max(), np.abs(Cl).max())
        lab = f"kappa = {m['kappa']:g}, theta = {m['theta']:g}"
        js = ", ".join(f"{k}-init J = {j:.5g}" for k, j in Js.items())
        show_u(fig, axes[r, 1], cf, C, f"{lab}: exact BISTABLE optimum u(s, t)\n({js})", v)
        show_u(fig, axes[r, 2], cf, Cl, f"exact LQ control (no reaction)\nJ on the bistable plant = {inf['J_lq_on_plant']:.5g} "
               f"({inf['J_lq_on_plant'] / inf['J']:.1f}x optimum)", v)
        show_state(fig, axes[r, 3], cf, row["strip"], f"strip state under the optimum (tracking err {100 * rel_track(cf, row['strip'], tf):.0f}%)",
                   c0=c0)
        show_state(fig, axes[r, 4], cf, row["strip_ref"], f"strip state, LQ control on the bistable plant ({100 * rel_track(cf, row['strip_ref'], tf):.0f}%)",
                   c0=c0)
        show_profiles(axes[r, 5], cf, C, Cl, f"one period: solid bistable, dashed LQ "
                      f"(|u - u_LQ|/|u_LQ| = {100 * np.linalg.norm(C - Cl) / np.linalg.norm(Cl):.0f}%)")
        show_sv(axes[r, 6], [(C, "o-", "bistable"), (Cl, "x--", "LQ")])
    fig.suptitle("Ground truth with a bistable reaction: y_t - Laplace y + kappa y (y - theta)(y - 1) = 0, moving target "
                 "(stable states 0 and 1; n = 9900, m = 99)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=105)
    print("wrote", out)


def field(y):
    F = np.zeros((NH + 1, NH + 1))
    F[1:NH, 0:NH] = y.reshape(NH - 1, NH)
    return F


def snapshot_figure(d, tag, out, nshow=8):
    row, _ = best_of(d, tag)
    cf, m = row["cf"], row["meta"]
    ks = row["snap_k"]
    pick = np.linspace(0, len(ks) - 1, nshow).astype(int)
    fig, axes = plt.subplots(2, nshow, figsize=(2.6 * nshow, 6.2), squeeze=False)
    for r, (key, nm) in enumerate((("snaps", "bistable optimum"), ("snaps_ref", "LQ control on bistable plant"))):
        for c, j in enumerate(pick):
            ax = axes[r, c]
            tk = (ks[j] + 1) * cf.dt
            ax.imshow(field(row[key][j].astype(float)), origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=0, vmax=1.2,
                      interpolation="bilinear")
            ax.axvline(cf.w, color="cyan", lw=0.8, ls="--")
            ax.plot([0.0], [m.get("c0", 0.5) + cf.a * np.sin(cf.omega * tk + cf.phi0)], marker=">", color="lime", ms=8)
            ax.set_xlim(0, 0.6)
            ax.set_title(f"{nm}\nt = {tk:.2f}" if c == 0 else f"t = {tk:.2f}", fontsize=8.5)
            ax.set_xticks([0, 0.2, 0.4])
            ax.set_yticks([0, 0.5, 1])
    fig.suptitle(f"State y(x, s, t) (x in [0, 0.6]; control acts on x = 0; cyan: observed strip; green marker: target centre / front), "
                 f"kappa = {m['kappa']:g}, theta = {m['theta']:g}, alpha = {cf.alpha:g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=105)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, required=True)
    ap.add_argument("--what", nargs="+", default=["box", "nagumo"])
    ap.add_argument("--tags", nargs="+", default=["k100_t50", "k100_t30", "k100_t70", "k300_t50"])
    ap.add_argument("--snap-tag", default="k100_t50")
    ap.add_argument("--prefix", default="nagumo")
    args = ap.parse_args()
    if "box" in args.what:
        box_figure(args.dir, [("box_pos", "heating only u >= 0"), ("box_pos40", "0 <= u <= 0.4 max|u_unc|"),
                              ("box_sym30", "|u| <= 0.3 max|u_unc|"), ("box_a1e-4_pos40", "0 <= u <= 0.4 max|u_unc|"),
                              ("box_a1e-5_pos40", "0 <= u <= 0.4 max|u_unc|")], args.dir / "box_gt.png")
    if "nagumo" in args.what:
        tags = [t for t in args.tags if (args.dir / f"nag_{t}_lq.npz").exists() or (args.dir / f"nag_{t}_zero.npz").exists()]
        nagumo_figure(args.dir, tags, args.dir / f"{args.prefix}_gt.png")
        snapshot_figure(args.dir, args.snap_tag, args.dir / f"{args.prefix}_snapshots.png")


if __name__ == "__main__":
    main()

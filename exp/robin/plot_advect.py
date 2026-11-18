#!/usr/bin/env python3
"""Figures for the advection-diffusion Peclet sweep (advect_pilot.py outputs in --dir):
advect_rank.png (ranks / POD floors / singular values / layers / tracking vs Pe) and advect_controls.png (u(s, t))."""

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

FLOWS = {"none": ("no flow (diffusion only)", "C0"), "along": ("flow along the edge, b = (0, 1)", "C1"),
         "into": ("flow into the domain, b = (1, 0)", "C2"), "lin": ("shear into the domain, bx = 0.25 + 1.5 s", "C3"),
         "pois": ("Poiseuille into the domain, bx = 6 s (1 - s)", "C4")}
FLOWS_DOWN = {"into_x06": ("uniform bx = 1, observe x in [0.6, 0.8]", "C2"), "lin_x06": ("shear bx = 0.25 + 1.5 s, observe x in [0.6, 0.8]", "C3"),
              "pois_x06": ("Poiseuille bx = 6 s (1 - s), observe x in [0.6, 0.8]", "C4")}
EPS = ["1", "0.1", "0.03", "0.01", "0.003", "0.001"]


def load(d, flow, e, nh):
    p = d / f"adv_{flow}_e{e}_n{nh}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    out = {k: z[k] for k in z.files if k != "res"}
    out["res"] = json.loads(str(z["res"]))
    return out


def rank_figure(d, out, flows, ref=None):
    """flows: solid (nh = 100) + hollow markers (nh = 200); ref: dashed comparison curves (nh = 100 only)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    tgt = None
    series = [(f, v, 100, "o", "-") for f, v in flows.items()] + [(f, v, 200, "s", "") for f, v in flows.items()]
    series += [(f, (v[0] + "  [observe x in [0, 0.2]]", v[1]), 100, "x", "--") for f, v in (ref or {}).items()]
    for flow, (lab, col), nh, mk, ls in series:
        if True:
            rows = [(float(e), load(d, flow, e, nh)) for e in EPS]
            rows = [(e, r["res"]) for e, r in rows if r is not None]
            if not rows:
                continue
            pe = np.array([1.0 / e for e, _ in rows])  # |b| ~ 1 for the flows; Pe = 1/eps also labels 'none'
            kw = dict(color=col, marker=mk, ls=ls if ls else "none", mfc="none" if nh == 200 else col, ms=7,
                      label=f"{lab}, nh = {nh}" if nh == 100 else f"   nh = 200, dt/2 (refinement)")
            axes[0, 0].semilogx(pe, [r["u"]["rank3"] for _, r in rows], **kw)
            axes[0, 1].semilogx(pe, [r["u_per"]["rank3"] for _, r in rows], **kw)
            axes[0, 2].loglog(pe, [r["pod_u"]["20"] for _, r in rows], **kw)
            axes[1, 2].semilogx(pe, [100 * r["track"] for _, r in rows], **kw)
            tgt = rows[0][1]["target"]["rank3"]
    for ax, ttl in ((axes[0, 0], "rank@1e-3 of the optimal control u(s, t) (= rank of q)"),
                    (axes[0, 1], "rank@1e-3 of the periodic turnpike (one period)"),
                    (axes[0, 2], "POD floor of u at rank 20 (relative L2)"),
                    (axes[1, 2], "strip tracking error (%) under the optimal control")):
        ax.set_xlabel("Peclet number 1/eps  (|b| ~ 1, L = 1)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[0, :2]:
        if tgt is not None:
            ax.axhline(tgt, color="gray", ls=":", lw=1.2, label=f"rank of the target y_d(s, t) itself ({tgt})")
    axes[0, 0].legend(fontsize=7.5)
    ax = axes[1, 0]
    for flow, (lab, col) in flows.items():
        for e, ls in (("1", ":"), ("0.03", "--"), ("0.003", "-")):
            r = load(d, flow, e, 100)
            if r is not None:
                sv = np.array(r["res"]["u"]["sv"])
                ax.semilogy(np.arange(1, len(sv) + 1), sv, color=col, ls=ls, lw=1.5, label=f"{flow}, eps = {e}")
    ax.axhline(1e-3, color="gray", lw=0.7, ls=":")
    ax.set_ylim(1e-6, 1.5)
    ax.set_title("singular values of u(s, t)", fontsize=10)
    ax.set_xlabel("index")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    for flow, (lab, col) in flows.items():
        for e, ls in (("1", ":"), ("0.03", "--"), ("0.003", "-")):
            r = load(d, flow, e, 100)
            if r is not None:
                cf_dt = 2.0 / len(r["layer"])
                ax.semilogy(np.arange(1, len(r["layer"]) + 1) * cf_dt, np.maximum(r["layer"], 1e-12), color=col, ls=ls, lw=1.4,
                            label=f"{flow}, eps = {e}")
    ax.set_ylim(1e-9, 3)
    ax.set_xlabel("t")
    ax.set_title("|u(t) - u_per(t)| / max|u_per|: initial / terminal layers around the periodic turnpike", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.suptitle("Advection-diffusion y_t - eps Laplace y + b . grad y = 0, moving hot spot, edge control (m = 99, alpha = 1e-3, T = 2): "
                 "where does the low rank break?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=105)
    print("wrote", out)


def field(y, nh):
    F = np.zeros((nh + 1, nh + 1))
    F[1:nh, 0:nh] = y.reshape(nh - 1, nh)
    return F


def controls_figure(d, out, flows, eps_show=("1", "0.03", "0.003")):
    fig, axes = plt.subplots(len(flows), len(eps_show) + 2, figsize=(5.2 * (len(eps_show) + 2), 4.3 * len(flows)), squeeze=False)
    for r_, (flow, (lab, _)) in enumerate(flows.items()):
        for c_, e in enumerate(eps_show):
            z = load(d, flow, e, 100)
            ax = axes[r_, c_]
            if z is None:
                ax.axis("off")
                continue
            C = z["C"].astype(float)
            t = np.arange(1, C.shape[0] + 1) * (2.0 / C.shape[0])
            s = np.arange(1, C.shape[1] + 1) / (C.shape[1] + 1)
            v = np.quantile(np.abs(C), 0.995)  # the colour scale ignores the short initial spike
            pc = ax.pcolormesh(t, s, C.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.plot(t, 0.5 + 0.3 * np.sin(2 * np.pi * t + 1.0), color="k", lw=0.6, ls=":")
            res = z["res"]
            ax.set_title(f"{lab}\neps = {e}: u(s, t), rank@1e-3 = {res['u']['rank3']}, max|u| = {np.abs(C).max():.0f}", fontsize=9)
            ax.set_xlabel("t")
            ax.set_ylabel("s")
            fig.colorbar(pc, ax=ax)
        z = load(d, flow, eps_show[-1], 100)
        if z is None:
            continue
        ax = axes[r_, len(eps_show)]
        St = z["strip"].astype(float)
        t = np.arange(1, St.shape[0] + 1) * (2.0 / St.shape[0])
        sy = np.arange(1, St.shape[1] + 1) / (St.shape[1] + 1)
        pc = ax.pcolormesh(t, sy, St.T, shading="auto", cmap="magma", vmin=0, vmax=max(1.0, St.max()))
        ax.plot(t, 0.5 + 0.3 * np.sin(2 * np.pi * t + 1.0), color="cyan", lw=0.8, ls="--")
        ax.set_title(f"strip-averaged state, eps = {eps_show[-1]} (tracking err {100 * z['res']['track']:.0f}%)", fontsize=9)
        ax.set_xlabel("t")
        fig.colorbar(pc, ax=ax)
        ax = axes[r_, len(eps_show) + 1]
        j = len(z["snap_k"]) // 2
        tk = (z["snap_k"][j] + 1) * (2.0 / St.shape[0])
        F = field(z["snaps"][j].astype(float), 100)
        ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=0, vmax=max(1.0, F.max()), interpolation="bilinear")
        xo = z["res"].get("xo", 0.0)
        for xv in ((xo, xo + 0.2) if xo > 0 else (0.2,)):
            ax.axvline(xv, color="cyan", lw=0.8, ls="--")
        ax.plot([0.0], [0.5 + 0.3 * np.sin(2 * np.pi * tk + 1.0)], marker=">", color="lime", ms=9)
        ax.set_title(f"state y(x, s) at t = {tk:.2f}, eps = {eps_show[-1]}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("s")
    fig.suptitle("Exact optimal edge controls u(s, t) for advection-diffusion (dotted: target centre); state panels: cyan = observed strip",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=100)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rank_figure(args.dir, args.dir / "advect_rank.png", FLOWS)
    controls_figure(args.dir, args.dir / "advect_controls.png", FLOWS)
    rank_figure(args.dir, args.dir / "advect_rank_down.png", FLOWS_DOWN, ref={k: FLOWS[k] for k in ("into", "lin", "pois")})
    controls_figure(args.dir, args.dir / "advect_controls_down.png", FLOWS_DOWN)
    rows = []
    for p in sorted(args.dir.glob("adv_*.npz")):
        r = json.loads(str(np.load(p)["res"]))
        rows.append({"case": p.stem, **{k: v for k, v in r.items() if not isinstance(v, dict)},
                     "rank_u": [r["u"]["rank3"], r["u"]["rank4"]], "rank_u_per": [r["u_per"]["rank3"], r["u_per"]["rank4"]],
                     "rank_target": r["target"]["rank3"], "pod_u": r["pod_u"], "pod_u_per": r["pod_u_per"]})
    (args.dir / "summary.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['case']:24s} Pe={r['Pe']:7.0f} rank u {r['rank_u']} per {r['rank_u_per']} tgt {r['rank_target']} "
              f"pod20 {r['pod_u']['20']:.1e} track {100 * r['track']:.0f}% J/J0 {r['J_over_J0']:.3f} pcg {r['pcg_iters']} "
              f"mid {r['per_vs_u_mid']:.1e} t {r['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()

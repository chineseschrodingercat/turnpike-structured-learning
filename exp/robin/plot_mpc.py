#!/usr/bin/env python3
"""Closed-loop MPC trajectories from mpc_semilinear.py outputs: applied control u(t) and mean state in the
observation window, one scenario per row (the median-gamma_eff scenario of each JSON)."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"exact_4": dict(color="k", lw=2.2, label="exact MPC, T_p = 4"),
         "tp_1": dict(color="C3", lw=1.3, ls="--", label="TurnpikeNet MPC, T_p = 1"),
         "lq_1": dict(color="C2", lw=1.1, ls=":", label="LQ-model MPC (ignores gamma y^3)"),
         "openloop": dict(color="C0", lw=1.1, ls="-.", label="open-loop optimum (no feedback)"),
         "const": dict(color="gray", lw=0.9, label="u = u_bar")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.json[:median|openloop_worst]")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    specs = [s.split("=", 1) for s in args.runs]
    fig, axes = plt.subplots(len(specs), 2, figsize=(16, 4.3 * len(specs)), squeeze=False)
    for row, (lab, path) in enumerate(specs):
        path, _, pick = path.partition(":")
        R = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if pick == "openloop_worst":  # scenario where feedback matters most
            ref = {r["scenario"]: r["J"] for r in R["runs"] if r["ctrl"] == "exact_4"}
            ol = {r["scenario"]: (r["J"] - ref[r["scenario"]]) / ref[r["scenario"]] for r in R["runs"] if r["ctrl"] == "openloop"}
            sc = next(s for s in R["scenarios"] if s["idx"] == max(ol, key=ol.get))
        else:
            sc = sorted(R["scenarios"], key=lambda s: s["mu"][4])[len(R["scenarios"]) // 2]
        runs = {r["ctrl"]: r for r in R["runs"] if r["scenario"] == sc["idx"]}
        Jref = runs["exact_4"]["J"]
        dist = R.get("disturbance", {"kicks": [2, 4, 6], "sink": 0.0})
        for c, st in STYLE.items():
            if c not in runs:
                continue
            u, ym = np.array(runs[c]["u"]), np.array(runs[c]["ymean"])
            t = np.arange(1, len(u) + 1) / 320
            extra = f" (J {100 * (runs[c]['J'] - Jref) / Jref:+.3f}%)" if c != "exact_4" else ""
            axes[row, 0].plot(t, u, **{**st, "label": st["label"] + extra})
            axes[row, 1].plot(t, ym, **st)
        for ax in axes[row]:
            for tk in dist["kicks"]:
                ax.axvline(tk, color="orange", lw=0.6, alpha=0.6)
            ax.grid(alpha=0.3)
            ax.set_xlabel("t")
        axes[row, 1].axhline(1.0, color="k", ls=":", lw=0.8)
        g = sc["geo"]
        ttl = (f"{lab}: gamma_eff = {sc['mu'][4]:.1f}, alpha = {g['alpha']:.2g}, beta = {g['beta']:.2g}, window [{g['xa']:.2f},{g['xb']:.2f}]"
               + (f", unmodeled sink {dist['sink']} from t = {dist.get('sink_start', 1)}" if dist.get("sink", 0) > 0 else ""))
        axes[row, 0].set_title("applied control u(t) — " + ttl, fontsize=9)
        axes[row, 1].set_title("mean state in the observation window (target 1); orange = state kicks", fontsize=9)
        axes[row, 0].legend(fontsize=7.5)
        axes[row, 0].set_ylabel("u(t)")
    fig.suptitle("Closed-loop receding-horizon control of y_t - y_xx + gamma y^3 = 0 (delta = 0.1, y(0) = 0)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=125)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

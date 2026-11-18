#!/usr/bin/env python3
"""Summary dot plots of all multi-spot surrogates (rounds 1-7): T = 8 test error (mean over seeds of the per-seed median,
whiskers = min/max over seeds) and the terminal transfer ratio R_E, one figure per kind.

    python plot_multispot_summary.py runs/multispot_data/models OUTDIR
"""

import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
# fixed categorical order (validated default palette, slots 1-4); the reference-only row uses secondary ink
FAMILY = {"final": ("#2a78d6", "horizon-independent residual CNN (cnn_res_tc)"),
          "layers": ("#eb6834", "turnpike surrogate with explicit end layers"),
          "nolayer": ("#1baf7a", "single temporal 2D CNN, no explicit layers"),
          "generic": ("#eda100", "generic CNN / residual CNN (zero padding)"),
          "ref": ("#52514e", "reference P only")}
ROWS = {"nl": [("ref_only", "P only", "ref"), ("tp_nocorr", "tp: layers, no $\\Delta$", "layers"),
               ("tp_local", "tp_local: pointwise $\\Delta$", "layers"), ("tp_local_bnd", "tp_local_bnd: + snapshot layers", "layers"),
               ("tp_qp", "tp_qp: + phases in $\\Delta$", "layers"), ("tp_qp_T12", "tp_qp", "layers"),
               ("tp_tc", "tp_tc: temporal $\\Delta$, W=63", "layers"), ("tp_tc_T12", "tp_tc", "layers"),
               ("tp_tce", "tp_tce: + end-distance channels", "layers"),
               ("tp_tce_nolayer", "tp_tce_nolayer, 800 ep", "nolayer"), ("tp_tce_nolayer_e1600", "tp_tce_nolayer, 1600 ep", "nolayer"),
               ("tp_tce_nolayer_W127_e1600", "tp_tce_nolayer, W=127, 1600 ep", "nolayer"),
               ("cnn_res_tc", "cnn_res_tc, W=63", "final"), ("cnn_res_tc_T12", "cnn_res_tc, W=63", "final"),
               ("cnn_res_tc_W127", "cnn_res_tc, W=127", "final"),
               ("cnn", "cnn, 300 ep", "generic"), ("cnn_e1200", "cnn, 1200 ep", "generic"),
               ("cnn_res_T1", "cnn_res, 300 ep", "generic"), ("cnn_res_T1_e1200", "cnn_res, 1200 ep", "generic"),
               ("cnn_res", "cnn_res, 300 ep", "generic"), ("cnn_res_e1200", "cnn_res, 1200 ep", "generic")],
        "lin": [("ref_only", "P only", "ref"), ("tp", "tp: MLP layer amplitudes", "layers"),
                ("tp_bnd", "tp_bnd: snapshot layer amplitudes", "layers"),
                ("cnn_res_tc", "cnn_res_tc, W=63", "final"), ("cnn_res_tc_W127", "cnn_res_tc, W=127", "final"),
                ("cnn", "cnn, 300 ep", "generic"), ("cnn_e1200", "cnn, 1200 ep", "generic"),
                ("cnn_res_T1", "cnn_res, 300 ep", "generic"), ("cnn_res_T1_e1200", "cnn_res, 1200 ep", "generic"),
                ("cnn_res", "cnn_res, 300 ep", "generic"), ("cnn_res_e1200", "cnn_res, 1200 ep", "generic")]}


def load():
    groups = {}
    for f in sorted(MODELS.glob("*.json")):
        r = json.loads(f.read_text())
        g = groups.setdefault(re.sub(r"_s\d+$", "", f.stem), {"train_T": r["train_T"], "runs": []})
        g["runs"] += r["runs"]
    return groups


def main():
    G = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False})
    for kind, rows in ROWS.items():
        rows = [(k, lab, fam) for k, lab, fam in rows if f"{kind}_{k}" in G]
        n = len(rows)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 0.33 * n + 1.4), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
        for i, (k, lab, fam) in enumerate(rows):
            g = G[f"{kind}_{k}"]
            e8 = np.array([100 * r["test"]["8"]["median"] for r in g["runs"]])
            re_ = np.array([r["test"]["transfer_term"]["median"] for r in g["runs"]])
            y = n - 1 - i
            col = FAMILY[fam][0]
            two = len(g["train_T"]) > 1
            kw = dict(color=col, ms=7, mew=1.6, mfc="white" if two else col, zorder=3)
            a1.plot([e8.min(), e8.max()], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            a1.plot(e8.mean(), y, "o", **kw)
            a1.text(e8.max() * 1.12, y, f"{e8.mean():.3g}", va="center", fontsize=8, color="#52514e")
            a2.plot(re_.mean(), y, "o", **kw)
            lab_full = f"{lab}  [T={','.join(map(str, g['train_T']))}; {len(e8)} seed{'s' if len(e8) > 1 else ''}]"
            rows[i] = (k, lab_full, fam)
        a1.set_yticks(range(n))
        a1.set_yticklabels([lab for _, lab, _ in rows][::-1], fontsize=8)
        a1.set_xscale("log")
        ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10) if a1.get_xlim()[0] <= t <= a1.get_xlim()[1]]
        a1.set_xticks(ticks)
        a1.set_xticklabels([f"{t:g}" for t in ticks])
        a1.minorticks_off()
        a1.set_xlabel("relative $L^2$ error at $T=8$ (%), median over test set")
        a1.grid(axis="x", color="#e4e3df", lw=0.8, zorder=0)
        a2.axvline(1.0, color="#52514e", lw=1, ls=":", zorder=1)
        a2.set_xticks(np.arange(1, int(np.ceil(a2.get_xlim()[1])) + 1))
        a2.set_xlabel("terminal-window ratio $R_E$\nerr($T$=8) / err($T$=1)")
        a2.grid(axis="x", color="#e4e3df", lw=0.8, zorder=0)
        fam_h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=l) for f, (c, l) in FAMILY.items()
                 if any(fam == f for _, _, fam in rows)]
        dat_h = [plt.Line2D([], [], marker="s", ls="", color="#a8a7a1", ms=7, label="filled: trained on $T=1$ only"),
                 plt.Line2D([], [], marker="s", ls="", color="#a8a7a1", mfc="white", mew=1.6, ms=7, label="hollow: trained on $T\\in\\{1,2\\}$")]
        fig.legend(handles=fam_h, title="model family", loc="lower left", fontsize=8, title_fontsize=8, frameon=False,
                   bbox_to_anchor=(0.02, 0.0))
        fig.legend(handles=dat_h, title="training data", loc="lower right", fontsize=8, title_fontsize=8, frameon=False,
                   bbox_to_anchor=(0.98, 0.0))
        title = "semilinear" if kind == "nl" else "linear"
        fig.suptitle(f"Two-spot quasi-periodic target, {title} case (dot = mean over seeds, bar = range over seeds)", fontsize=10)
        fig.tight_layout(rect=(0, 0.13 if kind == "nl" else 0.2, 1, 0.97))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"ms_summary_{kind}.{ext}", dpi=200)
        plt.close(fig)
        print("wrote", OUT / f"ms_summary_{kind}.pdf")


if __name__ == "__main__":
    main()

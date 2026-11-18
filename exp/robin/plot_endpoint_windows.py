#!/usr/bin/env python3
"""Figure for endpoint_windows.json: per experiment, median relative error vs horizon T for the whole horizon (solid),
the initial window (dashed), the terminal window (dotted) and the interior (dash-dot), for ours, the best generic
baseline, the reference-only row and the deterministic short-solve route."""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401

PANELS = [("1d_const", "1D LQ, constant target", ["ours", "cnn", "reference_only", "short_solve_T1", "short_solve_T2"]),
          ("1d_semilinear", "1D semilinear (N = 3400)", ["ours", "cnn", "reference_only", "short_solve_T1", "short_solve_T2"]),
          ("2d_semilinear", "2D semilinear, scalar control", ["ours", "cnn", "fno", "reference_only", "short_solve_T1"]),
          ("2d_semilinear_strip", "2D semilinear, strip target, m = 99", ["ours", "ours_tau_linear", "cnn", "reference_only", "short_solve_T1"]),
          ("semilinear_moving", "2D semilinear, moving hot spot", ["ours", "cnn", "reference_only"]),
          ("advection", "2D advection, downstream (train T = 2)", ["ours_w48", "ours_mirror", "cnn_per", "cnn", "fno", "reference_only", "short_solve_T2"])]
STYLE = {"total_rel": "-", "init_rel": "--", "term_rel": ":", "interior_rel": "-."}
COLORS = {"ours": "C0", "ours_w48": "C0", "ours_mirror": "C9", "ours_tau_linear": "C4", "cnn": "C1", "cnn_per": "C5", "fno": "C3",
          "deeponet": "C6", "reference_only": "0.5", "short_solve_T1": "C2", "short_solve_T2": "C8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    R = json.loads(args.json.read_text())
    fig, axes = plt.subplots(2, 3, figsize=(21, 11.5))
    for ax, (key, title, rows) in zip(axes.flat, PANELS):
        ex = R[key]
        for lab in rows:
            row = ex["rows"].get(lab)
            if row is None:
                continue
            for metric, ls in STYLE.items():
                Ts = [int(T) for T in ex["horizons"] if str(T) in row and row[str(T)].get(metric)]
                if not Ts:
                    continue
                ys = [100 * row[str(T)][metric]["median"] for T in Ts]
                ax.loglog(Ts, ys, ls=ls, marker="o", ms=4, color=COLORS.get(lab, "k"), lw=1.6 if metric == "total_rel" else 1.1,
                          label=lab if metric == "total_rel" else None)
        ax.set_title(f"{title}  (windows of {ex['tau']:g} at each end)", fontsize=10)
        ax.set_xlabel("horizon T")
        ax.set_ylabel("median relative error (%)")
        ax.set_xticks([int(T) for T in ex["horizons"]])
        ax.set_xticklabels([str(T) for T in ex["horizons"]])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("Where the error lives: whole horizon (solid), initial window (dashed), terminal window (dotted), interior (dash-dot). "
                 "Flat window curves = the end corrections transfer; a falling solid curve alone is the 1/sqrt(T) normalization.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=100)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

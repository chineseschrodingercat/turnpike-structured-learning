"""Plot the paper's V1.2 horizon-transfer and paired fresh-query results."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import numpy as np

HORIZONS = [1, 2, 4, 8]
FAMILIES = {
    "2D-semi": ("#0072B2", "o", "-"),
    "2D-semi-strip": ("#00845D", "s", "--"),
    "2D-semi-move": ("#C5531D", "D", "-."),
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.0, "axes.linewidth": 1.0,
    "axes.spines.top": True, "axes.spines.right": True, "axes.labelsize": 10,
    "axes.labelweight": "bold", "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "xtick.top": False, "ytick.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 3, "ytick.major.pad": 3,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

def endpoint_data(results):
    source = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    output = {}
    for family in FAMILIES:
        data = source[family]["arms"]["structured"]
        row = {"n": data["n_test_parameters"], "seeds": [0, 1, 2],
               "ratio_aggregation": "median within parameter over seeds, then quantiles across parameters",
               "whole_control_error_pct": {h: data["whole"][str(h)]["pooled_median_pct"] for h in HORIZONS}}
        for end, stats in data["windows"].items():
            row[end] = {"ratio_eligible_n": stats["ratio_eligible_n"],
                        "quantiles": {h: stats["quantiles"][str(h)] for h in HORIZONS}}
        output[family] = row
    return output


def cost_data(results):
    with (results / "paired_query_cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for family in FAMILIES:
        grouped = {}
        for row in rows:
            if row["family"] == family:
                grouped.setdefault(row["method"], []).append(row)
        output[family] = {}
        for name, cases in grouped.items():
            assert len(cases) == 3 and len({r["test_id"] for r in cases}) == 3
            output[family][name] = {
                "median_query_s": float(np.median([float(r["query_seconds"]) for r in cases])),
                "median_error_pct": float(100 * np.median([float(r["relative_control_error"]) for r in cases])),
                "sample_indices": [int(r["test_id"]) for r in cases]}
    return output


def export(fig, name, output):
    fig.savefig(output / f"{name}.svg")
    fig.savefig(output / f"{name}.png", dpi=600)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def check_shared_panels(fig, axes):
    """Check the physical plot areas of the two endpoint panels before export."""
    fig.canvas.draw()
    bounds = np.asarray([ax.get_window_extent().bounds for ax in axes]) * 72 / fig.dpi
    np.testing.assert_allclose(bounds[0, 1:], bounds[1, 1:], atol=1.5, rtol=0)
    if bounds[1, 0] <= bounds[0, 0] + bounds[0, 2]:
        raise ValueError("The two endpoint panels overlap")
    # The manuscript production environment also runs its full layout audit.
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ModuleNotFoundError:
        return
    require_matplotlib_panel_alignment(fig)

def draw_endpoints(data, output):
    marker_sizes = {"2D-semi": 6.2, "2D-semi-strip": 4.2, "2D-semi-move": 3.1}
    fig, axes = plt.subplots(1, 2, figsize=(6.12, 2.37), sharey=True)
    fig.subplots_adjust(left=0.128, right=0.982, bottom=0.30, top=0.858, wspace=0.12)
    for ax, label in zip(axes, ("initial", "terminal")):
        ax.axhline(1, color="#535353", lw=0.85, ls=(0, (2, 3)), zorder=0)
        handles = []
        for family, (color, marker, ls) in FAMILIES.items():
            low, center, high = np.array([data[family][label]["quantiles"][h] for h in HORIZONS]).T
            ax.fill_between(HORIZONS, low, high, color=color, alpha=0.14, lw=0)
            handle, = ax.plot(HORIZONS, center, color=color, lw=1.4, ls=ls, marker=marker,
                              markersize=marker_sizes[family], markerfacecolor="white", markeredgewidth=0.95, label=family)
            handles.append(handle)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(HORIZONS))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.87, 8.9)
        ax.set_xlabel("Evaluation horizon")
        ax.set_title(label.title() + " window", pad=5)
    axes[0].set_ylabel("Absolute error ratio to T = 1")
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.52, 0.035),
               borderaxespad=0, columnspacing=1.6, fontsize=8.5)
    check_shared_panels(fig, axes)
    export(fig, "fig_endpoint_transfer", output)

def draw_cost(data, output):
    fig, ax = plt.subplots(figsize=(6.12, 2.52))
    fig.subplots_adjust(left=0.145, right=0.735, bottom=0.235, top=0.955)
    markers = {"Structured": "o", "Stitching": "s", "Nonlinear solve": "^"}
    for family, methods in data.items():
        color = FAMILIES[family][0]
        for name, row in methods.items():
            if row["median_query_s"] <= 0 or row["median_error_pct"] <= 0:
                raise ValueError("Logarithmic cost and error coordinates must both be positive")
            ax.scatter(row["median_query_s"], row["median_error_pct"], marker=markers[name], s=48,
                       color=color, edgecolors="black", linewidths=0.45, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.65, 580)
    ax.set_ylim(2.4e-8, 1.15)
    ax.set_xlabel("Fresh query time (s)")
    ax.set_ylabel("Median control error (%)")
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    family_handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=c, markeredgecolor="black",
                             markeredgewidth=0.45, markersize=6, label=f) for f, (c, *_) in FAMILIES.items()]
    family_legend = ax.legend(handles=family_handles, loc="upper left", bbox_to_anchor=(1.02, 1.025),
                             title="Problem", borderaxespad=0.25, fontsize=8.5, title_fontsize=8.5)
    ax.add_artist(family_legend)
    method_handles = [Line2D([], [], marker=m, linestyle="none", markerfacecolor="white", markeredgecolor="black",
                             markeredgewidth=0.75, markersize=6, label=n) for n, m in markers.items()]
    ax.legend(handles=method_handles, loc="lower left", bbox_to_anchor=(1.02, 0.015),
              title="Method", borderaxespad=0.25, fontsize=8.5, title_fontsize=8.5)
    export(fig, "fig_paired_cost_accuracy", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Directory containing summary.json and paired_query_cases.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    endpoints, costs = endpoint_data(args.results), cost_data(args.results)
    (args.out / "figure_source_data.json").write_text(
        json.dumps({"endpoint": endpoints, "cost": costs}, indent=2) + "\n", encoding="utf-8")
    draw_endpoints(endpoints, args.out)
    draw_cost(costs, args.out)

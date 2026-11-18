#!/usr/bin/env python3
"""Summary of the 1D tests of the convolutional implementation (log section 32): per model, the per-seed errors at every
horizon, end-window errors and transfer ratios, the T = 8 error by slowest-rate bin, and the worst sample.
Reads runs/tc1d/{e01,e08}/*.json (single-seed files <name>_s<k>.json are merged); writes runs/tc1d/summary.json."""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path("runs/tc1d")
MODELS = {"e01": ["tp_T1", "hyb_T1", "hyb2_T1", "tc_T1", "tc_T2", "tc_T4", "oracle_T1", "oracle_T2", "oracle_T4"],
          "e08": ["tp_T1", "hyb_T1", "hyb2_T1", "tc_T1", "tc_T2", "oracle_T1", "oracle_T2"]}
BINS = {"e01": ((0, 4), (4, 6), (6, 100)), "e08": ((0, 4), (4, 8), (8, 100))}
LAM = {"e01": "lam_open_test", "e08": "lam_floor_test"}


def load(ex, name):
    f = ROOT / ex / f"{name}.json"
    files = [f] if f.exists() else sorted((ROOT / ex).glob(f"{name}_s*.json"))
    meta, runs = None, []
    for g in files:
        r = json.loads(g.read_text())
        meta = meta or r
        runs += r["runs"]
    return meta, sorted(runs, key=lambda r: -1 if r["seed"] is None else r["seed"])


def main():
    out = {}
    for ex, names in MODELS.items():
        out[ex] = {}
        print(f"\n===== {ex.upper()} =====")
        print(f"{'model':10s} {'seeds':6s} {'T=1 med/max %':>16s} {'T=8 med/max %':>16s} {'init tr.':>8s} "
              + " ".join(f"{'lam ' + str(lo) + '-' + str(hi):>12s}" for lo, hi in BINS[ex]) + "  worst (err%, lam)")
        for name in names:
            meta, runs = load(ex, name)
            if meta is None or not runs:
                print(f"{name:10s} missing")
                continue
            lam = np.array(meta[LAM[ex]])
            row = {"seeds": [r["seed"] for r in runs], "params": runs[0]["params"], "window": runs[0].get("window"),
                   "train_seconds": [r["train_seconds"] for r in runs], "train_T": meta["train_T"], "epochs": meta["epochs"],
                   "tdil": meta["tdil"], "edge_L": meta["edge_L"]}
            for T in ("1", "2", "4", "8"):
                if T not in runs[0]["test"]:
                    continue
                for k in ("median", "p90", "max", "init_median", "term_median", "interior_median"):
                    if k in runs[0]["test"][T]:
                        row[f"T{T}_{k}"] = [float(r["test"][T][k]) for r in runs]
            for k in ("transfer_init", "transfer_term"):
                if k in runs[0]["test"]:
                    row[k] = [float(r["test"][k]["median"]) for r in runs]
            if "stage1_tp_test" in runs[0]:
                row["stage1_tp"] = {T: [[float(r["stage1_tp_test"][T]["median"]), float(r["stage1_tp_test"][T]["max"])] for r in runs]
                                    for T in ("1", "8")}
            if "slowest_rate_test" in runs[0]:
                sr = np.array([r["slowest_rate_test"] for r in runs]).mean(0)
                row["slowest_rate_over_floor_median"] = float(np.median(sr / lam))
            per8 = np.array([r["test"]["8"]["per_sample"] for r in runs]).mean(0)
            row["T8_median_by_lam"] = {f"{lo}-{hi}": [int(((lam >= lo) & (lam < hi)).sum()), float(np.median(per8[(lam >= lo) & (lam < hi)]))]
                                       for lo, hi in BINS[ex]}
            i = int(np.argmax(per8))
            row["T8_worst"] = {"test_pos": i, "err": float(per8[i]), "lam": float(lam[i])}
            row["spearman_T8_err_vs_lam"] = float(np.corrcoef(np.argsort(np.argsort(per8)), np.argsort(np.argsort(lam)))[0, 1])
            out[ex][name] = row
            m = lambda key: 100 * np.mean(row[key]) if key in row else float("nan")  # noqa: E731
            tr = np.mean(row["transfer_init"]) if "transfer_init" in row and np.all(np.isfinite(row["transfer_init"])) else float("nan")
            print(f"{name:10s} {len(runs):<6d} {m('T1_median'):7.3f}/{m('T1_max'):7.2f} {m('T8_median'):7.3f}/{m('T8_max'):7.2f} {tr:8.2f} "
                  + " ".join(f"{100 * v[1]:12.4f}" for v in row["T8_median_by_lam"].values())
                  + f"  ({100 * row['T8_worst']['err']:.2f}, {row['T8_worst']['lam']:.2f})")
    (ROOT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()

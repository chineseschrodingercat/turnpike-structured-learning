#!/usr/bin/env python3
"""E2b: where does the train-POD projection fail, and does per-sample normalization help?

For each dataset and rank k, test-set relative projection errors for
    centered energy POD (report protocol), and
    normalized POD: SVD of u_i / ||u_i|| (train), no centering,
plus Spearman correlations of the error with x_a, log10 alpha and ||u||.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

K_LIST = [4, 6, 8, 12, 16]


def proj_err(U: np.ndarray, basis: np.ndarray, mean: np.ndarray) -> np.ndarray:
    C = U - mean
    R = C - (C @ basis.T) @ basis
    return np.linalg.norm(R, axis=1) / np.linalg.norm(U, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    report = {}
    for path in sorted(args.data_dir.glob("*_nx160_nt320.npz")):
        d = np.load(path)
        name = json.loads(str(d["metadata_json"]))["config"]
        U, tr, te = d["U"], d["train_idx"], d["test_idx"]
        xa, la = d["xa"][te], np.log10(d["alpha"][te])
        nrm = np.linalg.norm(U[te], axis=1)
        mean = U[tr].mean(axis=0)
        _, _, Vc = np.linalg.svd(U[tr] - mean, full_matrices=False)
        Un = U[tr] / np.linalg.norm(U[tr], axis=1, keepdims=True)
        _, _, Vn = np.linalg.svd(Un, full_matrices=False)
        res = {"norm_ratio_max_over_min": float(nrm.max() / nrm.min())}
        for k in K_LIST:
            ec = proj_err(U[te], Vc[:k], mean)
            en = proj_err(U[te], Vn[:k], np.zeros_like(mean))
            worst = np.argsort(ec)[-15:]
            res[str(k)] = {
                "centered_pod": {"median": float(np.median(ec)), "p90": float(np.percentile(ec, 90)), "max": float(ec.max())},
                "normalized_pod": {"median": float(np.median(en)), "p90": float(np.percentile(en, 90)), "max": float(en.max())},
                "centered_spearman": {"xa": float(spearmanr(ec, xa).statistic), "log_alpha": float(spearmanr(ec, la).statistic),
                                      "norm_u": float(spearmanr(ec, nrm).statistic)},
                "normalized_spearman": {"xa": float(spearmanr(en, xa).statistic), "log_alpha": float(spearmanr(en, la).statistic)},
                "worst15_centered": {"median_xa": float(np.median(xa[worst])), "median_log_alpha": float(np.median(la[worst]))},
            }
        report[name] = res
        print(name, f"(test ||u|| max/min = {res['norm_ratio_max_over_min']:.0f})")
        for k in K_LIST:
            r = res[str(k)]
            print(f"  k={k:>2} centered: med {r['centered_pod']['median']:.1e} p90 {r['centered_pod']['p90']:.1e} max {r['centered_pod']['max']:.1e}"
                  f" | normalized: med {r['normalized_pod']['median']:.1e} p90 {r['normalized_pod']['p90']:.1e} max {r['normalized_pod']['max']:.1e}"
                  f" | rho(err_c, xa/logalpha/||u||) = {r['centered_spearman']['xa']:+.2f}/{r['centered_spearman']['log_alpha']:+.2f}/{r['centered_spearman']['norm_u']:+.2f}"
                  f" | rho(err_n, xa/logalpha) = {r['normalized_spearman']['xa']:+.2f}/{r['normalized_spearman']['log_alpha']:+.2f}"
                  f" | worst15 xa {r['worst15_centered']['median_xa']:.3f} log_a {r['worst15_centered']['median_log_alpha']:+.2f}")
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Append an extra training pool (gen_semilinear_data.py --all-train) to the base semilinear dataset.

Validation and test splits stay those of the base set; train = base train + the first --n-extra extra samples.
Per-sample arrays missing from the extra pool (U_T4, U_T8, ...) are NaN-filled.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

SHARED = {"dt", "T_list", "metadata_json", "train_idx", "val_idx", "test_idx"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--extra", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n-extra", type=int, default=None)
    args = ap.parse_args()
    a, b = np.load(args.base), np.load(args.extra)
    n_a = a["X"].shape[0]
    k = b["X"].shape[0] if args.n_extra is None else args.n_extra
    out = {}
    for key in a.files:
        if key in SHARED:
            continue
        add = b[key][:k] if key in b.files else np.full((k,) + a[key].shape[1:], np.nan)
        out[key] = np.concatenate([a[key], add], axis=0)
    out["train_idx"] = np.concatenate([a["train_idx"], n_a + np.arange(k)])
    out["val_idx"], out["test_idx"] = a["val_idx"], a["test_idx"]
    for key in ("dt", "T_list", "metadata_json"):
        out[key] = a[key]
    np.savez_compressed(args.out, **out)
    print(f"{args.out}: train {len(out['train_idx'])}, val {len(out['val_idx'])}, test {len(out['test_idx'])} (base val/test)")


if __name__ == "__main__":
    main()

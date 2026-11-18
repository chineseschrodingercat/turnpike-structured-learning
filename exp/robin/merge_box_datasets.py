#!/usr/bin/env python3
"""Build a box-constrained dataset with a large training pool and the ORIGINAL test set.

train/val come from --train (a separately generated LHS set); test = the test samples of --test,
so every model in the data-scaling study is scored on the same 150 test samples.
--n-train limits the number of training samples (validation size stays fixed).
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

SHARED = {"dt", "T_list", "metadata_json", "train_idx", "val_idx", "test_idx"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--test", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    a, b = np.load(args.train), np.load(args.test)
    n_a = a["X"].shape[0]
    perm = np.random.default_rng(args.seed).permutation(n_a)
    val = np.sort(perm[: args.n_val])
    tr_pool = perm[args.n_val :]
    tr = np.sort(tr_pool[: args.n_train] if args.n_train else tr_pool)
    keep_a = np.sort(np.concatenate([tr, val]))
    remap = {old: new for new, old in enumerate(keep_a)}
    te_b = b["test_idx"]
    out = {}
    for k in a.files:
        if k in SHARED:
            continue
        out[k] = np.concatenate([a[k][keep_a], b[k][te_b]], axis=0)
    n_keep = len(keep_a)
    out["train_idx"] = np.array([remap[i] for i in tr])
    out["val_idx"] = np.array([remap[i] for i in val])
    out["test_idx"] = np.arange(n_keep, n_keep + len(te_b))
    for k in ("dt", "T_list", "metadata_json"):
        out[k] = a[k]
    np.savez_compressed(args.out, **out)
    print(f"{args.out}: train {len(tr)}, val {len(val)}, test {len(te_b)} (original test set)")


if __name__ == "__main__":
    main()

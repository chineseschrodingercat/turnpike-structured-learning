#!/usr/bin/env python3
"""Merge the frozen fresh-test JSONs (fresh_test_1d.py outputs) into one table and fresh_summary.json."""

import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
R = {}
for f in ("fresh_nn.json", "fresh_cnn.json", "fresh_cnn_res_T1.json", "fresh_krr.json"):
    p = d / f
    if p.exists():
        R.update(json.loads(p.read_text()))
(d / "fresh_summary.json").write_text(json.dumps(R, indent=1))
order = ["turnpike", "cnn", "cnn_res_T1", "krr", "reference_only", "short_solve_T1", "short_solve_T2"]
print("original test (mean of per-seed medians, %):")
for name in order:
    r = R.get(name, {})
    o = r.get("original_test_mean_median") or ({k: v["median"] for k, v in r["original_test"].items()} if "original_test" in r else None)
    if o:
        print(f"  {name:16s} " + "  ".join(f"T={T}: {100 * v:.3g}" for T, v in o.items()))
print("\nfresh test, seeds pooled: median % (max %)   total | init window | interior | term window")
for name in order:
    r = R.get(name)
    if not r:
        continue
    fr = r["fresh"]
    for T in ("1", "2", "3", "4", "6", "8", "12"):
        e = fr.get(T)
        if not e:
            continue
        cells = []
        for k in ("total_rel", "init_rel", "interior_rel", "term_rel"):
            s = e.get(k)
            cells.append("—" if s is None else f"{100 * s['median']:.3g} ({100 * s['max']:.3g})")
        print(f"  {name:16s} T={T:>2s}  " + " | ".join(cells))
    for k, v in fr.items():
        if k.startswith("transfer") and v:
            print(f"  {'':16s} {k}: median {v['median']:.3f}  p90 {v['p90']:.3f}  max {v['max']:.3f}")

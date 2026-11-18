#!/usr/bin/env python3
"""Print keys / shapes of one prediction file per directory and of every dataset-like npz under runs/ (no pred, no
per-sample files), to map predictions to their datasets."""

import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
seen = set()
for p in sorted(root.rglob("pred*.npz")):
    if p.parent in seen:
        continue
    seen.add(p.parent)
    z = np.load(p)
    print(f"PRED {p.relative_to(root)}: " + ", ".join(f"{k}{list(z[k].shape)}" for k in z.files))
for p in sorted(root.rglob("*.npz")):
    if p.name.startswith("pred") or "samples" in p.parts or p.stat().st_size < 2e6:
        continue
    z = np.load(p)
    print(f"DATA {p.relative_to(root)} ({p.stat().st_size / 1e6:.0f} MB): " + ", ".join(f"{k}{list(z[k].shape)}" for k in z.files))

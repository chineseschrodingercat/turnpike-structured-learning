#!/usr/bin/env python3
"""Single-thread timing for the linear strip-target problem (#6): reference part (static multi-input optimum) and exact
PCG solves at T = 1, 2, 8 for a few test samples -> the online cost of the deterministic short-horizon route
(reference + exact T_s solve) vs the full exact solve."""

import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_2d_strip_data import M, cfg_of  # noqa: E402
from robin2d_mi import pcg_solve, static_optimum_mi  # noqa: E402
from robin2d_multi import assemble_strip  # noqa: E402

d = np.load(sys.argv[1])
rows = []
for i in d["test_idx"][:3]:
    row, r = d["X"][i], {"idx": int(i)}
    t0 = time.perf_counter()
    static_optimum_mi(assemble_strip(cfg_of(row, 1), M))
    r["reference_s"] = time.perf_counter() - t0
    for T in (1, 2, 8):
        t0 = time.perf_counter()
        u, info = pcg_solve(assemble_strip(cfg_of(row, T), M), float(T))
        r[f"exact_T{T}_s"], r[f"pcg_iters_T{T}"] = time.perf_counter() - t0, info["iters"]
        if T in (1, 8):
            ref = d[f"U_T{T}"][i] if T == 1 else d["U_T8_test"][list(d["test_idx"]).index(i)]
            r[f"rel_vs_data_T{T}"] = float(np.linalg.norm(u - ref) / np.linalg.norm(ref))
    rows.append(r)
    print(json.dumps(r), flush=True)
med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "idx"}
pathlib.Path(sys.argv[2]).write_text(json.dumps({"protocol": "single thread, wall clock per mu", "samples": rows, "median": med}, indent=1))
print(json.dumps(med))

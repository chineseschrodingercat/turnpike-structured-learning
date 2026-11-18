#!/usr/bin/env python3
"""Code-path smoke test only: a 6-sample data.npz in the format of gen_multispot_data.py --merge, built from the two
smoke samples (runs/multispot_smoke/samples). T = 4, 8 controls are the T = 2 control tiled in time (not physical)."""

import json
import pathlib

import numpy as np

S = pathlib.Path("runs/multispot_smoke/samples")
rows = [dict(np.load(S / f"sample_{i:04d}.npz")) for i in (0, 3)] * 3
n = len(rows)
te, va, tr = np.array([0, 1]), np.array([2]), np.array([3, 4, 5])
pay = {}
for kind in ("nl", "lin"):
    for T in (1, 2):
        pay[f"U_{kind}_T{T}"] = np.stack([r[f"{kind}_{T}"] for r in rows])
    for T in (4, 8):
        pay[f"U_{kind}_T{T}_test"] = np.stack([np.tile(rows[i][f"{kind}_2"], (T // 2, 1)) for i in te])
X = np.stack([r["row"] for r in rows]) + 1e-3 * np.random.default_rng(0).standard_normal((n, 16))
np.savez_compressed("runs/multispot_smoke/fake_data.npz", X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=99, nh=100,
                    u0=np.stack([r["u0"] for r in rows]), uhat=np.stack([r["uhat"] for r in rows]), nharm=np.stack([r["nharm"] for r in rows]),
                    lam1=np.array([float(r["lam1"]) for r in rows]), metadata_json=json.dumps({"fake": True}), **pay)
print("wrote runs/multispot_smoke/fake_data.npz")

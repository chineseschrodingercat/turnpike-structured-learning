#!/usr/bin/env python3
"""Multi-spot results: table of all model JSONs in runs/multispot_data/models (pooled over seeds from the saved
predictions where available) and an example figure (semilinear, T = 8, test sample at the median error of tp_local)."""

import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

D = pathlib.Path("runs/multispot_data")
M = D / "models"
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else D
d = np.load(D / "data.npz")
te = d["test_idx"]
X = d["X"]


def rel(a, b):
    return np.linalg.norm((a - b).reshape(len(a), -1), axis=1) / np.linalg.norm(b.reshape(len(b), -1), axis=1)


groups = {}  # per-seed JSONs (e.g. nl_tp_local_s0..s2) are pooled into one row, named by the file stem without _s<seed>
for f in sorted(M.glob("*.json")):
    r = json.loads(f.read_text())
    key = re.sub(r"_s\d+$", "", f.stem)
    if key in groups:
        groups[key]["runs"] += r["runs"]
    else:
        groups[key] = r
rows = []
for name, r in groups.items():
    runs = r["runs"]
    get = lambda T, k: np.mean([x["test"][str(T)].get(k, np.nan) for x in runs])  # noqa: E731
    rows.append({"name": name, "kind": r["kind"], "model": r["model"], "pred_prefix": r.get("pred_prefix", f"pred_{r['kind']}_{r['model']}"),
                 "train_T": r["train_T"], "epochs": r["epochs"], "seeds": len(runs), "params": runs[0]["params"],
                 "T1_med": get(1, "median"), "T8_med": get(8, "median"), "T8_max": get(8, "max"),
                 "T8_init": get(8, "init_median"), "T8_term": get(8, "term_median"), "T8_interior": get(8, "interior_median"),
                 "transfer_init": np.mean([x["test"]["transfer_init"]["median"] for x in runs]),
                 "transfer_term": np.mean([x["test"]["transfer_term"]["median"] for x in runs]),
                 "transfer_term_p90": np.mean([x["test"]["transfer_term"]["p90"] for x in runs]),
                 "train_min": np.mean([x["train_seconds"] for x in runs]) / 60})
# pooled over seeds from the saved predictions
for r in rows:
    preds = sorted((M / "pred").glob(f"{r['pred_prefix']}_seed*.npz"))
    if preds:
        U8 = d[f"U_{r['kind']}_T8_test"].astype(float)
        e = np.concatenate([rel(np.load(p)["U_T8"].astype(float), U8) for p in preds])
        r["T8_pooled_med"], r["T8_pooled_max"] = float(np.median(e)), float(e.max())
(D / "results_table.json").write_text(json.dumps(rows, indent=1))
hdr = f"{'name':26s} {'trainT':8s} {'ep':>5s} {'seeds':>5s} {'params':>8s} {'T1 med%':>8s} {'T8 med%':>8s} {'T8 max%':>8s} {'init%':>7s} {'term%':>7s} {'inter%':>7s} {'R_I':>5s} {'R_E':>5s} {'R_E90':>6s} {'min':>6s}"
print(hdr)
for r in rows:
    print(f"{r['name']:26s} {str(r['train_T']):8s} {r['epochs']:5d} {r['seeds']:5d} {r['params']:8d} {100 * r['T1_med']:8.3f} {100 * r['T8_med']:8.3f} {100 * r['T8_max']:8.3f} "
          f"{100 * r['T8_init']:7.3f} {100 * r['T8_term']:7.3f} {100 * r['T8_interior']:7.3f} {r['transfer_init']:5.2f} {r['transfer_term']:5.2f} "
          f"{r['transfer_term_p90']:6.2f} {r['train_min']:6.1f}" + (f"   pooled T8 {100 * r['T8_pooled_med']:.3f}/{100 * r['T8_pooled_max']:.2f}" if "T8_pooled_med" in r else ""))

# example figure
ours = sys.argv[2] if len(sys.argv) > 2 else "pred_nl_tp_qp"  # prediction prefix of "ours", then the baselines drawn in the traces
base = sys.argv[3:] if len(sys.argv) > 3 else ["pred_nl_cnn", "pred_nl_cnn_res"]
pf = M / "pred" / f"{ours}_seed0.npz"
if pf.exists():
    U8 = d["U_nl_T8_test"].astype(float)
    P = np.load(pf)["U_T8"].astype(float)
    C = {b: np.load(M / "pred" / f"{b}_seed0.npz")["U_T8"].astype(float) for b in base if (M / "pred" / f"{b}_seed0.npz").exists()}
    e = rel(P, U8)
    order = np.argsort(e)
    t = np.arange(1, U8.shape[1] + 1) / 320
    s = np.arange(1, 100) / 100
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.4))
    for r_, (lab, j) in enumerate([("median", order[len(order) // 2]), ("worst", order[-1])]):
        i = te[j]
        row = X[i]
        th = [row[4 + 6 * q + 3] * t + row[4 + 6 * q + 5] for q in range(2)]
        Yd = sum(row[4 + 6 * q] * np.exp(-0.5 * ((s[None, :] - row[4 + 6 * q + 1] - row[4 + 6 * q + 2] * np.sin(th[q])[:, None]) / 10 ** row[4 + 6 * q + 4]) ** 2)
                 for q in range(2))
        u, up = U8[j], P[j]
        v = np.abs(u).max()
        for c_, (Z, ttl, kw) in enumerate([(Yd, f"{lab} sample: target", dict(cmap="viridis")), (u, "exact semilinear control", dict(cmap="RdBu_r", vmin=-v, vmax=v)),
                                           (np.maximum(np.abs(up - u) / v, 1e-7), f"ours: |error| / max|u| (rel. L2 {100 * e[j]:.2f}%)", dict(cmap="magma", norm=LogNorm(1e-6, 1e-1)))]):
            pcm = ax[r_, c_].pcolormesh(t, s, Z.T, shading="auto", rasterized=True, **kw)
            ax[r_, c_].set_title(ttl)
            ax[r_, c_].set_xlabel("$t$")
            ax[r_, c_].set_ylabel("$s$")
            fig.colorbar(pcm, ax=ax[r_, c_], pad=0.02)
        a = ax[r_, 3]
        jj = int(np.argmax(np.abs(u).mean(axis=0)))
        a.plot(t, u[:, jj], "k", lw=1.6, label="exact")
        a.plot(t, up[:, jj], "C3--", lw=1.0, label=f"ours ({100 * e[j]:.2f}%)")
        for q, (b, Cb) in enumerate(C.items()):
            lab = b.replace("pred_nl_", "").replace("cnn_res", "CNN-res").replace("cnn", "CNN")
            a.plot(t, Cb[j, :, jj], f"C{q}:", lw=1.0, label=f"{lab} ({100 * rel(Cb[j:j + 1], u[None])[0]:.2f}%)")
        a.set_xlabel("$t$")
        a.set_title(f"time trace at $s$ = {s[jj]:.2f}, $\\gamma$ = {10 ** row[3]:.0f}")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "multispot_results_example.png", dpi=110)
    print("wrote", out / "multispot_results_example.png")

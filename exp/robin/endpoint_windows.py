#!/usr/bin/env python3
"""Endpoint-window error decomposition for every saved prediction (Codex review point: a whole-horizon relative error
falls like T^{-1/2} when the error is confined to fixed-width end windows, so a decreasing total error alone does not
show transfer).

For each experiment, model, horizon T and test sample, the error e = u_pred - u* is split into
    initial window [0, tau], terminal window [T - tau, T], interior (tau, T - tau)
and reported as relative errors within each window (||e||_W / ||u*||_W, Euclidean over s, L2 in t) plus the absolute
window errors. Windows have a fixed physical width, so the initial-window error at T = 8 is directly comparable with
the one at T = 1: if the learned end corrections transfer, they agree. Rows:
    models        all saved predictions, seeds pooled (each sample counted once per seed)
    reference     the computed horizon-independent part only (static plateau / periodic orbit / linear periodic orbit)
    short_solve   deterministic route: reference in the interior + the EXACT short-horizon solution at the two ends
                  (constant targets: T_s = 1, windows of 0.5; advection: T_s = 2 with matching end phase, windows of 1).
                  Its accuracy is the superposition error; its online cost is one exact short-horizon solve.
Output: JSON with median / p90 / max per (experiment, model, T, window) and the per-sample endpoint transfer ratio
||e||_{W_I}(T_max) / ||e||_{W_I}(T_min) (same for W_E)."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

DT = 1.0 / 320

# experiment: dataset, window tau, reference kind, short-solve horizon, model -> prediction glob
EXPS = {
    "1d_const": dict(data="turnpike/turnpike_data.npz", tau=0.5, ref="ubar", short=(1, 2),
                     models={"ours": "unified/pred_const1d_tp_T1/pred_turnpike_seed*.npz", "cnn": "unified/pred_const1d_cnn_T12/pred_cnn_seed*.npz"}),
    "1d_osc": dict(data="turnpike_osc/osc_data.npz", tau=0.5, ref="osc", short=None,
                   models={"ours": "unified/pred_osc1d_tp_T1/pred_turnpike_seed*.npz", "cnn": "unified/pred_osc1d_cnn_T12/pred_cnn_seed*.npz"}),
    "2d_scalar": dict(data="robin2d/data_2d.npz", tau=0.5, ref="ubar", short=(1, 2),
                      models={"ours": "unified/pred_scalar2d_tp_T1/pred_turnpike_seed*.npz", "cnn": "unified/pred_scalar2d_cnn_T12/pred_cnn_seed*.npz"}),
    "2d_mi": dict(data="robin2d_mi/data_mi.npz", tau=0.5, ref="ubar", short=(1, 2),
                  models={"ours": "unified/pred_mi2d_tp_T1/pred_turnpike_seed*.npz", "cnn": "unified/pred_mi2d_cnn_T12/pred_cnn_seed*.npz"}),
    "2d_strip": dict(data="robin2d_strip/data_strip.npz", tau=0.5, ref="ubar", short=(1, 2),
                     models={"ours_tau_linear": "examples2d/strip/pred/pred_tau_turnpike_seed*.npz"}),
    "1d_semilinear": dict(data="semilinear/data_semilinear_big.npz", tau=0.5, ref="ubar", short=(1, 2),
                          models={"ours": "semilinear/models/pred_T1_big_r8_lam1_xlong/pred_turnpike_seed*.npz",
                                  "cnn": "semilinear/models/pred_T12_big/pred_cnn_seed*.npz"}),
    "1d_semilinear_n700": dict(data="semilinear/data_semilinear.npz", tau=0.5, ref="ubar", short=(1, 2),
                               models={"ours": "semilinear/models/pred_T1_r8_lam1_long/pred_turnpike_seed*.npz",
                                       "cnn": "semilinear/models/pred_T12/pred_cnn_seed*.npz",
                                       "deeponet": "semilinear/models/pred_T12/pred_deeponet_seed*.npz",
                                       "fno": "semilinear/models/pred_T12/pred_fno_seed*.npz"}),
    "y0_linear": dict(data="y0/data_y0_linear.npz", tau=0.5, ref="ubar", short=(1, 2),
                      models={"ours": "y0/linear/pred/pred_tp_lin_seed*.npz", "cnn_T1": "y0/linear/pred/pred_cnn_seed*.npz"}),
    "y0_semilinear": dict(data="y0/data_y0_semilinear_big.npz", tau=0.5, ref="ubar", short=(1, 2),
                          models={"ours": "y0/semilinear_big/pred/pred_tp_hyb_seed*.npz", "cnn_T1": "y0/semilinear_big/pred/pred_cnn_T1_seed*.npz"}),
    "2d_semilinear": dict(data="semilinear2d/data_semilinear2d_strong.npz", tau=0.5, ref="ubar", short=(1, 2),
                          models={"ours": "semilinear2d/models_strong/pred_tp_r8_T1/pred_turnpike_seed*.npz",
                                  "cnn": "semilinear2d/models_strong/pred_cnn_T12/pred_cnn_seed*.npz",
                                  "deeponet": "semilinear2d/models_strong/pred_deeponet_T12/pred_deeponet_seed*.npz",
                                  "fno": "semilinear2d/models_strong/pred_fno_T12/pred_fno_seed*.npz"}),
    "2d_semilinear_strip": dict(data="semilinear2d_mi/data.npz", tau=0.5, ref="ubar", short=(1, 2),
                                models={"ours": "semilinear2d_mi/models/pred/pred_turnpike_r8_seed*.npz",
                                        "ours_tau_linear": "semilinear2d_mi/models/pred/pred_tau_turnpike_r8_seed*.npz",
                                        "cnn": "semilinear2d_mi/models/pred/pred_cnn_r4_seed*.npz"}),
    "moving": dict(data="moving/data.npz", tau=0.5, ref="pred:moving/models/pred/pred_periodic_seed0.npz", short=None,
                   models={"ours": "moving/models/pred/pred_turnpike_seed*.npz", "cnn": "moving/models/pred/pred_cnn_seed*.npz"}),
    "semilinear_moving": dict(data="semilinear_moving/data.npz", tau=0.5, ref="pred:semilinear_moving/models/pred/pred_lin_only_seed0.npz",
                              short=None, models={"ours": "semilinear_moving/models/pred/pred_tp_local_seed*.npz",
                                                  "ours_global": "semilinear_moving/models/pred/pred_tp_lin_seed*.npz",
                                                  "cnn": "semilinear_moving/models/pred/pred_cnn_seed*.npz",
                                                  "cnn_res_T1": "semilinear_moving/models/pred/pred_cnn_res_seed*.npz",
                                                  "cnn_res_T12": "semilinear_moving/models/pred_T12/pred_cnn_res_seed*.npz"}),
    "advection": dict(data="advect_data/advect_data.npz", tau=1.0, ref="pred:advect_data/pred/pred_per_only_seed0.npz", short=(2,),
                      models={"ours_mirror": "advect_data/pred/pred_tp_transport_mirror_seed*.npz",
                              "ours_mirror_w48": "advect_data/pred/pred_tp_transport_mirror_w48_seed*.npz",
                              "ours_base": "advect_data/pred/pred_tp_transport_seed*.npz",
                              "ours_w48": "advect_data/pred/pred_tp_transport_w48_seed*.npz",
                              "cnn_per": "advect_data/pred/pred_cnn_per_seed*.npz", "cnn": "advect_data/pred/pred_cnn_seed*.npz",
                              "fno": "advect_data/pred/pred_fno_seed*.npz", "deeponet": "advect_data/pred/pred_deeponet_seed*.npz"}),
}


def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x)), "n": int(len(x))}


class Exp:
    def __init__(self, root, cfg):
        self.root, self.cfg = root, cfg
        self.d = np.load(root / cfg["data"])
        self.test_idx = self.d["test_idx"]
        self.pos = {int(i): k for k, i in enumerate(self.test_idx)}
        self.ref_pred = np.load(root / cfg["ref"][5:]) if cfg["ref"].startswith("pred:") else None

    def horizons(self):
        Ts = []
        for T in (1, 2, 4, 8):
            if f"U_T{T}" in self.d.files or f"U_T{T}_test" in self.d.files or (self.cfg["ref"] == "osc" and f"U_const_T{T}" in self.d.files):
                Ts.append(T)
        return Ts

    def exact(self, T, idx):
        d = self.d
        if self.cfg["ref"] == "osc":
            A = d["A"][idx][:, None]
            return 1.25 * d[f"U_const_T{T}"][idx] + A * d[f"U_sin_T{T}"][idx]
        if f"U_T{T}" in d.files and d[f"U_T{T}"].shape[0] == len(d["X"]):
            return d[f"U_T{T}"][idx]
        return d[f"U_T{T}_test"][[self.pos[int(i)] for i in idx]]

    def reference(self, T, idx):
        d, kind = self.d, self.cfg["ref"]
        N = int(round(T / DT))
        if kind == "ubar":
            ub = d["u_bar"][idx]
            return np.repeat(ub[:, None], N, axis=1) if ub.ndim == 1 else np.repeat(ub[:, None, :], N, axis=1)
        if kind == "osc":
            t = np.arange(1, N + 1) * DT
            uh, om, A = d["u_hat"][idx], d["omega"][idx], d["A"][idx]
            per = np.imag(uh[:, None] * np.exp(1j * om[:, None] * t[None, :]))
            return 1.25 * d["u_bar"][idx][:, None] + A[:, None] * per
        rp = self.ref_pred
        if f"U_T{T}" not in rp.files:
            return None
        rpos = {int(i): k for k, i in enumerate(rp["test_idx"])}
        return rp[f"U_T{T}"][[rpos[int(i)] for i in idx]]


def windows(N, tau):
    k = int(round(tau / DT))
    I = np.zeros(N, bool)
    E = np.zeros(N, bool)
    I[:k] = True
    E[N - k:] = True
    E &= ~I  # T = 2 tau: the two windows tile the horizon
    M = ~(I | E)
    return {"init": I, "term": E, "interior": M}


def errs(P, U, tau):
    """Per-sample window errors for predictions P and exact U, both (B, N[, m])."""
    B, N = U.shape[:2]
    D = (P - U).reshape(B, N, -1)
    Uf = U.reshape(B, N, -1)
    e2 = (D**2).sum(-1) * DT
    u2 = (Uf**2).sum(-1) * DT
    out = {"total_rel": np.sqrt(e2.sum(1) / u2.sum(1))}
    for w, msk in windows(N, tau).items():
        if msk.any():
            out[f"{w}_rel"] = np.sqrt(e2[:, msk].sum(1) / u2[:, msk].sum(1))
            out[f"{w}_abs"] = np.sqrt(e2[:, msk].sum(1))
    return out


def short_solve(ex, T, Ts, idx, tau):
    """Reference in the interior + exact short-horizon solution in the two end windows (tau = Ts / 2)."""
    if T <= Ts:
        return None
    Ref, Us = ex.reference(T, idx), ex.exact(Ts, idx)
    Ref_s = ex.reference(Ts, idx)
    if Ref is None or Ref_s is None:
        return None
    k = int(round(tau / DT))
    P = Ref.copy()
    P[:, :k] = Us[:, :k]
    P[:, -k:] = Us[:, -k:]
    return P


def run(root, names):
    res = {}
    for name in names:
        cfg = EXPS[name]
        ex = Exp(root, cfg)
        tau = cfg["tau"]
        Ts_all = ex.horizons()
        out = {"tau": tau, "horizons": Ts_all, "rows": {}}
        rows = {lab: sorted(glob.glob(str(root / pat))) for lab, pat in cfg["models"].items()}
        rows = {lab: f for lab, f in rows.items() if f}
        per_sample = {}
        for lab, files in rows.items():
            acc = {}
            for f in files:
                z = np.load(f)
                idx = z["test_idx"]
                for T in Ts_all:
                    if f"U_T{T}" not in z.files:
                        continue
                    e = errs(z[f"U_T{T}"].astype(float), ex.exact(T, idx).astype(float), tau)
                    for k, v in e.items():
                        acc.setdefault(T, {}).setdefault(k, []).append(v)
            per_sample[lab] = {T: {k: np.concatenate(v) for k, v in dd.items()} for T, dd in acc.items()}
        idx = ex.test_idx
        ref_acc, short_acc = {}, {}
        for T in Ts_all:
            U = ex.exact(T, idx).astype(float)
            R = ex.reference(T, idx)
            if R is not None:
                ref_acc[T] = errs(R.astype(float), U, tau)
            for Ts in cfg["short"] or ():
                if Ts in Ts_all:
                    P = short_solve(ex, T, Ts, idx, Ts / 2)
                    if P is not None:
                        short_acc.setdefault(Ts, {})[T] = errs(P.astype(float), U, tau)
        per_sample["reference_only"] = ref_acc
        for Ts, acc in short_acc.items():
            per_sample[f"short_solve_T{Ts}"] = acc
        for lab, dd in per_sample.items():
            row = {str(T): {k: summ(v) for k, v in e.items()} for T, e in dd.items()}
            Tk = sorted(dd)
            if len(Tk) >= 2:  # endpoint transfer: same samples, smallest vs largest horizon
                t0, t1 = Tk[0], Tk[-1]
                for w in ("init", "term"):
                    a0, a1 = dd[t0].get(f"{w}_abs"), dd[t1].get(f"{w}_abs")
                    if a0 is not None and a1 is not None and len(a0) == len(a1):
                        ok = dd[t0][f"{w}_rel"] > 1e-6  # ratios of float32 round-off are meaningless
                        if ok.any():
                            row[f"transfer_{w}_T{t1}_over_T{t0}"] = summ(a1[ok] / a0[ok])
            out["rows"][lab] = row
        res[name] = out
        print_table(name, out)
    return res


def fmt(s, k="median"):
    return "â€”" if s is None else f"{100 * s[k]:.3g}"


def print_table(name, out):
    print(f"\n=== {name} (tau = {out['tau']}, horizons {out['horizons']}) â€” median % (max %)")
    print(f"{'row':22s} {'T':>3s} {'total':>16s} {'init window':>16s} {'interior':>16s} {'term window':>16s}")
    for lab, row in out["rows"].items():
        for T in out["horizons"]:
            r = row.get(str(T))
            if r is None:
                continue
            cells = []
            for key in ("total_rel", "init_rel", "interior_rel", "term_rel"):
                s = r.get(key)
                cells.append("â€”" if s is None else f"{fmt(s)} ({fmt(s, 'max')})")
            print(f"{lab:22s} {T:>3d} " + " ".join(f"{c:>16s}" for c in cells))
        for k, v in row.items():
            if k.startswith("transfer") and v is not None:
                print(f"{'':22s}     {k}: median {v['median']:.3f}, p90 {v['p90']:.3f}, max {v['max']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/endpoint_windows.json"))
    args = ap.parse_args()
    names = args.only or list(EXPS)
    res = run(args.runs, names)
    old = json.loads(args.out.read_text()) if args.out.exists() and args.only else {}
    old.update(res)
    args.out.write_text(json.dumps(old, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

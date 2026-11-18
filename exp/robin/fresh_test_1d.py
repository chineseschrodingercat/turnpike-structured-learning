#!/usr/bin/env python3
"""Frozen fresh test for the 1D LQ constant-target problem (#1).

The model configurations are frozen at the ones of results/unified/const1d_*.json (same code, hyper-parameters and seeds,
retrained because weights were not stored); the original test numbers are recomputed as a reproducibility check. They
are then evaluated, without any further change, on a FRESH parameter sample (new LHS seed) and on horizons never used
in development (T = 3, 6, 12, besides 1, 2, 4, 8). Rows:
    turnpike     TurnpikeNet r = 4, shared rates, floor lambda_1(beta), trained on T = 1             (ours, frozen)
    cnn          dilated coordinate-free CNN, trained on T in {1, 2}                                   (baseline, frozen)
    cnn_res_T1   the same CNN as a residual on the plateau (u / u_bar = 1 + CNN), trained on T = 1 only: the same
                 reference information and the same training data as ours
    krr          no neural network: per-sample least squares of u / u_bar - 1 on a fixed dictionary of end layers
                 e^{-c_k lambda_1(beta) t}, e^{-c_k lambda_1(beta) (T - t)} at T = 1, then kernel ridge regression
                 mu -> coefficients (hyper-parameters chosen on the validation set at T = 1)
    reference_only, short_solve_T1, short_solve_T2   non-learned rows (exact short-horizon solve at the ends)
Metrics: whole-horizon relative error and the endpoint-window decomposition of endpoint_windows.py (tau = 0.5)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines_multihorizon as bm  # noqa: E402
from endpoint_windows import errs, summ  # noqa: E402
from turnpike_surrogate import first_eig, grid, rel_err, summarize  # noqa: E402

TAU = 0.5
C_MULT = np.array([1.0, 1.5, 2.5, 4.0, 7.0, 12.0, 20.0, 40.0])


class ResCNN(bm.DilatedCNN):
    """u / u_bar = 1 + CNN(...), last layer initialized at ~0."""

    def __init__(self, n_par):
        super().__init__(n_par)
        with torch.no_grad():
            self.proj[-1].weight.mul_(0.05)
            self.proj[-1].bias.zero_()

    def forward(self, x, yd, t, s):
        return 1.0 + super().forward(x, yd, t, s)


def fresh_problem(path, p_orig, X_orig, tr):
    d = np.load(path)
    mean, std = X_orig[tr].mean(0), X_orig[tr].std(0)
    Ts = [int(T) for T in d["T_list"]]
    return {"x": torch.tensor((d["X"] - mean) / std), "floor": torch.tensor([first_eig(b) for b in d["beta"]]),
            "ubar": d["u_bar"], "U": {T: d[f"U_T{T}"] for T in Ts}, "Ts": Ts, "n": len(d["X"])}


def predict_fresh(model, f, T, dt, chunk=50):
    t, s = grid(T, dt)
    out = []
    for k in range(0, f["n"], chunk):
        idx = torch.arange(k, min(k + chunk, f["n"]))
        x = f["x"][idx]
        if isinstance(model, bm.TurnpikeConst):
            out.append(model.net(x, t, s, f["floor"][idx]))
        else:
            out.append(model(x, torch.ones(len(idx), len(t)), t, s))
    return torch.cat(out).cpu().numpy() * f["ubar"][:, None]


def window_rows(P_by_T, U_by_T):
    res = {}
    for T in sorted(P_by_T):
        e = errs(P_by_T[T].astype(float), U_by_T[T].astype(float), TAU)
        res[T] = e
    return res


def summarize_rows(per_T):
    out = {str(T): {k: summ(v) for k, v in e.items()} for T, e in per_T.items()}
    Tk = sorted(per_T)
    for w in ("init", "term"):
        a0, a1 = per_T[Tk[0]].get(f"{w}_abs"), per_T[Tk[-1]].get(f"{w}_abs")
        ok = per_T[Tk[0]][f"{w}_rel"] > 1e-6
        if ok.any():
            out[f"transfer_{w}_T{Tk[-1]}_over_T{Tk[0]}"] = summ(a1[ok] / a0[ok])
    return out


# ---------------- kernel ridge model on a fixed end-layer dictionary ----------------
def dictionary(floor, T, dt):
    N = int(round(T / dt))
    t = np.arange(1, N + 1) * dt
    lam = floor * C_MULT
    return np.concatenate([np.exp(-lam[None, :] * t[:, None]), np.exp(-lam[None, :] * (T - t)[:, None])], 1)  # (N, 16)


def fit_coefs(Y1, floors, dt, eps):
    C = []
    for y, fl in zip(Y1, floors):
        Phi = dictionary(fl, 1.0, dt)
        A = Phi.T @ Phi
        C.append(np.linalg.solve(A + eps * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), Phi.T @ (y - 1.0)))
    return np.array(C)


def rbf(A, B, ell):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / ell**2)


class KRR:
    def fit(self, Xtr, Ctr, ell, lam):
        self.Xtr, self.ell = Xtr, ell
        self.mu, self.sd = Ctr.mean(0), Ctr.std(0) + 1e-30
        K = rbf(Xtr, Xtr, ell)
        self.alpha = np.linalg.solve(K + lam * np.eye(len(K)), (Ctr - self.mu) / self.sd)
        return self

    def coefs(self, X):
        return rbf(X, self.Xtr, self.ell) @ self.alpha * self.sd + self.mu

    def predict(self, X, floors, T, dt):
        C = self.coefs(X)
        return np.array([1.0 + dictionary(fl, T, dt) @ c for fl, c in zip(floors, C)])


def run_krr(d, X_std, floors, dt, fresh, log):
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    Y1 = d["U_T1"] / d["u_bar"][:, None]
    best = None
    for eps in (1e-10, 1e-8, 1e-6, 1e-4):
        Ctr = fit_coefs(Y1[tr], floors[tr], dt, eps)
        floor_fit = np.median([np.linalg.norm(1 + dictionary(floors[i], 1.0, dt) @ c - Y1[i]) / np.linalg.norm(Y1[i])
                               for i, c in zip(tr, Ctr)])
        for ell in (0.3, 0.5, 0.8, 1.2, 2.0):
            for lam in (1e-8, 1e-6, 1e-4, 1e-2):
                m = KRR().fit(X_std[tr], Ctr, ell, lam)
                P = m.predict(X_std[va], floors[va], 1.0, dt)
                v = float(np.median(np.linalg.norm(P - Y1[va], axis=1) / np.linalg.norm(Y1[va], axis=1)))
                if best is None or v < best[0]:
                    best = (v, eps, ell, lam, floor_fit)
    v, eps, ell, lam, floor_fit = best
    log(f"krr: best val median {v:.3e} at eps {eps:g}, ell {ell}, lam {lam:g} (per-sample dictionary fit at T=1: median {floor_fit:.2e})")
    m = KRR().fit(X_std[tr], fit_coefs(Y1[tr], floors[tr], dt, eps), ell, lam)
    orig = {T: m.predict(X_std[te], floors[te], T, dt) * d["u_bar"][te, None] for T in (1, 2, 4, 8)}
    fr = {T: m.predict(fresh["x"].numpy(), fresh["floor"].numpy(), T, dt) * fresh["ubar"][:, None] for T in fresh["Ts"]}
    return orig, fr, {"val_median_T1": v, "eps": eps, "ell": ell, "lam": lam, "dictionary_fit_T1_median": floor_fit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--fresh", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["turnpike", "cnn", "cnn_res_T1", "krr", "references"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true", help="float32 (GPU runs of the CNN rows); the frozen configs used float64 on CPU")
    args = ap.parse_args()
    dtype = torch.float32 if args.fp32 else torch.float64
    torch.set_default_dtype(dtype)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    log = lambda s: print(s, flush=True)  # noqa: E731
    p = bm.load_problem("const", args.data)
    dev = lambda v: v.to(args.device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else (  # noqa: E731
        v.to(args.device) if isinstance(v, torch.Tensor) else v)
    p = {k: ({kk: dev(vv) for kk, vv in v.items()} if isinstance(v, dict) else dev(v)) for k, v in p.items()}
    d = np.load(args.data)
    dt = p["dt"]
    fresh = fresh_problem(args.fresh, p, d["X"], d["train_idx"])
    fresh["x"], fresh["floor"] = dev(fresh["x"]), dev(fresh["floor"])
    te = p["te"]
    U_orig = {T: d[f"U_T{T}"][d["test_idx"]] for T in (1, 2, 4, 8)}
    log(f"fresh set: {fresh['n']} samples, horizons {fresh['Ts']}")
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    cfg = {"turnpike": dict(train_T=[1], epochs=6000, lr=2e-3, batch=700, make=lambda: bm.TurnpikeConst(p, r=4, floor="lam1")),
           "cnn": dict(train_T=[1, 2], epochs=300, lr=1e-3, batch=50, make=lambda: bm.DilatedCNN(p["x"].shape[1])),
           "cnn_res_T1": dict(train_T=[1], epochs=300, lr=1e-3, batch=50, make=lambda: ResCNN(p["x"].shape[1]))}
    for name in args.models:
        if name in cfg:
            c = cfg[name]
            orig_rows, fresh_win, orig_tot = [], {}, []
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = c["make"]()
                t0 = time.perf_counter()
                model = bm.train(model, p, c["train_T"], c["epochs"], c["lr"], c["batch"])
                with torch.no_grad():
                    o = {str(T): summarize(rel_err(bm.predict(model, p, te, T), p["Y"][T][te]).cpu().numpy()) for T in (1, 2, 4, 8)}
                    Pf = {T: predict_fresh(model, fresh, T, dt) for T in fresh["Ts"]}
                orig_tot.append(o)
                for T, e in window_rows(Pf, fresh["U"]).items():
                    for k, v in e.items():
                        fresh_win.setdefault(T, {}).setdefault(k, []).append(v)
                log(f"{name} seed {seed} ({time.perf_counter() - t0:.0f}s): original test T=8 med {o['8']['median']:.3e}; fresh "
                    + "  ".join(f"T={T} {np.median(np.concatenate(fresh_win[T]['total_rel'][-1:])):.3e}" for T in fresh["Ts"]))
            report[name] = {"original_test_per_seed": orig_tot,
                            "original_test_mean_median": {T: float(np.mean([o[T]["median"] for o in orig_tot])) for T in ("1", "2", "4", "8")},
                            "fresh": summarize_rows({T: {k: np.concatenate(v) for k, v in e.items()} for T, e in fresh_win.items()})}
        elif name == "krr":
            X_std = ((d["X"] - d["X"][d["train_idx"]].mean(0)) / d["X"][d["train_idx"]].std(0))
            floors = p["floor"].cpu().double().numpy()
            fresh = dict(fresh, x=fresh["x"].cpu().double(), floor=fresh["floor"].cpu().double())
            orig, fr, info = run_krr(d, X_std, floors, dt, fresh, log)
            o = {str(T): summarize(np.linalg.norm(orig[T] - U_orig[T], axis=1) / np.linalg.norm(U_orig[T], axis=1)) for T in orig}
            report["krr"] = {"info": info, "original_test": o, "fresh": summarize_rows(window_rows(fr, fresh["U"]))}
            log(f"krr: original test T=8 med {o['8']['median']:.3e}")
        elif name == "references":
            Ts = fresh["Ts"]
            ref = {T: np.repeat(fresh["ubar"][:, None], fresh["U"][T].shape[1], 1) for T in Ts}
            report["reference_only"] = {"fresh": summarize_rows(window_rows(ref, fresh["U"]))}
            for Tsh in (1, 2):
                P = {}
                for T in Ts:
                    if T <= Tsh:
                        continue
                    k = int(round(Tsh / 2 / dt))
                    Q = ref[T].copy()
                    Q[:, :k] = fresh["U"][Tsh][:, :k]
                    Q[:, -k:] = fresh["U"][Tsh][:, -k:]
                    P[T] = Q
                report[f"short_solve_T{Tsh}"] = {"fresh": summarize_rows(window_rows(P, fresh["U"]))}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=float))
    # table
    log("\nfresh test (median % (max %)): total | init window | interior | term window")
    for name, r in report.items():
        for T in [str(T) for T in fresh["Ts"]]:
            e = r["fresh"].get(T)
            if not e:
                continue
            cells = []
            for k in ("total_rel", "init_rel", "interior_rel", "term_rel"):
                s = e.get(k)
                cells.append("—" if s is None else f"{100 * s['median']:.3g} ({100 * s['max']:.3g})")
            log(f"{name:16s} T={T:>2s}  " + " | ".join(cells))
        for k, v in r["fresh"].items():
            if k.startswith("transfer") and v:
                log(f"{'':16s} {k}: median {v['median']:.3f} p90 {v['p90']:.3f} max {v['max']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-horizon data for Robin boundary control with box constraints u_a <= u <= u_b (target y_d = 1).

For a horizon T the reduced problem is the box-constrained QP
    min 1/2 u^T (H_T + alpha I) u - g_T^T u   s.t.  u_a <= u_k <= u_b,
with H_T, g_T as in gen_turnpike_data.py. With q := (g_T - H_T u) / alpha (= p(0, t_k) / alpha,
the discrete adjoint trace) optimality is exactly u = clip(q, u_a, u_b). Solved by the
primal-dual active set / semismooth Newton method; residual ||u - clip(q)|| / ||u|| is stored.

Bounds are relative to the unconstrained static optimum u_bar:
    u_a = theta_a u_bar, theta_a in [0, 0.6];   u_b = theta_b u_bar, theta_b in [0.7, 2].
Static constrained optimum (turnpike): u_c = clip(u_bar, u_a, u_b), and
    q_bar = psi^T (G - M_obs psi u_c) / alpha,   K_beta psi = e0.
Stored per T: U (constrained control), Q (adjoint trace / alpha), Uunc (unconstrained control).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators, impulse_response, interval_load  # noqa: E402

T_LIST = (1, 2, 4, 8)
STEPS = 320
COLS = ["w", "c_rel", "log10_alpha", "log10_beta", "theta_a", "theta_b"]
RANGES = {"w": (0.1, 0.5), "c_rel": (0.0, 1.0), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0),
          "theta_a": (0.0, 0.6), "theta_b": (0.7, 2.0)}
ONE = lambda x, t: np.ones_like(np.asarray(x, float))


def box_qp(H: np.ndarray, g: np.ndarray, alpha: float, lo: float, hi: float, max_iter: int = 80):
    """Primal-dual active set for u = clip((g - H u)/alpha, lo, hi)."""
    N = len(g)
    Ha = H + alpha * np.eye(N)
    u = np.clip(sla.solve(Ha, g, assume_a="pos"), lo, hi)
    prev = None
    it = 0
    for it in range(1, max_iter + 1):
        q = (g - H @ u) / alpha
        up, dn = q > hi, q < lo
        key = (up.tobytes(), dn.tobytes())
        if key == prev:
            break
        prev = key
        I = ~(up | dn)
        u = np.where(up, hi, np.where(dn, lo, 0.0))
        if I.any():
            A = ~I
            rhs = g[I] - (H[np.ix_(I, A)] @ u[A] if A.any() else 0.0)
            u[I] = sla.solve(Ha[np.ix_(I, I)], rhs, assume_a="pos")
    q = (g - H @ u) / alpha
    res = float(np.linalg.norm(u - np.clip(q, lo, hi)) / max(np.linalg.norm(u), 1e-300))
    return u, q, it, res


def fista_box(H, g, alpha, lo, hi, u0, tol=1e-14, max_iter=50000):
    """Accelerated projected gradient (FISTA, adaptive restart) for the same QP; H + alpha I is well conditioned."""
    v = np.random.default_rng(0).standard_normal(len(g))
    for _ in range(60):  # power iteration for lambda_max(H)
        v = H @ v
        v /= np.linalg.norm(v)
    L = 1.05 * float(v @ (H @ v)) + alpha
    u = np.clip(u0, lo, hi)
    yk, tk = u.copy(), 1.0
    for _ in range(max_iter):
        u_new = np.clip(yk - ((H @ yk) + alpha * yk - g) / L, lo, hi)
        if (u_new - u) @ (yk - u_new) > 0:  # restart
            tk = 1.0
            yk = u.copy()
            continue
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        yk = u_new + ((tk - 1.0) / t_new) * (u_new - u)
        if np.linalg.norm(u_new - u) <= tol * max(np.linalg.norm(u_new), 1e-300):
            u = u_new
            break
        u, tk = u_new, t_new
    return u


def polish(H, g, alpha, lo, hi, u):
    """One active-set solve on the active sets implied by u."""
    Ha = H + alpha * np.eye(len(g))
    q = (g - H @ u) / alpha
    up, dn = q > hi, q < lo
    I = ~(up | dn)
    v = np.where(up, hi, np.where(dn, lo, 0.0))
    if I.any():
        A = ~I
        v[I] = sla.solve(Ha[np.ix_(I, I)], g[I] - (H[np.ix_(I, A)] @ v[A] if A.any() else 0.0), assume_a="pos")
    return v


def fp_res(H, g, alpha, lo, hi, u):
    return float(np.linalg.norm(u - np.clip((g - H @ u) / alpha, lo, hi)) / max(np.linalg.norm(u), 1e-300))


def robust_box_qp(H, g, alpha, lo, hi):
    """PDAS; if it cycles, FISTA from the PDAS iterate, then polish. Returns the candidate with smallest residual."""
    u, q, it, res = box_qp(H, g, alpha, lo, hi)
    method = "pdas"
    if res > 1e-12:
        cands = [(res, u, "pdas")]
        uf = fista_box(H, g, alpha, lo, hi, u)
        cands.append((fp_res(H, g, alpha, lo, hi, uf), uf, "fista"))
        up = polish(H, g, alpha, lo, hi, uf)
        cands.append((fp_res(H, g, alpha, lo, hi, up), up, "fista+polish"))
        res, u, method = min(cands, key=lambda c: c[0])
        q = (g - H @ u) / alpha
    return u, q, it, res, method


def lbfgsb(H, g, alpha, lo, hi):
    Ha = H + alpha * np.eye(len(g))
    f = lambda u: (0.5 * u @ (Ha @ u) - g @ u, Ha @ u - g)
    r = minimize(f, np.clip(np.zeros_like(g), lo, hi), jac=True, method="L-BFGS-B",
                 bounds=[(lo, hi)] * len(g), options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    return r.x


def solve_family(row: np.ndarray, T_list=T_LIST, check: bool = False) -> dict:
    w, c_rel, la, lb, th_a, th_b = row
    c = w / 2 + c_rel * (1 - w)
    T_max = max(T_list)
    cfg = RobinConfig(alpha=10.0**la, beta=10.0**lb, xa=c - w / 2, xb=c + w / 2, T=float(T_max), Nt=STEPS * T_max)
    ops = assemble_operators(cfg)
    a = impulse_response(ops)
    G = interval_load(ONE, cfg, 0.0, cfg.xa, cfg.xb)
    R = a @ (ops.M_obs @ a.T)
    aG_cs = np.cumsum(a @ G)
    sizes = [STEPS * T for T in T_list]
    n_max = STEPS * T_max
    H = {N: np.zeros((N, N)) for N in sizes}
    for d in range(n_max):
        cs = np.cumsum(np.diagonal(R, offset=-d))
        for N in sizes:
            if N > d:
                j = np.arange(d, N)
                H[N][j - d, j] = cs[N - 1 - j]
                H[N][j, j - d] = cs[N - 1 - j]

    psi = spla.splu(ops.K_beta.tocsc()).solve(ops.e0)
    u_bar = float(psi @ G / (psi @ (ops.M_obs @ psi) + cfg.alpha))
    lo, hi = th_a * u_bar, th_b * u_bar
    u_c = float(np.clip(u_bar, lo, hi))
    q_bar = float(psi @ (G - ops.M_obs @ (psi * u_c)) / cfg.alpha)

    out = {"u_bar": u_bar, "u_c": u_c, "q_bar": q_bar, "lo": lo, "hi": hi}
    for T, N in zip(T_list, sizes):
        g = aG_cs[N - 1 - np.arange(N)]
        u_unc = np.linalg.solve(H[N] + cfg.alpha * np.eye(N), g)
        u, q, it, res, method = robust_box_qp(H[N], g, cfg.alpha, lo, hi)
        out[T] = {"U": u, "Q": q, "Uunc": u_unc, "iters": it, "res": res, "fallback": method != "pdas",
                  "frac_upper": float(np.mean(u >= hi - 1e-12 * u_bar)), "frac_lower": float(np.mean(u <= lo + 1e-12 * u_bar))}
        if check:
            ul = lbfgsb(H[N], g, cfg.alpha, lo, hi)
            obj = lambda v: 0.5 * v @ ((H[N] + cfg.alpha * np.eye(N)) @ v) - g @ v
            out[T]["check"] = {"rel_diff_vs_lbfgsb": float(np.linalg.norm(u - ul) / np.linalg.norm(u)),
                               "obj_pdas_minus_lbfgsb": float(obj(u) - obj(ul))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)

    if args.check:
        for row in X[:4]:
            fam = solve_family(row, T_list=(1, 2), check=True)
            for T in (1, 2):
                r = fam[T]
                print(json.dumps({"T": T, "theta_a": float(row[4]), "theta_b": float(row[5]), "iters": r["iters"], "res": r["res"],
                                  "frac_upper": r["frac_upper"], "frac_lower": r["frac_lower"], **r["check"]}))
        return

    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        fams = pool.map(solve_family, list(X), chunksize=2)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
    payload = {}
    for T in T_LIST:
        for key in ("U", "Q", "Uunc"):
            payload[f"{key}_T{T}"] = np.array([f[T][key] for f in fams])
        payload[f"iters_T{T}"] = np.array([f[T]["iters"] for f in fams])
        payload[f"res_T{T}"] = np.array([f[T]["res"] for f in fams])
        payload[f"fallback_T{T}"] = np.array([f[T]["fallback"] for f in fams])
        payload[f"frac_upper_T{T}"] = np.array([f[T]["frac_upper"] for f in fams])
        payload[f"frac_lower_T{T}"] = np.array([f[T]["frac_lower"] for f in fams])
    w, c_rel = X[:, 0], X[:, 1]
    cc = w / 2 + c_rel * (1 - w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, xa=cc - w / 2, xb=cc + w / 2, alpha=10.0 ** X[:, 2], beta=10.0 ** X[:, 3],
        u_bar=np.array([f["u_bar"] for f in fams]), u_c=np.array([f["u_c"] for f in fams]),
        q_bar=np.array([f["q_bar"] for f in fams]), lo=np.array([f["lo"] for f in fams]), hi=np.array([f["hi"] for f in fams]),
        train_idx=np.sort(perm[:n_tr]), val_idx=np.sort(perm[n_tr : n_tr + n_va]), test_idx=np.sort(perm[n_tr + n_va :]),
        dt=1.0 / STEPS, T_list=np.array(T_LIST),
        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "target": "y_d = 1",
                                  "bounds": "u_a = theta_a u_bar, u_b = theta_b u_bar", "seed": args.seed}),
        **payload,
    )
    res = max(float(payload[f"res_T{T}"].max()) for T in T_LIST)
    nfb = {T: int(payload[f"fallback_T{T}"].sum()) for T in T_LIST}
    print(f"{args.n} samples x {T_LIST} in {time.perf_counter() - t0:.0f}s, max fixed-point residual {res:.2e}, "
          f"FISTA fallback used per T {nfb} -> {args.out}")


if __name__ == "__main__":
    main()

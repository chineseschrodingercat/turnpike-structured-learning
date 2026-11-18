#!/usr/bin/env python3
"""Validation gates for robin_solver.py (E0).

G1  adjoint gradient vs centered finite differences of the discrete cost
G2  reduced Hessian: symmetric, SPD, and equal to gradient differences
G3  manufactured solution with an internal subzone: spatial and temporal rates
G4  beta = 0, omega = (0, L) regression against the package's ctrl1d/solver_p1.py
G5  reduced closed form (H + alpha I) u = g against the monolithic KKT solve
G6  fast H and g against the dense Toeplitz S; Toeplitz S against a forward solve
G7  interval mass on (0, L) against the analytic P1 mass matrix

Usage: python validate_robin.py --pkg <path to ctrl1d_p2 package> --out <json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robin_solver import (  # noqa: E402
    RobinConfig,
    assemble_operators,
    assemble_problem_data,
    dense_control_to_state,
    eval_cost,
    eval_gradient,
    reduced_control,
    reduced_system,
    solve_kkt,
    solve_state,
)


def const(value: float):
    return lambda x, t: np.full_like(np.asarray(x, dtype=float), value)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def gate_gradient(beta: float) -> dict:
    cfg = RobinConfig(alpha=1e-2, beta=beta, xa=0.31, xb=0.62, nx=40, Nt=80)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, const(1.0))
    rng = np.random.default_rng(1)
    u = rng.standard_normal(cfg.Nt)
    d = rng.standard_normal(cfg.Nt)

    def J(v):
        return eval_cost(cfg, ops, data, solve_state(ops, data, v), v)[0]

    eps = 1e-3
    fd = (J(u + eps * d) - J(u - eps * d)) / (2 * eps)
    an = float(eval_gradient(ops, data, u) @ d)
    return {"beta": beta, "fd": fd, "adjoint": an, "rel_err": abs(fd - an) / abs(an)}


def gate_hessian(beta: float) -> dict:
    cfg = RobinConfig(alpha=1e-2, beta=beta, xa=0.31, xb=0.62, nx=40, Nt=80)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, const(1.0))
    red = reduced_system(ops, data)
    Hred = cfg.dt * (red.H + cfg.alpha * np.eye(cfg.Nt))  # Euclidean reduced Hessian
    lam = np.linalg.eigvalsh(0.5 * (Hred + Hred.T))
    rng = np.random.default_rng(2)
    u = rng.standard_normal(cfg.Nt)
    d = rng.standard_normal(cfg.Nt)
    hv = eval_gradient(ops, data, u + d) - eval_gradient(ops, data, u)
    return {
        "beta": beta,
        "sym_err": float(np.linalg.norm(Hred - Hred.T) / np.linalg.norm(Hred)),
        "lambda_min": float(lam[0]),
        "lower_bound_dt_alpha": cfg.dt * cfg.alpha,
        "hessvec_rel_err": rel(Hred @ d, hv),
    }


def mms_fields(beta: float, kappa: float, alpha: float, T: float):
    a = 1.0 + beta / kappa  # phi = (1-x)(1+a x) has -kappa phi'(0) + beta phi(0) = 0
    dcoef = 1.0 / (kappa + beta)  # psi = d (1-x) has -kappa psi'(0) + beta psi(0) = 1

    phi = lambda x: (1.0 - x) * (1.0 + a * x)
    psi = lambda x: dcoef * (1.0 - x)
    u = lambda t: np.cos(np.pi * t / (2 * T)) / alpha
    du = lambda t: -(np.pi / (2 * T)) * np.sin(np.pi * t / (2 * T)) / alpha

    y = lambda x, t: np.sin(np.pi * t / T) * phi(x) + u(t) * psi(x)
    p = lambda x, t: np.cos(np.pi * t / (2 * T)) * phi(x)
    # f = y_t - kappa y_xx, with phi'' = -2a and psi'' = 0
    f = lambda x, t: (np.pi / T) * np.cos(np.pi * t / T) * phi(x) + du(t) * psi(x) + 2 * a * kappa * np.sin(np.pi * t / T)
    # y_d = y on omega, so the adjoint source is h = -p_t - kappa p_xx on (0, L)
    hsrc = lambda x, t: (np.pi / (2 * T)) * np.sin(np.pi * t / (2 * T)) * phi(x) + 2 * a * kappa * np.cos(np.pi * t / (2 * T))
    y0 = lambda x: y(x, 0.0)
    return y, p, u, f, hsrc, y0


def mms_errors(nx: int, Nt: int, beta: float = 1.0) -> dict:
    cfg = RobinConfig(alpha=0.1, beta=beta, xa=0.31, xb=0.62, nx=nx, Nt=Nt)
    y, p, u, f, hsrc, y0 = mms_fields(beta, cfg.kappa, cfg.alpha, cfg.T)
    res = solve_kkt(cfg, yd_func=y, f_func=f, y0_func=y0, h_func=hsrc)
    x = res.operators.x_free
    t = np.arange(1, Nt + 1) * cfg.dt
    M = res.operators.M

    def l2(field):
        return np.sqrt(cfg.dt * sum(float(v @ (M @ v)) for v in field))

    Ye = np.array([y(x, tk) for tk in t])
    Pe = np.array([p(x, tk) for tk in t])
    Ue = u(t)
    return {
        "nx": nx,
        "Nt": Nt,
        "y": l2(res.Y - Ye) / l2(Ye),
        "p": l2(res.P - Pe) / l2(Pe),
        "u": float(np.linalg.norm(res.U - Ue) / np.linalg.norm(Ue)),
        "kkt_rel_res": res.rel_res,
    }


def rates(rows: list[dict], key: str) -> dict:
    out = {}
    for var in ("y", "p", "u"):
        e = [r[var] for r in rows]
        out[var] = float(np.log(e[-2] / e[-1]) / np.log(2.0))
    return out


def gate_mms() -> dict:
    space = [mms_errors(nx, 2048) for nx in (10, 20, 40, 80)]
    time_ = [mms_errors(400, Nt) for Nt in (16, 32, 64, 128)]
    return {
        "space_sweep": space,
        "space_rates_finest_pair": rates(space, "nx"),
        "time_sweep": time_,
        "time_rates_finest_pair": rates(time_, "Nt"),
    }


def gate_mms_space_selfconv(Nt: int = 128, nx_ref: int = 1280) -> dict:
    """Spatial error at fixed Nt against a fine-space reference with the same Nt.

    The analytic MMS sweep above is dominated by the O(dt) time error once h is
    small; comparing against nx_ref at equal Nt isolates the spatial error.
    """
    beta = 1.0
    cfg_of = lambda nx: RobinConfig(alpha=0.1, beta=beta, xa=0.31, xb=0.62, nx=nx, Nt=Nt)
    y, p, u, f, hsrc, y0 = mms_fields(beta, 1.0, 0.1, 1.0)
    ref = solve_kkt(cfg_of(nx_ref), yd_func=y, f_func=f, y0_func=y0, h_func=hsrc)
    rows = []
    for nx in (10, 20, 40, 80, 160):
        res = solve_kkt(cfg_of(nx), yd_func=y, f_func=f, y0_func=y0, h_func=hsrc)
        stride = nx_ref // nx
        M = res.operators.M
        dt = res.cfg.dt

        def l2(field):
            return np.sqrt(dt * sum(float(v @ (M @ v)) for v in field))

        Yr, Pr = ref.Y[:, ::stride], ref.P[:, ::stride]
        rows.append({
            "nx": nx, "Nt": Nt,
            "y": l2(res.Y - Yr) / l2(Yr),
            "p": l2(res.P - Pr) / l2(Pr),
            "u": float(np.linalg.norm(res.U - ref.U) / np.linalg.norm(ref.U)),
        })
    return {"nx_ref": nx_ref, "sweep": rows, "rates_finest_pair": rates(rows, "nx"),
            "rates_all_pairs": {v: [float(np.log(rows[i][v] / rows[i + 1][v]) / np.log(2)) for i in range(len(rows) - 1)]
                                for v in ("y", "p", "u")}}


def gate_beta0_regression(pkg: pathlib.Path) -> dict:
    sys.path.insert(0, str(pkg))
    from ctrl1d.solver_p1 import SolverConfig as P1Config
    from ctrl1d.solver_p1 import solve_kkt as p1_solve

    alpha = 1e-2
    ref = p1_solve(P1Config(alpha=alpha, nx=160, Nt=320), f_func=const(0.0), yd_func=const(1.3))
    rob = solve_kkt(RobinConfig(alpha=alpha, beta=0.0, xa=0.0, xb=1.0, nx=160, Nt=320), yd_func=const(1.3))
    return {
        "U_robin_vs_minus_U_p1": rel(rob.U, -ref.U),
        "Y_rel_diff": rel(rob.Y, ref.Y),
        "P_rel_diff": rel(rob.P, ref.P),
        "J_total_rel_diff": abs(rob.J_total - ref.J_total) / abs(ref.J_total),
        "kkt_rel_res_robin": rob.rel_res,
        "kkt_rel_res_p1": ref.rel_res,
    }


def gate_reduced_vs_kkt() -> list[dict]:
    out = []
    for beta, alpha, xa, xb, yd in [
        (1.0, 1e-3, 0.37, 0.58, const(1.7)),
        (0.1, 1e-2, 0.05, 0.40, const(0.8)),
        (10.0, 1e-1, 0.55, 0.95, lambda x, t: np.full_like(np.asarray(x, float), 0.5 * np.sin(7.0 * t) + 1.25)),
    ]:
        cfg = RobinConfig(alpha=alpha, beta=beta, xa=xa, xb=xb)
        t0 = time.perf_counter()
        res = solve_kkt(cfg, yd_func=yd)
        t_kkt = time.perf_counter() - t0
        t0 = time.perf_counter()
        red = reduced_system(res.operators, res.data)
        u_red = reduced_control(red, alpha)
        t_red = time.perf_counter() - t0
        u_full_eig = reduced_control(red, alpha, rank=cfg.Nt)
        out.append({
            "beta": beta, "alpha": alpha, "xa": xa, "xb": xb,
            "rel_diff_reduced_vs_kkt": rel(u_red, res.U),
            "rel_diff_eig_fullrank_vs_kkt": rel(u_full_eig, res.U),
            "kkt_rel_res": res.rel_res,
            "seconds_kkt": t_kkt, "seconds_reduced": t_red,
        })
    return out


def gate_toeplitz() -> dict:
    cfg = RobinConfig(alpha=1e-2, beta=2.0, xa=0.2, xb=0.45, nx=30, Nt=50)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, const(1.0))
    red = reduced_system(ops, data)
    S = dense_control_to_state(ops, red.impulse)
    C = sp.kron(sp.eye(cfg.Nt), ops.M_obs).toarray()
    u = np.random.default_rng(3).standard_normal(cfg.Nt)
    return {
        "S_vs_forward_solve": rel(S @ u, solve_state(ops, data, u).ravel()),
        "H_fast_vs_dense": rel(red.H, S.T @ C @ S),
        "g_fast_vs_dense": rel(red.g, S.T @ np.concatenate(data.G_list)),
    }


def gate_mass() -> dict:
    cfg = RobinConfig(nx=160)
    ops = assemble_operators(cfg)
    h = cfg.h
    diag = np.full(cfg.nx, 2 * h / 3)
    diag[0] = h / 3
    off = np.full(cfg.nx - 1, h / 6)
    M_ref = sp.diags([off, diag, off], [-1, 0, 1]).toarray()
    sub = assemble_operators(RobinConfig(nx=160, xa=0.3, xb=0.6)).M_obs
    return {
        "M_vs_analytic": rel(ops.M.toarray(), M_ref),
        "M_obs_total_mass_vs_width": abs(float(np.ones(cfg.nx) @ (sub @ np.ones(cfg.nx))) - 0.3) / 0.3,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--gates", nargs="*", default=None, help="subset of gate names to run (default: all)")
    args = ap.parse_args()

    report = json.loads(args.out.read_text(encoding="utf-8")) if (args.gates and args.out.exists()) else {}
    for name, fn in [
        ("G7_mass", gate_mass),
        ("G1_gradient", lambda: [gate_gradient(b) for b in (0.0, 1.0, 10.0)]),
        ("G2_hessian", lambda: [gate_hessian(b) for b in (0.0, 1.0, 10.0)]),
        ("G6_toeplitz", gate_toeplitz),
        ("G4_beta0_regression", lambda: gate_beta0_regression(args.pkg)),
        ("G5_reduced_vs_kkt", gate_reduced_vs_kkt),
        ("G3_mms", gate_mms),
        ("G3b_mms_space_selfconv", gate_mms_space_selfconv),
    ]:
        if args.gates and name not in args.gates:
            continue
        t0 = time.perf_counter()
        report[name] = fn()
        print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
        print(json.dumps(report[name], indent=1), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Closed-loop receding-horizon control (MPC) of the semilinear plant y_t - y_xx + gamma y^3 = 0 (T_ref = 1 units).

Plant: semilinear1d dynamics with the scenario's mu, y(0) = 0; at t = 2, 4, 6 the state is kicked by a random smooth
field (same law as the y0 disturbances in training). Every delta (= 32 steps of dt = 1/320) the controller sees the
current state and returns a control over its prediction horizon T_p; the first delta of it is applied.
Controllers
    exact_Tp    Newton-KKT over [0, T_p] from the current state (warm start = previous solution shifted by delta)
    tp_Tp       TurnpikeNet-y0 (trained on T = 1 only): static OCP once per mu, then one network query per step
    lq_Tp       exact MPC with the gamma = 0 model (fast linear solve; model mismatch)
    openloop    exact optimum over [0, T_sim] computed once at t = 0 (no feedback; kicks unknown)
    const       u = u_bar
Closed-loop cost: the dt-weighted tracking + control cost of the realized trajectory on [0, T_sim].
Optional unmodeled persistent disturbance (--sink q): the plant has a uniform heat loss f = -q for t >= --sink-start,
which no controller's model contains (only feedback through the measured state can react to it).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators  # noqa: E402
from semilinear1d import build, cost, forward, solve_kkt, static_ocp_sparse  # noqa: E402
from y0_family import AMP, K_SAMPLE, Projector, basis  # noqa: E402

DT, DELTA_STEPS, T_SIM = 1.0 / 320, 32, 8.0
KICKS = (2.0, 4.0, 6.0)
SINK, SINK_START = 0.0, 1.0  # unmodeled uniform heat loss f = -SINK for t >= SINK_START (plant only)


def plant_forward(s, U, F):
    """semilinear1d.forward with an extra source load F (the controllers' models do not know it)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    Y, y = np.zeros((len(U), s.A.shape[0])), s.y0.copy()
    for k in range(len(U)):
        rhs = s.Mdt @ y + s.e0 * U[k] + F
        z = y.copy()
        for _ in range(50):
            Fz = s.A @ z + s.gamma * s.ML * z**3 - rhs
            if np.linalg.norm(Fz) <= 1e-13 * max(np.linalg.norm(rhs), 1e-300):
                break
            z = z - spla.spsolve((s.A + sp.diags(3 * s.gamma * s.ML * z**2)).tocsc(), Fz)
        y = z
        Y[k] = y
    return Y


def load_surrogate(path):
    import torch

    from y0_turnpike_surrogate import Y0Turnpike

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mode = {"tp_lin": "lin", "tp_hyb": "hyb", "tp_mlp": "mlp", "tp_noz": "noz"}[ck["model"]]
    net = Y0Turnpike(ck["n_mu"], ck["n_z"], r=ck["r"], mode=mode)
    net.load_state_dict({k: v.double() for k, v in ck["state"].items()})
    net.eval()
    return net, ck["norm"], ck.get("floor_init", "lam1")


def run(task):
    sc, ctrl, Tp, model_path, dist = task
    global KICKS, SINK, SINK_START
    KICKS, SINK, SINK_START = tuple(dist["kicks"]), dist["sink"], dist["sink_start"]
    geo, gamma, mu = sc["geo"], sc["gamma"], np.asarray(sc["mu"])
    cfg_d = RobinConfig(nx=160, Nt=DELTA_STEPS, T=DELTA_STEPS * DT, **geo)
    x = np.arange(160) / 160
    setup_start = time.perf_counter()
    st = static_ocp_sparse(build(RobinConfig(nx=160, Nt=320, **geo), gamma, 1.0))
    reference_seconds = time.perf_counter() - setup_start
    ub, ybar, lam1 = st["u_bar"], st["y_bar"], st["lam1_lin"]
    kicks = [basis(x, K_SAMPLE) @ np.random.default_rng([sc["seed"], j]).normal(0.0, AMP / np.arange(1, K_SAMPLE + 1)) for j in range(len(KICKS))]
    n_steps = int(round(T_SIM / (DELTA_STEPS * DT)))
    solve_secs, iters, solver_residuals = [], [], []
    state = {"warm": None}

    if ctrl.startswith("tp"):
        import torch

        from turnpike_surrogate import first_eig, grid

        net, norm, floor_init = load_surrogate(model_path)
        proj = Projector(x, assemble_operators(RobinConfig(nx=160, **geo)).M)
        xm = torch.tensor(((mu - norm["x_mean"]) / norm["x_std"])[None, :])
        t_g, s_g = grid(Tp, DT)
        fl = torch.tensor([[first_eig(geo["beta"]), lam1]]) if floor_init == "open" else torch.tensor([lam1])

        def controller(y, k):
            z = torch.tensor((proj(y - ybar) / norm["z_scale"])[None, :])
            with torch.no_grad():
                return ub * net(xm, z, t_g, s_g, fl)[0].numpy()
    elif ctrl.startswith("exact") or ctrl.startswith("lq"):
        g_model = gamma if ctrl.startswith("exact") else 0.0
        cfg_p = RobinConfig(nx=160, Nt=int(round(Tp / DT)), T=float(Tp), **geo)

        def controller(y, k):
            init = None
            if state["warm"] is not None and g_model > 0:
                Y, P, U = state["warm"]
                sh = DELTA_STEPS
                init = (np.vstack([Y[sh:], np.repeat(Y[-1:], sh, 0)]), np.vstack([P[sh:], np.repeat(P[-1:], sh, 0)]),
                        np.concatenate([U[sh:], np.repeat(U[-1:], sh)]))
            Y, P, U, info = solve_kkt(build(cfg_p, g_model, 1.0, y), init=init)
            state["warm"] = (Y, P, U)
            iters.append(info["newton_iters"])
            solver_residuals.append(float(info["res_rel"]))
            return U
    elif ctrl == "openloop":
        U_ol = solve_kkt(build(RobinConfig(nx=160, Nt=int(T_SIM / DT), T=T_SIM, **geo), gamma, 1.0))[2]

        def controller(y, k):
            return U_ol[k * DELTA_STEPS : (k + 1) * DELTA_STEPS]
    else:

        def controller(y, k):
            return np.full(DELTA_STEPS, ub)

    y = np.zeros(160)
    J, u_hist, ymean = 0.0, [], []
    mask = (x >= geo["xa"]) & (x <= geo["xb"])
    setup_seconds = time.perf_counter() - setup_start
    for k in range(n_steps):
        t0 = time.perf_counter()
        u = np.asarray(controller(y, k))[:DELTA_STEPS]
        solve_secs.append(time.perf_counter() - t0)
        if (k + 1) % 10 == 0:
            print(f"step {k+1}/{n_steps}: controller {solve_secs[-1]:.6f} s", flush=True)
        sp_ = build(cfg_d, gamma, 1.0, y)
        t_start = k * DELTA_STEPS * DT
        Y = plant_forward(sp_, u, -SINK * sp_.ML) if (SINK > 0 and t_start >= SINK_START - 1e-9) else forward(sp_, u)
        J += cost(sp_, Y, u)
        u_hist.append(u)
        ymean.append(Y[:, mask].mean(1))
        y = Y[-1].copy()
        t_now = (k + 1) * DELTA_STEPS * DT
        for j, tk in enumerate(KICKS):
            if abs(t_now - tk) < 1e-9:
                y = y + kicks[j]
    return {"controller_seconds": solve_secs, "solver_residuals": solver_residuals,
            "reference_seconds": reference_seconds, "setup_seconds": setup_seconds,
            "initialization_to_first_control_seconds": setup_seconds + solve_secs[0],
            "scenario": sc["idx"], "ctrl": ctrl, "Tp": Tp, "J": J, "solve_s_median": float(np.median(solve_secs)),
            "solve_s_total": float(np.sum(solve_secs)), "newton_iters_median": float(np.median(iters)) if iters else None,
            "u": np.concatenate(u_hist).tolist(), "ymean": np.concatenate(ymean).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True, help="semilinear y0 dataset (scenario parameters)")
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--n-scen", type=int, default=12)
    ap.add_argument("--ctrls", nargs="+", default=["exact:1", "exact:4", "tp:1", "tp:4", "lq:1", "openloop:0", "const:0"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--kicks", type=float, nargs="*", default=[2.0, 4.0, 6.0])
    ap.add_argument("--sink", type=float, default=0.0, help="unmodeled uniform heat loss (plant only), active for t >= --sink-start")
    ap.add_argument("--sink-start", type=float, default=1.0)
    args = ap.parse_args()
    dist = {"kicks": args.kicks, "sink": args.sink, "sink_start": args.sink_start}
    d = np.load(args.data)
    te = d["test_idx"]
    ge = d["X"][te, 4]
    rows = [int(te[np.argmin(np.abs(ge - q))]) for q in np.quantile(ge, np.linspace(0.04, 0.96, args.n_scen))]
    scen = [{"idx": i, "seed": 1000 + i, "gamma": float(d["gamma"][i]), "mu": d["X"][i].tolist(),
             "geo": {k: float(d[k][i]) for k in ("alpha", "beta", "xa", "xb")}} for i in rows]
    tasks = []
    for c in args.ctrls:
        name, Tp = c.split(":")
        for sc in scen:
            tasks.append((sc, f"{name}_{Tp}" if name in ("exact", "tp", "lq") else name, int(Tp), str(args.model), dist))
    tasks.sort(key=lambda t: -(t[2] if t[1].startswith("exact") else 0))  # long exact runs first
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(run, tasks, chunksize=1)
    ctrls = sorted({r["ctrl"] for r in res})
    ref = {r["scenario"]: r["J"] for r in res if r["ctrl"] == "exact_4"} or {r["scenario"]: r["J"] for r in res if r["ctrl"] == "exact_1"}
    summ = {}
    for c in ctrls:
        rr = [r for r in res if r["ctrl"] == c]
        rel = np.array([(r["J"] - ref[r["scenario"]]) / ref[r["scenario"]] for r in rr])
        summ[c] = {"rel_cost_vs_exact_median": float(np.median(rel)), "rel_cost_vs_exact_max": float(np.max(rel)),
                   "rel_cost_vs_exact_min": float(np.min(rel)), "solve_s_per_step_median": float(np.median([r["solve_s_median"] for r in rr]))}
        print(f"{c:10s} cost vs exact: median {summ[c]['rel_cost_vs_exact_median']:+.2e} max {summ[c]['rel_cost_vs_exact_max']:+.2e}  "
              f"solve/step {summ[c]['solve_s_per_step_median']:.2e}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scenarios": scen, "summary": summ, "disturbance": dist, "reference": "exact_4" if any(r["ctrl"] == "exact_4" for r in res) else "exact_1",
                                    "seconds": time.perf_counter() - t0, "runs": res}), encoding="utf-8")
    print(f"wrote {args.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()

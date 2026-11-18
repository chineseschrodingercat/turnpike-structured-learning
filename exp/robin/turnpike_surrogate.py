#!/usr/bin/env python3
"""Turnpike-structured surrogate vs baselines: train on short horizons, test on T = 1, 2, 4, 8.

Model (TurnpikeNet), with u_bar(mu) the closed-form static optimal control:
    u(t; mu, T) / u_bar = 1 + sum_i a_i(mu) exp(-lam_i(mu) t) + sum_i b_i(mu) exp(-kap_i(mu) (T - t))
An encoder MLP of mu = (w, c_rel, log10 alpha, log10 beta) outputs rates (> lam_min)
and amplitudes. T enters only through T - t, so a model trained on T = 1 can be
evaluated at any horizon.

Baselines
    DeepONet   branch(mu) . trunk(t, T - t) (+ bias), same u_bar scaling, same training data
    VarPro fit per-test-sample fit of the same exponential form to its own T=1 trajectory
               (no learning across mu; isolates the error of the functional form)
    stretch    oracle u_1(t / T) (what any time-normalized surrogate can at best do)
    plateau    u = u_bar (no layers)
Model selection uses validation error on the training horizons only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def first_eig(beta: float) -> float:
    """Smallest generalized eigenvalue of (K_beta, M): the slowest open-loop decay rate."""
    import sys

    import scipy.linalg as sla

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from robin_solver import RobinConfig, assemble_operators

    ops = assemble_operators(RobinConfig(beta=beta))
    return float(sla.eigh(ops.K_beta.toarray(), ops.M.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0])


def load(path: pathlib.Path) -> dict:
    d = np.load(path)
    T_list = [int(T) for T in d["T_list"]]
    X = d["X"]
    mean, std = X[d["train_idx"]].mean(0), X[d["train_idx"]].std(0)
    return {
        "floor": torch.tensor([first_eig(b) for b in d["beta"]]),
        "Xn": torch.tensor((X - mean) / std),
        "ubar": d["u_bar"],
        "dt": float(d["dt"]),
        "T_list": T_list,
        "Y": {T: torch.tensor(d[f"U_T{T}"] / d["u_bar"][:, None]) for T in T_list},  # u / u_bar
        "alpha": d["alpha"],
        "xa": d["xa"],
        "tr": torch.tensor(d["train_idx"]),
        "va": torch.tensor(d["val_idx"]),
        "te": torch.tensor(d["test_idx"]),
    }


def grid(T: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    N = int(round(T / dt))
    t = torch.arange(1, N + 1) * dt
    return t, T - t


def mlp(i: int, width: int, o: int, act=nn.GELU) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, width), act(), nn.Linear(width, width), act(), nn.Linear(width, o))


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


class TurnpikeNet(nn.Module):
    """use_floor: rates = lambda_1(beta) + softplus(.) (the slowest open-loop decay rate as a lower bound)."""

    def __init__(self, r: int = 8, width: int = 128, shared: bool = False, use_floor: bool = False,
                 lam_min: float = 0.5, lam_lo: float = 2.0, lam_hi: float = 400.0, n_in: int = 4):
        super().__init__()
        self.r, self.shared, self.use_floor, self.lam_min = r, shared, use_floor, lam_min
        self.enc = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        n_rates = r if shared else 2 * r
        self.rate_head = nn.Linear(width, n_rates)
        self.amp_head = nn.Linear(width, 2 * r)
        lam0 = torch.logspace(math.log10(lam_lo), math.log10(lam_hi), r)
        offset = 3.0 if use_floor else lam_min  # typical lambda_1(beta) is 2.7..8.2
        with torch.no_grad():
            self.rate_head.weight.mul_(0.05)
            self.rate_head.bias.copy_(inv_softplus(torch.clamp(lam0 - offset, min=0.5)).repeat(1 if shared else 2))
            self.amp_head.weight.mul_(0.05)
            self.amp_head.bias.zero_()

    def params(self, x: torch.Tensor, fl: torch.Tensor | None = None):
        h = self.enc(x)
        base = self.lam_min
        if self.use_floor and fl is not None:
            # fl (B,): one floor for all rates; fl (B, 2): separate floors for initial / terminal rates (needs shared=False)
            assert fl.dim() == 1 or not self.shared
            base = fl[:, None] if fl.dim() == 1 else torch.repeat_interleave(fl, self.r, dim=1)
        rates = base + nn.functional.softplus(self.rate_head(h))
        lam = rates[:, : self.r]
        kap = lam if self.shared else rates[:, self.r :]
        a, b = self.amp_head(h).chunk(2, dim=-1)
        return lam, kap, a, b

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, fl: torch.Tensor | None = None) -> torch.Tensor:
        lam, kap, a, b = self.params(x, fl)
        L0 = torch.einsum("bi,bin->bn", a, torch.exp(-lam[:, :, None] * t[None, None, :]))
        L1 = torch.einsum("bi,bin->bn", b, torch.exp(-kap[:, :, None] * s[None, None, :]))
        return 1.0 + L0 + L1


class DeepONetT(nn.Module):
    def __init__(self, p: int = 16, width: int = 128):
        super().__init__()
        self.branch = mlp(4, width, p)
        self.trunk = mlp(2, width, p, act=nn.Tanh)
        self.bias = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, fl: torch.Tensor | None = None) -> torch.Tensor:
        tau = self.trunk(torch.stack([t, s], dim=-1))  # (N, p)
        return self.branch(x) @ tau.T + self.bias


def rel_err(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - y, dim=1) / torch.linalg.norm(y, dim=1)


def train(model: nn.Module, data: dict, train_T: list[int], epochs: int, lr: float, log_every: int = 500) -> tuple[nn.Module, list]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-3)
    grids = {T: grid(T, data["dt"]) for T in train_T}
    tr, va = data["tr"], data["va"]
    fl = data["floor"]
    best, best_state, hist = float("inf"), None, []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = sum((rel_err(model(data["Xn"][tr], *grids[T], fl[tr]), data["Y"][T][tr]) ** 2).mean() for T in train_T) / len(train_T)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v = float(np.mean([float(rel_err(model(data["Xn"][va], *grids[T], fl[va]), data["Y"][T][va]).median()) for T in train_T]))
            hist.append((ep, float(loss.detach()), v))
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
            if ep % log_every == 0:
                print(f"  ep {ep:5d} loss {float(loss):.3e} val median rel {v:.3e}", flush=True)
    model.load_state_dict(best_state)
    return model, hist


def varpro_fit(Y1: torch.Tensor, dt: float, r: int, iters: int = 3000, lr: float = 0.02):
    """Per-sample fit of u/u_bar - 1 = sum a_i e^{-lam_i t} + sum b_i e^{-kap_i (1-t)} (rates by Adam, amplitudes by LS)."""
    t, s = grid(1, dt)
    target = Y1 - 1.0
    B = Y1.shape[0]
    lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
    z = nn.Parameter(torch.log(lam0).repeat(B, 2))
    opt = torch.optim.Adam([z], lr=lr)

    def design(zz, tt, ss):
        rates = torch.exp(zz)
        return torch.cat([torch.exp(-rates[:, :r, None] * tt[None, None, :]),
                          torch.exp(-rates[:, r:, None] * ss[None, None, :])], dim=1).transpose(1, 2)  # (B, N, 2r)

    def solve(Phi):
        G = Phi.transpose(1, 2) @ Phi
        ridge = 1e-12 * torch.diagonal(G, dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(2 * r)
        return torch.linalg.solve(G + ridge, Phi.transpose(1, 2) @ target[:, :, None])  # (B, 2r, 1)

    for _ in range(iters):
        opt.zero_grad()
        Phi = design(z, t, s)
        c = solve(Phi)
        loss = ((Phi @ c).squeeze(-1) - target).pow(2).sum(1).div(Y1.pow(2).sum(1)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        c = solve(design(z, t, s))

    def predict(T: float) -> torch.Tensor:
        tt, ss = grid(T, dt)
        with torch.no_grad():
            return 1.0 + (design(z, tt, ss) @ c).squeeze(-1)

    return predict


def stretch_predict(Y1: np.ndarray, T: float, dt: float) -> np.ndarray:
    t1 = np.arange(1, Y1.shape[1] + 1) * dt
    tT = np.arange(1, int(round(T / dt)) + 1) * dt
    return np.array([np.interp(tT / T, t1, y) for y in Y1])


def summarize(e: np.ndarray) -> dict:
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(np.max(e))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=["turnpike_net_T1", "turnpike_net_shared_T1", "turnpike_net_T1T2", "deeponet_T1"])
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load(args.data)
    te = data["te"]
    T_eval = data["T_list"]
    Yte = {T: data["Y"][T][te] for T in T_eval}
    results: dict[str, dict] = {}
    preds_T4: dict[str, np.ndarray] = {}

    def evaluate(name: str, predict) -> None:
        res = {}
        for T in T_eval:
            pred = predict(T)
            pred = pred if isinstance(pred, torch.Tensor) else torch.tensor(pred)
            res[str(T)] = rel_err(pred, Yte[T]).numpy()
            if T == 4:
                preds_T4[name] = pred.numpy()
        results.setdefault(name, {"per_seed": []})["per_seed"].append({k: summarize(v) for k, v in res.items()})
        results[name].setdefault("_errors", []).append(res)
        print(f"{name}: " + "  ".join(f"T={T}: med {np.median(res[str(T)]):.2e} max {np.max(res[str(T)]):.2e}" for T in T_eval), flush=True)

    if not args.no_baselines:
        evaluate("plateau_only", lambda T: torch.ones_like(Yte[T]))
        evaluate("stretch_oracle_T1", lambda T: stretch_predict(Yte[1].numpy(), T, data["dt"]))
        t0 = time.perf_counter()
        vp = varpro_fit(Yte[1], data["dt"], args.rank)
        evaluate("varpro_fit_T1_per_sample", vp)
        print(f"  varpro {time.perf_counter() - t0:.0f}s", flush=True)

    registry = {
        "turnpike_net_T1": (lambda: TurnpikeNet(r=args.rank), [1]),
        "turnpike_net_shared_T1": (lambda: TurnpikeNet(r=args.rank, shared=True), [1]),
        "turnpike_net_T1T2": (lambda: TurnpikeNet(r=args.rank), [1, 2]),
        "deeponet_T1": (lambda: DeepONetT(), [1]),
        "tp_r4_shared_floor_T1": (lambda: TurnpikeNet(r=4, shared=True, use_floor=True), [1]),
        "tp_r4_sep_floor_T1": (lambda: TurnpikeNet(r=4, shared=False, use_floor=True), [1]),
        "tp_r4_shared_floor_w256_T1": (lambda: TurnpikeNet(r=4, width=256, shared=True, use_floor=True), [1]),
        "tp_r4_shared_floor_T1T2": (lambda: TurnpikeNet(r=4, shared=True, use_floor=True), [1, 2]),
    }
    configs = [(name, *registry[name]) for name in args.configs]
    fl_te = data["floor"][te]
    for name, make, train_T in configs:
        for seed in args.seeds:
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            print(f"{name} seed {seed} (train T={train_T})", flush=True)
            model, hist = train(make(), data, train_T, args.epochs, args.lr)
            model.eval()

            def predict(T, model=model):
                with torch.no_grad():
                    return model(data["Xn"][te], *grid(T, data["dt"]), fl_te)

            evaluate(name, predict)
            print(f"  {time.perf_counter() - t0:.0f}s", flush=True)

    summary = {}
    for name, r in results.items():
        errs = r["_errors"]
        summary[name] = {
            "per_seed": r["per_seed"],
            "mean_over_seeds": {str(T): {k: float(np.mean([s[str(T)][k] for s in r["per_seed"]])) for k in ("median", "p90", "max")}
                                for T in T_eval},
        }
        if name == "turnpike_net_T1":
            e8 = errs[0]["8"]
            summary[name]["spearman_err_T8_vs_log_alpha"] = float(np.corrcoef(np.argsort(np.argsort(e8)),
                                                                              np.argsort(np.argsort(np.log10(data["alpha"][te.numpy()]))))[0, 1])
    (args.out_dir / "turnpike_surrogate_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    for name in summary:
        med = [summary[name]["mean_over_seeds"][str(T)]["median"] for T in T_eval]
        ax.semilogy(T_eval, med, marker="o", label=name)
    ax.set_xscale("log", base=2); ax.set_xticks(T_eval, [str(T) for T in T_eval])
    ax.set_xlabel("horizon T (training used T=1 unless noted)"); ax.set_ylabel("test median rel. L2")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("extrapolation in the horizon")
    ax = axes[1]
    for name in summary:
        mx = [summary[name]["mean_over_seeds"][str(T)]["max"] for T in T_eval]
        ax.semilogy(T_eval, mx, marker="o", label=name)
    ax.set_xscale("log", base=2); ax.set_xticks(T_eval, [str(T) for T in T_eval])
    ax.set_xlabel("horizon T"); ax.set_ylabel("test max rel. L2"); ax.grid(alpha=0.3); ax.set_title("worst test sample")
    ax = axes[2]
    ref = args.configs[0]
    i = int(np.argsort(results[ref]["_errors"][0]["4"])[len(te) // 2])
    tt = np.arange(1, 4 * 320 + 1) / 320
    ax.plot(tt, Yte[4][i].numpy(), "k", lw=2.5, label="truth (KKT), T=4")
    for name in [ref] + [n for n in ("deeponet_T1", "stretch_oracle_T1") if n in preds_T4 and n != ref]:
        ax.plot(tt, preds_T4[name][i], label=name)
    ax.set_xlabel("t"); ax.set_ylabel("u / u_bar"); ax.set_title("median test sample at T=4 (models saw only T=1)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "turnpike_surrogate.png", dpi=140)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()

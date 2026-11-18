#!/usr/bin/env python3
"""Turnpike surrogate for time-varying targets y_d = 1.25 + A sin(omega t): train on T = 1, test T = 1..8.

Linearity: u = 1.25 * u_bar * Vc + A * Im(u_hat * Vw), where
    Vc(t)  = 1 + sum_i a_i e^{-lam_i t} + sum_i b_i e^{-kap_i (T-t)}                       (target 1)
    Vw(t)  = e^{i w t} + sum_i alpha_i e^{-lam_i t} + e^{i w T} sum_i beta_i e^{-kap_i (T-t)}   (target e^{i w t})
u_bar, u_hat(w) are closed-form (static / periodic optimal controls). Rates lam, kap
depend on the system only (mu_g) and are shared by both components, with the floor
lambda_1(beta). a, b are real functions of mu_g; alpha, beta are complex functions of (mu_g, w).
The e^{i w T} factor on the terminal layer is the phase of the periodic turnpike at t = T.

Configs
    tp_osc        trained on targets 1, sin, cos at T = 1 (complex Vw observed)
    tp_osc_sin    ablation: trained on targets 1, sin only (only Im(u_hat Vw) observed)
    deeponet_osc  branch(mu_g, w, A) . trunk(t, T-t), trained on the real T=1 controls
Baselines: closed-form periodic turnpike without layers; stretched T=1 truth (oracle).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from turnpike_surrogate import first_eig, grid, inv_softplus, mlp, rel_err, stretch_predict, summarize  # noqa: E402

torch.set_default_dtype(torch.float64)
C_OFF = 1.25


def load(path: pathlib.Path) -> dict:
    d = np.load(path)
    X = d["X"]
    tr = d["train_idx"]
    std = lambda Z: torch.tensor((Z - Z[tr].mean(0)) / Z[tr].std(0))
    uh = d["u_hat"]
    data = {
        "xg": std(X[:, :4]), "xo": std(X[:, :5]), "xd": std(X[:, :6]),
        "omega": torch.tensor(d["omega"]), "A": torch.tensor(d["A"]),
        "ubar": torch.tensor(d["u_bar"]), "uhr": torch.tensor(uh.real), "uhi": torch.tensor(uh.imag),
        "floor": torch.tensor([first_eig(b) for b in d["beta"]]),
        "dt": float(d["dt"]), "T_list": [int(T) for T in d["T_list"]],
        "tr": torch.tensor(tr), "va": torch.tensor(d["val_idx"]), "te": torch.tensor(d["test_idx"]),
        "alpha": d["alpha"], "xa": d["xa"],
    }
    for T in data["T_list"]:
        data[f"c{T}"] = torch.tensor(d[f"U_const_T{T}"])
        data[f"s{T}"] = torch.tensor(d[f"U_sin_T{T}"])
        data[f"k{T}"] = torch.tensor(d[f"U_cos_T{T}"])
        data[f"u{T}"] = C_OFF * data[f"c{T}"] + data["A"][:, None] * data[f"s{T}"]
    return data


class OscTurnpikeNet(nn.Module):
    def __init__(self, r: int = 4, width: int = 128):
        super().__init__()
        self.r = r
        self.rate_net = mlp(4, width, 2 * r)
        self.amp_const = mlp(4, width, 2 * r)
        self.amp_osc = mlp(5, width, 4 * r)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            for net, bias in ((self.rate_net, inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)).repeat(2)),
                              (self.amp_const, torch.zeros(2 * r)), (self.amp_osc, torch.zeros(4 * r))):
                net[-1].weight.mul_(0.05)
                net[-1].bias.copy_(bias)

    def forward(self, data: dict, idx: torch.Tensor, T: float):
        t, s = grid(T, data["dt"])
        r = self.r
        rates = data["floor"][idx, None] + nn.functional.softplus(self.rate_net(data["xg"][idx]))
        E = torch.exp(-rates[:, :r, None] * t[None, None, :])   # initial-layer modes
        F = torch.exp(-rates[:, r:, None] * s[None, None, :])   # terminal-layer modes
        ab = self.amp_const(data["xg"][idx])
        Vc = 1.0 + torch.einsum("bi,bin->bn", ab[:, :r], E) + torch.einsum("bi,bin->bn", ab[:, r:], F)
        q = self.amp_osc(data["xo"][idx])
        ar, ai, br, bi = q[:, :r], q[:, r : 2 * r], q[:, 2 * r : 3 * r], q[:, 3 * r :]
        w = data["omega"][idx, None]
        Br, Bi = torch.einsum("bi,bin->bn", br, F), torch.einsum("bi,bin->bn", bi, F)
        cT, sT = torch.cos(w * T), torch.sin(w * T)
        Vr = torch.cos(w * t) + torch.einsum("bi,bin->bn", ar, E) + cT * Br - sT * Bi
        Vi = torch.sin(w * t) + torch.einsum("bi,bin->bn", ai, E) + sT * Br + cT * Bi
        uhr, uhi = data["uhr"][idx, None], data["uhi"][idx, None]
        Ur, Ui = uhr * Vr - uhi * Vi, uhr * Vi + uhi * Vr      # physical V = U_cos + i U_sin
        u_const = data["ubar"][idx, None] * Vc
        return u_const, Ur, Ui

    def predict(self, data, idx, T):
        u_const, _, Ui = self(data, idx, T)
        return C_OFF * u_const + data["A"][idx, None] * Ui


class DeepONetOsc(nn.Module):
    def __init__(self, p: int = 32, width: int = 128):
        super().__init__()
        self.branch = mlp(6, width, p)
        self.trunk = mlp(2, width, p, act=nn.Tanh)
        self.bias = nn.Parameter(torch.ones(()))

    def predict(self, data, idx, T):
        t, s = grid(T, data["dt"])
        y = self.branch(data["xd"][idx]) @ self.trunk(torch.stack([t, s], -1)).T + self.bias
        return C_OFF * data["ubar"][idx, None] * y


def loss_fn(model, data, idx, mode: str) -> torch.Tensor:
    if mode == "deeponet":
        return (rel_err(model.predict(data, idx, 1), data["u1"][idx]) ** 2).mean()
    u_const, Ur, Ui = model(data, idx, 1)
    L = (rel_err(u_const, data["c1"][idx]) ** 2).mean()
    if mode == "complex":
        num = (Ur - data["k1"][idx]).pow(2).sum(1) + (Ui - data["s1"][idx]).pow(2).sum(1)
        den = data["k1"][idx].pow(2).sum(1) + data["s1"][idx].pow(2).sum(1)
        return L + (num / den).mean()
    return L + (rel_err(Ui, data["s1"][idx]) ** 2).mean()  # sin-only


def train(model, data, mode, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-3)
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model, data, data["tr"], mode)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 25 == 0:
            with torch.no_grad():
                v = float(rel_err(model.predict(data, data["va"], 1), data["u1"][data["va"]]).median())
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
        if ep % 1000 == 0:
            print(f"  ep {ep} loss {float(loss.detach()):.3e} val median (T=1) {v:.3e}", flush=True)
    model.load_state_dict(best_state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=["tp_osc", "tp_osc_w256", "tp_osc_sin", "deeponet_osc"])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load(args.data)
    te, Ts = data["te"], data["T_list"]
    results, examples = {}, {}

    def record(name, pred_fn, components=None):
        res = {}
        for T in Ts:
            pred = pred_fn(T)
            res[str(T)] = rel_err(pred, data[f"u{T}"][te]).numpy()
            if T == 4:
                examples[name] = pred.numpy()
        entry = {k: summarize(v) for k, v in res.items()}
        if components is not None:
            entry["components"] = components
        results.setdefault(name, []).append(entry)
        print(f"{name}: " + "  ".join(f"T={T}: med {np.median(res[str(T)]):.2e} max {np.max(res[str(T)]):.2e}" for T in Ts), flush=True)
        return res

    def periodic_closed_form(T):
        t, _ = grid(T, data["dt"])
        w = data["omega"][te, None]
        return C_OFF * data["ubar"][te, None] + data["A"][te, None] * (data["uhr"][te, None] * torch.sin(w * t) + data["uhi"][te, None] * torch.cos(w * t))

    record("closed_form_periodic_turnpike_no_layers", periodic_closed_form)
    record("stretch_oracle_T1", lambda T: torch.tensor(stretch_predict(data["u1"][te].numpy(), T, data["dt"])))

    registry = {
        "tp_osc": (lambda: OscTurnpikeNet(), "complex"),
        "tp_osc_w256": (lambda: OscTurnpikeNet(width=256), "complex"),
        "tp_osc_sin": (lambda: OscTurnpikeNet(), "sin"),
        "deeponet_osc": (lambda: DeepONetOsc(), "deeponet"),
    }
    for name in args.configs:
        make, mode = registry[name]
        for seed in args.seeds:
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            print(f"{name} seed {seed}", flush=True)
            model = train(make(), data, mode, args.epochs, args.lr)
            with torch.no_grad():
                comps = None
                if mode != "deeponet":
                    comps = {}
                    for T in Ts:
                        uc, Ur, Ui = model(data, te, T)
                        comps[str(T)] = {
                            "const_component_median": float(rel_err(uc, data[f"c{T}"][te]).median()),
                            "sin_component_median": float(rel_err(Ui, data[f"s{T}"][te]).median()),
                            "sin_component_max": float(rel_err(Ui, data[f"s{T}"][te]).max()),
                        }
                record(name, lambda T: model.predict(data, te, T), comps)
            print(f"  {time.perf_counter() - t0:.0f}s", flush=True)

    summary = {name: {"per_seed": runs,
                      "mean_over_seeds": {str(T): {k: float(np.mean([r[str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in Ts}}
               for name, runs in results.items()}
    (args.out_dir / "osc_turnpike_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, key, title in ((axes[0], "median", "test median rel. L2"), (axes[1], "max", "test max rel. L2")):
        for name in summary:
            ax.semilogy(Ts, [summary[name]["mean_over_seeds"][str(T)][key] for T in Ts], marker="o", label=name)
        ax.set_xscale("log", base=2); ax.set_xticks(Ts, [str(T) for T in Ts]); ax.grid(alpha=0.3)
        ax.set_xlabel("horizon T (all models trained on T=1)"); ax.set_title(title); ax.legend(fontsize=7)
    ax = axes[2]
    errs4 = rel_err(torch.tensor(examples["tp_osc"]), data["u4"][te]).numpy()
    i = int(np.argsort(errs4)[len(errs4) // 2])
    tt = np.arange(1, 4 * 320 + 1) / 320
    ax.plot(tt, data["u4"][te][i].numpy(), "k", lw=2.5, label="truth, T=4")
    for name in ("tp_osc", "tp_osc_sin", "deeponet_osc", "stretch_oracle_T1"):
        if name in examples:
            ax.plot(tt, examples[name][i], label=name)
    ax.set_title(f"median test sample, omega={float(data['omega'][te][i]):.2f}, A={float(data['A'][te][i]):.2f}")
    ax.set_xlabel("t"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "osc_turnpike.png", dpi=140)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()

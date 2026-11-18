#!/usr/bin/env python3
"""Box-constrained controls: turnpike surrogate on the adjoint trace + projection vs ablations and baselines.

All quantities are normalized by the unconstrained static optimum u_bar, so the bounds are
exactly theta_a, theta_b. With q = p(0, t)/alpha the optimal control is u = clip(q, theta_a, theta_b).

    tp_q         q/u_bar = q_bar/u_bar + sum a e^{-lam t} + sum b e^{-kap (T-t)}; u = clip(q). Loss on q.
    tp_q_uloss   same model, loss only on u = clip(q) (projection inside, adjoint never observed)
    tp_u_direct  no projection: u/u_bar = u_c/u_bar + same exponential layers. Loss on u.
    cnn_u        dilated CNN without coordinates predicting u (trained on T in {1, 2})
    cnn_q_clip   same CNN predicting q, output clip(q) (generic net + the projection prior)
Closed-form pieces: q_bar, u_c = clip(1, theta_a, theta_b) (static constrained optimum), rate floor lambda_1(beta).
Oracle reference: clip of the exact unconstrained control.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines_multihorizon import DilatedCNN  # noqa: E402
from turnpike_surrogate import first_eig, grid, inv_softplus, rel_err, summarize  # noqa: E402

T_EVAL = [1, 2, 4, 8]


def load(path):
    d = np.load(path)
    tr = d["train_idx"]
    X = d["X"]
    ub = d["u_bar"]
    p = {
        "x": torch.tensor((X - X[tr].mean(0)) / X[tr].std(0)),
        "lo": torch.tensor(X[:, 4]), "hi": torch.tensor(X[:, 5]),
        "qbar": torch.tensor(d["q_bar"] / ub), "uc": torch.tensor(d["u_c"] / ub),
        "floor": torch.tensor([first_eig(b) for b in d["beta"]]),
        "dt": float(d["dt"]), "tr": torch.tensor(tr), "va": torch.tensor(d["val_idx"]), "te": torch.tensor(d["test_idx"]),
        "sat": torch.tensor(X[:, 5] < 1.0),
    }
    p["one"] = torch.ones(len(ub))
    p["dqbar"] = p["qbar"] - 1.0  # constraint-induced shift of the adjoint plateau (closed form)
    for T in T_EVAL:
        p[f"UU{T}"] = torch.tensor(d[f"Uunc_T{T}"] / ub[:, None])
        p[f"U{T}"] = torch.tensor(d[f"U_T{T}"] / ub[:, None])
        p[f"Q{T}"] = torch.tensor(d[f"Q_T{T}"] / ub[:, None])
        p[f"C{T}"] = torch.tensor(np.clip(d[f"Uunc_T{T}"], d["lo"][:, None], d["hi"][:, None]) / ub[:, None])
    return p


def clip(v, p, idx):
    return torch.minimum(torch.maximum(v, p["lo"][idx, None]), p["hi"][idx, None])


class BoxTurnpike(nn.Module):
    """n_in = 6 uses (geometry, alpha, beta, theta_a, theta_b); n_in = 4 drops the bounds."""

    def __init__(self, r=4, width=128, shared=True, base="qbar", n_in=6):
        super().__init__()
        self.r, self.shared, self.base, self.n_in = r, shared, base, n_in
        self.enc = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rate_head = nn.Linear(width, r if shared else 2 * r)
        self.amp_head = nn.Linear(width, 2 * r)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            self.rate_head.weight.mul_(0.05)
            self.rate_head.bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)).repeat(1 if shared else 2))
            self.amp_head.weight.mul_(0.05)
            self.amp_head.bias.zero_()

    def raw(self, p, idx, T):
        t, s = grid(T, p["dt"])
        h = self.enc(p["x"][idx][:, : self.n_in])
        rates = p["floor"][idx, None] + nn.functional.softplus(self.rate_head(h))
        lam = rates[:, : self.r]
        kap = lam if self.shared else rates[:, self.r :]
        a, b = self.amp_head(h).chunk(2, dim=-1)
        return (p[self.base][idx, None] + torch.einsum("bi,bin->bn", a, torch.exp(-lam[:, :, None] * t[None, None, :]))
                + torch.einsum("bi,bin->bn", b, torch.exp(-kap[:, :, None] * s[None, None, :])))


class Hybrid(nn.Module):
    """q = unconstrained turnpike (mu_g only, trained first on unconstrained controls, then frozen)
         + constraint correction (closed-form plateau shift q_bar - u_bar plus learned layers)."""

    def __init__(self):
        super().__init__()
        self.unc = BoxTurnpike(base="one", n_in=4)
        self.corr = BoxTurnpike(base="dqbar", n_in=6)
        with torch.no_grad():
            self.corr.amp_head.weight.mul_(0.1)

    def raw(self, p, idx, T):
        return self.unc.raw(p, idx, T) + self.corr.raw(p, idx, T)


class CNNWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DilatedCNN(6)

    def raw(self, p, idx, T):
        t, s = grid(T, p["dt"])
        return self.net(p["x"][idx], torch.ones(len(idx), len(t)), t, s)


SPECS = {
    # name: (maker, loss target, project output)
    "tp_q": (lambda: BoxTurnpike(base="qbar"), "q", True),
    "tp_q_r6": (lambda: BoxTurnpike(r=6, base="qbar"), "q", True),
    "tp_q_uloss": (lambda: BoxTurnpike(base="qbar"), "u", True),
    "tp_q_uloss_r6": (lambda: BoxTurnpike(r=6, base="qbar"), "u", True),
    "tp_q_uloss_r8": (lambda: BoxTurnpike(r=8, base="qbar"), "u", True),
    "tp_u_direct": (lambda: BoxTurnpike(base="uc"), "u", False),
    "cnn_u": (lambda: CNNWrap(), "u", False),
    "cnn_q_clip": (lambda: CNNWrap(), "q", True),
    # learned unconstrained turnpike (mu_g only, trained on unconstrained controls), output clipped
    "unc_clip": (lambda: BoxTurnpike(base="one", n_in=4), "uu", True),
    # two-stage: unc trained on unconstrained controls, then frozen; correction trained on u (through clip) / on q
    "hybrid_uloss": (lambda: Hybrid(), "u", True),
    "hybrid_qloss": (lambda: Hybrid(), "q", True),
}


def outputs(model, p, idx, T, project):
    raw = model.raw(p, idx, T)
    return raw, (clip(raw, p, idx) if project else raw)


def val_err(model, p, idx, T, target, project):
    raw, u = outputs(model, p, idx, T, project)
    if target == "uu":
        return rel_err(raw, p[f"UU{T}"][idx])
    if target == "q":
        return rel_err(raw, p[f"Q{T}"][idx])
    return rel_err(u, p[f"U{T}"][idx])


def loss_fn(model, p, idx, T, target, project):
    return (val_err(model, p, idx, T, target, project) ** 2).mean()


def fit(model, p, target, project, train_T, epochs, lr, batch):
    """Hybrid models: stage 1 trains the unconstrained part on unconstrained controls and freezes it."""
    if isinstance(model, Hybrid):
        print("  stage 1: unconstrained turnpike", flush=True)
        train(model.unc, p, "uu", False, train_T, epochs, lr, batch)
        for q in model.unc.parameters():
            q.requires_grad_(False)
        print("  stage 2: constraint correction", flush=True)
    return train(model, p, target, project, train_T, epochs, lr, batch)


def train(model, p, target, project, train_T, epochs, lr, batch):
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=lr)
    steps = int(np.ceil(len(p["tr"]) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=lr * 1e-3)
    best, best_state, v = float("inf"), copy.deepcopy(model.state_dict()), float("nan")
    for ep in range(1, epochs + 1):
        perm = p["tr"][torch.randperm(len(p["tr"]), device=p["tr"].device)]
        for b in range(steps):
            idx = perm[b * batch : (b + 1) * batch]
            opt.zero_grad()
            loss = sum(loss_fn(model, p, idx, T, target, project) for T in train_T) / len(train_T)
            loss.backward()
            opt.step()
            sched.step()
        if ep % max(1, epochs // 40) == 0 or ep == epochs:
            with torch.no_grad():
                v = float(np.mean([float(val_err(model, p, p["va"], T, target if target == "uu" else "u", project).median())
                                   for T in train_T]))
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
        if ep % max(1, epochs // 5) == 0:
            print(f"    ep {ep} loss {float(loss.detach()):.3e} val(u) {v:.3e}", flush=True)
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=700)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--oracle", action="store_true", help="also report clip(unconstrained) oracle")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load(args.data)
    te = p["te"]
    sat = p["sat"][te].cpu().numpy()
    report = {"train_T": args.train_T, "models": {}}

    def evaluate(pred_fn):
        res = {}
        for T in T_EVAL:
            e = rel_err(pred_fn(T), p[f"U{T}"][te]).cpu().numpy()
            res[str(T)] = {"all": summarize(e), "saturated": summarize(e[sat]), "free": summarize(e[~sat])}
        return res

    if args.oracle:
        report["models"]["clip_of_unconstrained_oracle"] = {"runs": [{"test": evaluate(lambda T: p[f"C{T}"][te])}]}
    for name in args.models:
        make, target, project = SPECS[name]
        runs = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = make()
            t0 = time.perf_counter()
            print(f"{name} seed {seed} (train T={args.train_T}, loss on {target}, project={project})", flush=True)
            model = fit(model, p, target, project, args.train_T, args.epochs, args.lr, args.batch)
            with torch.no_grad():
                res = evaluate(lambda T: outputs(model, p, te, T, project)[1])
                if target == "q" or name.startswith(("tp_q", "hybrid")):
                    for T in T_EVAL:
                        res[str(T)]["q_err"] = summarize(rel_err(model.raw(p, te, T), p[f"Q{T}"][te]).cpu().numpy())
            runs.append({"seed": seed, "train_seconds": time.perf_counter() - t0, "test": res})
            print(f"  {name}: " + "  ".join(f"T={T}: med {res[str(T)]['all']['median']:.2e} max {res[str(T)]['all']['max']:.2e}" for T in T_EVAL), flush=True)
        report["models"][name] = {
            "runs": runs,
            "mean_over_seeds": {str(T): {g: {k: float(np.mean([r["test"][str(T)][g][k] for r in runs])) for k in ("median", "p90", "max")}
                                         for g in ("all", "saturated", "free")} for T in T_EVAL},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

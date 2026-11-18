#!/usr/bin/env python3
"""Surrogates for the nonlinear moving-target problem (gen_semilinear_moving_data.py), u(s, t), m = 99.

All turnpike variants:  u = rho [ Per(s, theta(t)) + sum_i e^{-lam_i t} A_i(s, theta(t)) + sum_i e^{-lam_i (T - t)} B_i(s, theta(t)) ]
with theta(t) = omega t + phi0 and rates lam_i >= lambda_1(beta). Time enters only through theta(t), t and T - t
(T-invariant by construction); the periodic part is periodic by construction. Layers are Floquet-type modes: the
linearization around a time-periodic orbit gives decaying modes whose shape is modulated by the phase, so
A_i(s, theta) = sum_{k<=K_l} [alpha_ik(s) cos k theta + beta_ik(s) sin k theta] with coefficients from (mu, phi0) and B_i
likewise from (mu, theta_T), theta_T = omega T + phi0.
    tp_lin    Per = (1 + kappa(mu)) P_lin + sum_{k<=K} Re(c_k(s; mu) e^{i k theta});  P_lin = linear periodic turnpike
              (harmonic balance), rho = RMS(P_lin) on T = 1
    tp_pure   no harmonic balance: Per = sum_{k<=K} Re(c_k(s; mu) e^{i k (theta)}) and a learned scale rho(mu)
    lin_only  u = linear periodic turnpike (no learning)
    cnn       dilated CNN, input: mu, cos/sin theta(t), the target at the m control nodes, ones; normalized by rho
mu = (w, log alpha, log beta, a, omega, log sigma, log gamma) standardized.
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
from moving_surrogate import periodic  # noqa: E402
from turnpike_surrogate import grid, inv_softplus, summarize  # noqa: E402

T_EVAL = [1, 2, 4, 8]
MU_COLS = [0, 1, 2, 3, 4, 5, 7]


def rel(pred, y):
    return torch.linalg.norm((pred - y).flatten(1), dim=1) / torch.linalg.norm(y.flatten(1), dim=1)


def load(path):
    d = np.load(path)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    X = d["X"]
    mu = X[:, MU_COLS]
    nH = int(d["n_harm"].max()) + 1
    uhat = torch.tensor(d["uhat_lin"][:, :nH].astype(np.complex128))
    omega, phi0 = torch.tensor(X[:, 4]), torch.tensor(X[:, 6])
    dt = float(d["dt"])
    p = {"mu": torch.tensor((mu - mu[tr].mean(0)) / mu[tr].std(0)), "omega": omega, "phi0": phi0, "floor": torch.tensor(d["lam1"]),
         "a": torch.tensor(X[:, 3]), "sigma": torch.tensor(10.0 ** X[:, 5]), "dt": dt, "m": int(d["m"]),
         "tr": torch.tensor(tr), "va": torch.tensor(va), "te": torch.tensor(te), "row": {}, "P": {}, "dP": {}, "Y": {}, "U": {}}
    nvec = torch.arange(nH, dtype=torch.float64)
    duhat = uhat * (1j * nvec)[None, :, None]  # d/dtheta of the linear periodic turnpike
    t1, _ = grid(1, dt)
    rho = torch.sqrt((periodic(uhat, omega, phi0, t1) ** 2).mean(dim=(1, 2)))
    p["rho"] = rho
    for T in T_EVAL:
        t, _ = grid(T, dt)
        rows = torch.arange(len(X)) if T <= 2 else torch.tensor(te)
        rowmap = torch.full((len(X),), -1, dtype=torch.long)
        rowmap[rows] = torch.arange(len(rows))
        p["row"][T] = rowmap
        U = torch.tensor((d[f"U_T{T}"] if T <= 2 else d[f"U_T{T}_test"]).astype(np.float64))
        p["U"][T] = U
        p["Y"][T] = U / rho[rows, None, None]
        p["P"][T] = torch.cat([periodic(uhat[rows[k : k + 50]], omega[rows[k : k + 50]], phi0[rows[k : k + 50]], t)
                               for k in range(0, len(rows), 50)]) / rho[rows, None, None]
        p["dP"][T] = torch.cat([periodic(duhat[rows[k : k + 50]], omega[rows[k : k + 50]], phi0[rows[k : k + 50]], t)
                                for k in range(0, len(rows), 50)]) / (10.0 * rho[rows, None, None])
    return p


def mlp(i, width, o, zero_last=True):
    net = nn.Sequential(nn.Linear(i, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, o))
    if zero_last:
        with torch.no_grad():
            net[-1].weight.mul_(0.05)
            net[-1].bias.zero_()
    return net


def fourier(theta, K):
    k = torch.arange(1, K + 1, dtype=theta.dtype)
    return torch.cat([torch.cos(theta[..., None] * k), torch.sin(theta[..., None] * k)], -1)


class MovingTurnpikeNL(nn.Module):
    def __init__(self, m, base: bool, r=8, K=24, K_layer=2, width=256):
        super().__init__()
        self.m, self.r, self.K, self.Kl, self.base = m, r, K, K_layer, base
        self.rate = mlp(7, width, r, zero_last=False)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            self.rate[-1].weight.mul_(0.05)
            self.rate[-1].bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))
        self.per = mlp(7, width, (2 * K + 1) * m)  # harmonic coefficients (cos/sin) of the periodic part
        self.kappa = mlp(7, 64, 1)
        nb = 2 * K_layer + 1
        self.lay_a = mlp(9, width, r * nb * m)
        self.lay_b = mlp(9, width, r * nb * m)
        self.log_rho = mlp(7, 64, 1, zero_last=False) if not base else None

    def basis(self, theta, K):
        return torch.cat([torch.ones_like(theta)[..., None], fourier(theta, K)], -1)  # (..., 2K+1)

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        mu, ph0, om = p["mu"][idx], p["phi0"][idx], p["omega"][idx]
        B = len(idx)
        th = om[:, None] * t[None, :] + ph0[:, None]  # (B, N)
        thT = om * T + ph0
        Ph = self.basis(th, self.K)  # (B, N, 2K+1)
        coef = self.per(mu).view(B, 2 * self.K + 1, self.m)
        per = torch.einsum("bnk,bkm->bnm", Ph, coef)
        if self.base:
            per = per + (1.0 + self.kappa(mu)[:, :, None]) * p["P"][T][p["row"][T][idx]]
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(mu))
        Pl = self.basis(th, self.Kl)  # (B, N, 2Kl+1)
        nb = 2 * self.Kl + 1
        ca = self.lay_a(torch.cat([mu, torch.cos(ph0)[:, None], torch.sin(ph0)[:, None]], 1)).view(B, self.r, nb, self.m)
        cb = self.lay_b(torch.cat([mu, torch.cos(thT)[:, None], torch.sin(thT)[:, None]], 1)).view(B, self.r, nb, self.m)
        E = torch.exp(-lam[:, :, None] * t[None, None, :])  # (B, r, N)
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        L0 = torch.einsum("brn,bnk,brkm->bnm", E, Pl, ca)
        L1 = torch.einsum("brn,bnk,brkm->bnm", F, Pl, cb)
        out = per + L0 + L1
        if not self.base:  # no harmonic balance at inference: the scale is learned too
            out = out * torch.exp(self.log_rho(mu))[:, :, None]
        return out


class MovingTurnpikeLocal(nn.Module):
    """tp_local: Per = P_lin + Delta, Delta(s, theta) from a 1D CNN over s applied at every phase theta(t) to the local
    channels [P_lin, dP_lin/dtheta, target, mu]: the nonlinear correction of the periodic orbit is mainly a local
    function of the heat input along the edge, so weights are shared over s and over time (periodic and T-invariant by
    construction). Layers: Floquet modes as in MovingTurnpikeNL (smaller heads)."""

    def __init__(self, m, r=6, K_layer=1, width=48, width_mlp=128):
        super().__init__()
        self.m, self.r, self.Kl = m, r, K_layer
        self.lift = nn.Conv1d(3 + 7, width, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8)])
        self.out = nn.Conv1d(width, 1, 1)
        with torch.no_grad():
            self.out.weight.mul_(0.05)
            self.out.bias.zero_()
        self.rate = mlp(7, width_mlp, r, zero_last=False)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            self.rate[-1].weight.mul_(0.05)
            self.rate[-1].bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))
        nb = 2 * K_layer + 1
        self.lay_a = mlp(9, width_mlp, r * nb * m)
        self.lay_b = mlp(9, width_mlp, r * nb * m)

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        mu, ph0, om = p["mu"][idx], p["phi0"][idx], p["omega"][idx]
        B, N = len(idx), len(t)
        rows = p["row"][T][idx]
        P, dP = p["P"][T][rows], p["dP"][T][rows]  # (B, N, m)
        th = om[:, None] * t[None, :] + ph0[:, None]
        sg = torch.arange(1, self.m + 1, dtype=t.dtype) / (self.m + 1)
        c = 0.5 + p["a"][idx, None] * torch.sin(th)
        g = torch.exp(-0.5 * ((sg[None, None, :] - c[:, :, None]) / p["sigma"][idx, None, None]) ** 2)  # (B, N, m)
        h = torch.stack([P, dP, g], 2).reshape(B * N, 3, self.m)
        h = torch.cat([h, mu[:, None, :, None].expand(B, N, 7, self.m).reshape(B * N, 7, self.m)], 1)
        h = nn.functional.gelu(self.lift(h))
        for conv in self.blocks:
            h = h + nn.functional.gelu(conv(h))
        per = P + self.out(h).view(B, N, self.m)
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(mu))
        Pl = torch.cat([torch.ones_like(th)[..., None], fourier(th, self.Kl)], -1)
        nb = 2 * self.Kl + 1
        thT = om * T + ph0
        ca = self.lay_a(torch.cat([mu, torch.cos(ph0)[:, None], torch.sin(ph0)[:, None]], 1)).view(B, self.r, nb, self.m)
        cb = self.lay_b(torch.cat([mu, torch.cos(thT)[:, None], torch.sin(thT)[:, None]], 1)).view(B, self.r, nb, self.m)
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        return per + torch.einsum("brn,bnk,brkm->bnm", E, Pl, ca) + torch.einsum("brn,bnk,brkm->bnm", F, Pl, cb)


class LinOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, p, idx, T):
        return p["P"][T][p["row"][T][idx]] + 0 * self.dummy


class CNNmove(nn.Module):
    def __init__(self, m, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.m = m
        self.lift = nn.Conv1d(7 + 2 + m + 1, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=dd, dilation=dd) for dd in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))

    def forward(self, p, idx, T):
        t, _ = grid(T, p["dt"])
        B, N = len(idx), len(t)
        th = p["omega"][idx, None] * t[None, :] + p["phi0"][idx, None]
        s = torch.arange(1, self.m + 1, dtype=t.dtype) / (self.m + 1)
        c = 0.5 + p["a"][idx, None] * torch.sin(th)
        g = torch.exp(-0.5 * ((s[None, :, None] - c[:, None, :]) / p["sigma"][idx, None, None]) ** 2)
        h = torch.cat([p["mu"][idx, :, None].expand(-1, -1, N), torch.cos(th)[:, None], torch.sin(th)[:, None], g, torch.ones(B, 1, N)], 1)
        h = self.lift(h)
        for conv in self.convs:
            h = h + nn.functional.gelu(conv(h))
        return self.proj(h).transpose(1, 2)


class CNNres(CNNmove):
    """Residual CNN with the same reference information as tp_local: the inputs of CNNmove plus the exact linear periodic
    turnpike P_lin and dP_lin/dtheta (m channels each); output = P_lin + CNN(...), last layer initialized at ~0. It has no
    explicit end-layer structure: the ends are visible only through the zero padding, as for CNNmove."""

    def __init__(self, m, width=64):
        super().__init__(m, width=width)
        self.lift = nn.Conv1d(7 + 2 + m + 1 + 2 * m, width, 1)
        with torch.no_grad():
            self.proj[-1].weight.mul_(0.05)
            self.proj[-1].bias.zero_()

    def forward(self, p, idx, T):
        t, _ = grid(T, p["dt"])
        B, N = len(idx), len(t)
        rows = p["row"][T][idx]
        P, dP = p["P"][T][rows], p["dP"][T][rows]  # (B, N, m)
        th = p["omega"][idx, None] * t[None, :] + p["phi0"][idx, None]
        s = torch.arange(1, self.m + 1, dtype=t.dtype) / (self.m + 1)
        c = 0.5 + p["a"][idx, None] * torch.sin(th)
        g = torch.exp(-0.5 * ((s[None, :, None] - c[:, None, :]) / p["sigma"][idx, None, None]) ** 2)
        h = torch.cat([p["mu"][idx, :, None].expand(-1, -1, N), torch.cos(th)[:, None], torch.sin(th)[:, None], g, torch.ones(B, 1, N),
                       P.transpose(1, 2), dP.transpose(1, 2)], 1)
        h = self.lift(h)
        for conv in self.convs:
            h = h + nn.functional.gelu(conv(h))
        return P + self.proj(h).transpose(1, 2)


def target_of(model, p, idx, T):
    """Training/evaluation target: the unnormalized control for tp_pure (own scale), the rho-normalized one otherwise."""
    rows = p["row"][T][idx]
    return p["U"][T][rows] if isinstance(model, MovingTurnpikeNL) and not model.base else p["Y"][T][rows]


def train(model, p, train_T, epochs, lr, batch):
    if epochs == 0:
        return model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = int(np.ceil(len(p["tr"]) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=lr * 1e-3)
    best, best_state, v = float("inf"), copy.deepcopy(model.state_dict()), float("nan")
    for ep in range(1, epochs + 1):
        perm = p["tr"][torch.randperm(len(p["tr"]), device=p["tr"].device)]
        for b in range(steps):
            idx = perm[b * batch : (b + 1) * batch]
            opt.zero_grad()
            loss = sum((rel(model(p, idx, T), target_of(model, p, idx, T)) ** 2).mean() for T in train_T) / len(train_T)
            loss.backward()
            opt.step()
            sched.step()
        if ep % max(1, epochs // 40) == 0 or ep == epochs:
            with torch.no_grad():
                v = float(np.mean([float(rel(model(p, p["va"], T), target_of(model, p, p["va"], T)).median()) for T in train_T]))
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
        if ep % max(1, epochs // 5) == 0:
            print(f"    ep {ep} loss {float(loss.detach()):.3e} val {v:.3e}", flush=True)
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=["tp_lin", "tp_pure", "tp_local", "lin_only", "cnn", "cnn_res"], required=True)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--K-layer", type=int, default=2)
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load(args.data)
    if args.fp32:
        torch.set_default_dtype(torch.float32)
        conv = lambda v: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v  # noqa: E731
        p = {k: ({kk: conv(vv) for kk, vv in v.items()} if isinstance(v, dict) else conv(v)) for k, v in p.items()}
    te = p["te"]
    runs = []
    for seed in (args.seeds if args.model != "lin_only" else [0]):
        torch.manual_seed(seed)
        model = {"tp_lin": lambda: MovingTurnpikeNL(p["m"], True, r=args.r, K=args.K, K_layer=args.K_layer),
                 "tp_pure": lambda: MovingTurnpikeNL(p["m"], False, r=args.r, K=args.K, K_layer=args.K_layer),
                 "tp_local": lambda: MovingTurnpikeLocal(p["m"], r=args.r, K_layer=args.K_layer),
                 "lin_only": LinOnly, "cnn": lambda: CNNmove(p["m"]), "cnn_res": lambda: CNNres(p["m"])}[args.model]()
        n_par = sum(q.numel() for q in model.parameters())
        print(f"{args.model} seed {seed} ({n_par} params, train T={args.train_T})", flush=True)
        t0 = time.perf_counter()
        model = train(model, p, args.train_T, args.epochs if args.model != "lin_only" else 0, args.lr, args.batch)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            res, preds = {}, {}
            for T in T_EVAL:
                out = torch.cat([model(p, te[k : k + 20], T) for k in range(0, len(te), 20)])
                Uref = p["U"][T][p["row"][T][te]]
                u = out if (isinstance(model, MovingTurnpikeNL) and not model.base) else out * p["rho"][te, None, None]
                res[str(T)] = summarize(rel(u, Uref).cpu().double().numpy())
                if T in (1, 8):
                    preds[f"U_T{T}"] = u.cpu().numpy().astype(np.float32)
            if args.save_pred is not None:
                args.save_pred.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_pred / f"pred_{args.model}_seed{seed}.npz", test_idx=te.cpu().numpy(), **preds)
        runs.append({"seed": seed, "params": n_par, "train_seconds": secs, "test": res})
        print(f"  {args.model}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in T_EVAL)
              + f" ({secs:.0f}s)", flush=True)
    rep = {"model": args.model, "train_T": args.train_T, "K": args.K, "K_layer": args.K_layer, "r": args.r, "runs": runs,
           "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in T_EVAL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

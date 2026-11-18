#!/usr/bin/env python3
"""Surrogates for the moving-target problem (gen_moving_data.py), u(s, t) with m = 99, train on short horizons.

turnpike   u = u_hat(s, t) + sum_i a_i(s) e^{-lam_i t} + sum_i b_i(s) e^{-lam_i (T - t)}
           u_hat: exact periodic turnpike (stored harmonics); deviations from it solve the time-invariant homogeneous
           optimality system, so they are exponential modes: a_i from (mu, cos phi0, sin phi0), b_i from
           (mu, cos theta_T, sin theta_T) with theta_T = omega T + phi0 (target phase at the horizon), rates >= lambda_1.
periodic   u = u_hat only (no layers; reference)
cnn        dilated CNN in time without coordinates; input channels: mu (6), cos/sin of the target phase, the target
           sampled at the m control nodes, ones; output m channels.
Normalization by rho = RMS of u_hat over T = 1; error = relative L2 over (t, s).
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
from turnpike_surrogate import grid, inv_softplus, summarize  # noqa: E402

T_EVAL = [1, 2, 4, 8]


def rel(pred, y):
    return torch.linalg.norm((pred - y).flatten(1), dim=1) / torch.linalg.norm(y.flatten(1), dim=1)


def periodic(uhat, omega, phi0, t):
    """u_hat(t) for a batch: uhat (B, H, m) complex, omega/phi0 (B,), t (N,) -> (B, N, m) real."""
    H = uhat.shape[1]
    n = torch.arange(H, dtype=t.dtype)
    th = (omega[:, None, None] * t[None, :, None] + phi0[:, None, None]) * n[None, None, :]
    E = torch.exp(1j * th)
    E[:, :, 1:] *= 2.0
    return torch.real(torch.einsum("bnh,bhm->bnm", E, uhat))


def load(path):
    d = np.load(path)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    X = d["X"]
    mu = X[:, :6]
    nH = int(d["n_harm"].max()) + 1
    uhat = torch.tensor(d["uhat"][:, :nH].astype(np.complex128))
    omega, phi0 = torch.tensor(X[:, 4]), torch.tensor(X[:, 6])
    dt = float(d["dt"])
    p = {"mu": torch.tensor((mu - mu[tr].mean(0)) / mu[tr].std(0)), "omega": omega, "phi0": phi0, "floor": torch.tensor(d["lam1"]),
         "a": torch.tensor(X[:, 3]), "sigma": torch.tensor(10.0 ** X[:, 5]), "dt": dt, "m": int(d["m"]),
         "tr": torch.tensor(tr), "va": torch.tensor(va), "te": torch.tensor(te), "row": {}, "P": {}, "Y": {}}
    t1, _ = grid(1, dt)
    rho = torch.sqrt((periodic(uhat, omega, phi0, t1) ** 2).mean(dim=(1, 2)))
    p["rho"] = rho
    for T in T_EVAL:
        t, _ = grid(T, dt)
        rows = torch.arange(len(X)) if T <= 2 else torch.tensor(te)
        rowmap = torch.full((len(X),), -1, dtype=torch.long)
        rowmap[rows] = torch.arange(len(rows))
        p["row"][T] = rowmap
        U = d[f"U_T{T}"] if T <= 2 else d[f"U_T{T}_test"]
        p["Y"][T] = torch.tensor(U.astype(np.float64)) / rho[rows, None, None]
        Ps = [periodic(uhat[rows[k : k + 50]], omega[rows[k : k + 50]], phi0[rows[k : k + 50]], t) for k in range(0, len(rows), 50)]
        p["P"][T] = torch.cat(Ps) / rho[rows, None, None]
    return p


def mlp(i, width, o):
    return nn.Sequential(nn.Linear(i, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, o))


class MoveTurnpike(nn.Module):
    def __init__(self, m, r=8, width=256):
        super().__init__()
        self.m, self.r = m, r
        self.rate = mlp(6, width, r)
        self.amp_a = mlp(8, width, r * m)
        self.amp_b = mlp(8, width, r * m)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            self.rate[-1].weight.mul_(0.05)
            self.rate[-1].bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))
            for net in (self.amp_a, self.amp_b):
                net[-1].weight.mul_(0.05)
                net[-1].bias.zero_()

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        mu, ph0 = p["mu"][idx], p["phi0"][idx]
        thT = p["omega"][idx] * T + ph0
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(mu))
        A = self.amp_a(torch.cat([mu, torch.cos(ph0)[:, None], torch.sin(ph0)[:, None]], 1)).view(-1, self.r, self.m)
        Bm = self.amp_b(torch.cat([mu, torch.cos(thT)[:, None], torch.sin(thT)[:, None]], 1)).view(-1, self.r, self.m)
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        return p["P"][T][p["row"][T][idx]] + torch.einsum("bin,bim->bnm", E, A) + torch.einsum("bin,bim->bnm", F, Bm)


class PeriodicOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, p, idx, T):
        return p["P"][T][p["row"][T][idx]] + 0 * self.dummy


class CNNmove(nn.Module):
    def __init__(self, m, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.m = m
        self.lift = nn.Conv1d(6 + 2 + m + 1, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=dd, dilation=dd) for dd in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))

    def forward(self, p, idx, T):
        t, _ = grid(T, p["dt"])
        B, N = len(idx), len(t)
        th = p["omega"][idx, None] * t[None, :] + p["phi0"][idx, None]
        s = torch.arange(1, self.m + 1, dtype=t.dtype) / (self.m + 1)
        c = 0.5 + p["a"][idx, None] * torch.sin(th)  # (B, N)
        g = torch.exp(-0.5 * ((s[None, :, None] - c[:, None, :]) / p["sigma"][idx, None, None]) ** 2)  # (B, m, N)
        h = torch.cat([p["mu"][idx, :, None].expand(-1, -1, N), torch.cos(th)[:, None], torch.sin(th)[:, None], g, torch.ones(B, 1, N)], 1)
        h = self.lift(h)
        for conv in self.convs:
            h = h + nn.functional.gelu(conv(h))
        return self.proj(h).transpose(1, 2)


def train(model, p, train_T, epochs, lr, batch):
    if not list(model.parameters()) or epochs == 0:
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
            loss = sum((rel(model(p, idx, T), p["Y"][T][p["row"][T][idx]]) ** 2).mean() for T in train_T) / len(train_T)
            loss.backward()
            opt.step()
            sched.step()
        if ep % max(1, epochs // 40) == 0 or ep == epochs:
            with torch.no_grad():
                v = float(np.mean([float(rel(model(p, p["va"], T), p["Y"][T][p["row"][T][p["va"]]]).median()) for T in train_T]))
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
    ap.add_argument("--model", choices=["turnpike", "periodic", "cnn"], required=True)
    ap.add_argument("--r", type=int, default=8)
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
    for seed in (args.seeds if args.model != "periodic" else [0]):
        torch.manual_seed(seed)
        model = {"turnpike": lambda: MoveTurnpike(p["m"], r=args.r), "periodic": PeriodicOnly, "cnn": lambda: CNNmove(p["m"])}[args.model]()
        n_par = sum(q.numel() for q in model.parameters())
        print(f"{args.model} seed {seed} ({n_par} params, train T={args.train_T})", flush=True)
        t0 = time.perf_counter()
        model = train(model, p, args.train_T, args.epochs if args.model != "periodic" else 0, args.lr, args.batch)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            res, preds = {}, {}
            for T in T_EVAL:
                out = torch.cat([model(p, te[k : k + 25], T) for k in range(0, len(te), 25)])
                res[str(T)] = summarize(rel(out, p["Y"][T][p["row"][T][te]]).cpu().double().numpy())
                if T in (1, 8):
                    preds[f"U_T{T}"] = (out * p["rho"][te, None, None]).cpu().numpy().astype(np.float32)
            if args.save_pred is not None:
                args.save_pred.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_pred / f"pred_{args.model}_seed{seed}.npz", test_idx=te.cpu().numpy(), **preds)
        runs.append({"seed": seed, "params": n_par, "train_seconds": secs, "test": res})
        print(f"  {args.model}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in T_EVAL)
              + f" ({secs:.0f}s)", flush=True)
    rep = {"model": args.model, "train_T": args.train_T, "runs": runs,
           "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in T_EVAL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

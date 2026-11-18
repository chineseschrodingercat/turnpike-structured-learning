#!/usr/bin/env python3
"""Multi-input (u(s,t), m = 99) turnpike surrogate vs dilated CNN; train on short horizons, test T = 1, 2, 4, 8.

    u(s,t)/rho = u_bar(s)/rho + sum_i a_i(s) e^{-lam_i t} + sum_i b_i(s) e^{-lam_i (T-t)}
u_bar(s): closed-form static multi-input optimum; rho = RMS of u_bar (scale); lam_i >= lambda_1(beta), shared
between the two ends and across s; the encoder outputs the r rates and the 2 r m amplitudes.
cnn: dilated residual CNN in time without coordinates, input [mu (broadcast), ones], output m channels.
Errors: relative L2 over all (t, s) per test sample.
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


def load(path):
    d = np.load(path)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    X = d["X"]
    ub = d["u_bar"]
    rho = np.sqrt(np.mean(ub**2, axis=1))
    p = {"x": torch.tensor((X - X[tr].mean(0)) / X[tr].std(0)), "floor": torch.tensor(d["lam1"]),
         "ubar_n": torch.tensor(ub / rho[:, None]), "rho": torch.tensor(rho), "dt": float(d["dt"]), "m": int(d["m"]),
         "tr": torch.tensor(tr), "va": torch.tensor(va), "te": torch.tensor(te)}
    for T in (1, 2):
        p[f"Y{T}"] = torch.tensor(d[f"U_T{T}"].astype(np.float64) / rho[:, None, None])
    for T in (4, 8):
        Yt = torch.zeros(len(X), *d[f"U_T{T}_test"].shape[1:], dtype=torch.float64)
        Yt[te] = torch.tensor(d[f"U_T{T}_test"].astype(np.float64) / rho[te, None, None])
        p[f"Y{T}"] = Yt  # only test rows are filled
    return p


class MITurnpike(nn.Module):
    def __init__(self, m, r=4, width=256):
        super().__init__()
        self.r, self.m = r, m
        self.enc = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rate = nn.Linear(width, r)
        self.amp = nn.Linear(width, 2 * r * m)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            self.rate.weight.mul_(0.05)
            self.rate.bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))
            self.amp.weight.mul_(0.05)
            self.amp.bias.zero_()

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        h = self.enc(p["x"][idx])
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(h))
        ab = self.amp(h).view(-1, 2, self.r, self.m)
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        return (p["ubar_n"][idx][:, None, :] + torch.einsum("bin,bim->bnm", E, ab[:, 0]) + torch.einsum("bin,bim->bnm", F, ab[:, 1]))


class CNNmi(nn.Module):
    def __init__(self, m, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.lift = nn.Conv1d(5 + 1, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))

    def forward(self, p, idx, T):
        N = int(round(T / p["dt"]))
        x = p["x"][idx]
        h = self.lift(torch.cat([x[:, :, None].expand(-1, -1, N), torch.ones(len(idx), 1, N)], dim=1))
        for c in self.convs:
            h = h + nn.functional.gelu(c(h))
        return self.proj(h).transpose(1, 2)  # (B, N, m)


def train(model, p, train_T, epochs, lr, batch, wd=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if wd > 0 else torch.optim.Adam(model.parameters(), lr=lr)
    steps = int(np.ceil(len(p["tr"]) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=lr * 1e-3)
    best, best_state, v = float("inf"), copy.deepcopy(model.state_dict()), float("nan")
    for ep in range(1, epochs + 1):
        perm = p["tr"][torch.randperm(len(p["tr"]), device=p["tr"].device)]
        for b in range(steps):
            idx = perm[b * batch : (b + 1) * batch]
            opt.zero_grad()
            loss = sum((rel(model(p, idx, T), p[f"Y{T}"][idx]) ** 2).mean() for T in train_T) / len(train_T)
            loss.backward()
            opt.step()
            sched.step()
        if ep % max(1, epochs // 40) == 0 or ep == epochs:
            with torch.no_grad():
                v = float(np.mean([float(rel(model(p, p["va"], T), p[f"Y{T}"][p["va"]]).median()) for T in train_T]))
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
    ap.add_argument("--model", choices=["turnpike", "cnn"], required=True)
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=420)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true", help="float32 (consumer GPUs have very slow float64)")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None, help="directory for test predictions u(s,t) at T = 1, 8 (float32)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float32 if args.fp32 else torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load(args.data)
    if args.fp32:
        p = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in p.items()}
    te = p["te"]
    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = MITurnpike(p["m"], width=args.width) if args.model == "turnpike" else CNNmi(p["m"])
        n_par = sum(q.numel() for q in model.parameters())
        t0 = time.perf_counter()
        print(f"{args.model} seed {seed} ({n_par} params, train T={args.train_T})", flush=True)
        model = train(model, p, args.train_T, args.epochs, args.lr, args.batch)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            res = {str(T): summarize(rel(model(p, te, T), p[f"Y{T}"][te]).cpu().numpy()) for T in T_EVAL}
            one = te[:1]
            model(p, one, 8)
            t1 = time.perf_counter()
            for _ in range(20):
                model(p, one, 8)
            infer = (time.perf_counter() - t1) / 20
            if args.save_pred is not None:
                args.save_pred.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_pred / f"pred_{args.model}_seed{seed}.npz", test_idx=te.cpu().numpy(),
                         **{f"U_T{T}": (model(p, te, T) * p["rho"][te, None, None]).cpu().numpy().astype(np.float32) for T in (1, 8)})
        runs.append({"seed": seed, "params": n_par, "train_seconds": secs, "infer_single_query_T8_s": infer, "test": res})
        print(f"  {args.model}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in T_EVAL)
              + f" ({secs:.0f}s)", flush=True)
    rep = {"model": args.model, "train_T": args.train_T, "runs": runs,
           "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in T_EVAL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

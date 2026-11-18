#!/usr/bin/env python3
"""Competing multi-region targets (m = 99): tau-linear turnpike vs generic turnpike vs CNN; train T = 1, test T = 1..8.

tau_turnpike   u = sum_k tau_k [ u_bar_k(s) + sig_k ( sum_i a_ki(s) e^{-lam_i t} + sum_i b_ki(s) e^{-lam_i (T-t)} ) ]
               encoder input: geometry + alpha + beta only (xa_1..3, log alpha, log beta); tau enters exactly.
               u_bar_k: closed-form per-region static optima; sig_k = RMS(u_bar_k); rates >= lambda_1, shared.
turnpike       u = u_bar(s) + rho ( sum a_i(s) e^{-lam_i t} + sum b_i(s) e^{-lam_i (T-t)} ), encoder sees all 8 inputs.
cnn            dilated CNN without coordinates, all 8 inputs broadcast, m output channels.
Targets and outputs are normalized by rho = RMS(u_bar) per sample; errors are relative L2 over (t, s).
"""

from __future__ import annotations

import argparse
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
from mi_turnpike_surrogate import rel, train  # noqa: E402
from turnpike_surrogate import grid, inv_softplus, summarize  # noqa: E402

T_EVAL = [1, 2, 4, 8]
GEO = [0, 1, 2, 6, 7]


def load(path):
    d = np.load(path)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    X = d["X"]
    Xn = (X - X[tr].mean(0)) / X[tr].std(0)
    ub, ubk = d["u_bar"], d["u_bar_k"]
    rho = np.maximum(np.sqrt(np.mean(ub**2, axis=1)), 1e-12)
    sig = np.maximum(np.sqrt(np.mean(ubk**2, axis=2)), 1e-12)  # (n, K)
    geo = list(d["geo_cols"]) if "geo_cols" in d.files else GEO      # strip data store their own layout
    tau = d["coefs"] if "coefs" in d.files else X[:, 3:6]
    p = {"x": torch.tensor(Xn), "xg": torch.tensor(Xn[:, geo]), "tau": torch.tensor(tau), "floor": torch.tensor(d["lam1"]),
         "ubar_n": torch.tensor(ub / rho[:, None]), "ubk_n": torch.tensor(ubk / rho[:, None, None]),
         "sig_n": torch.tensor(sig / rho[:, None]), "rho": torch.tensor(rho), "dt": float(d["dt"]), "m": int(d["m"]),
         "tr": torch.tensor(tr), "va": torch.tensor(va), "te": torch.tensor(te)}
    for T in (1, 2):
        p[f"Y{T}"] = torch.tensor(d[f"U_T{T}"].astype(np.float64) / rho[:, None, None])
    for T in (4, 8):
        Yt = torch.zeros(len(X), *d[f"U_T{T}_test"].shape[1:], dtype=torch.float64)
        Yt[te] = torch.tensor(d[f"U_T{T}_test"].astype(np.float64) / rho[te, None, None])
        p[f"Y{T}"] = Yt
    return p


def rate_init(layer, r):
    lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
    with torch.no_grad():
        layer.weight.mul_(0.05)
        layer.bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))


class TauTurnpike(nn.Module):
    def __init__(self, m, K=3, r=4, width=256, n_geo=len(GEO)):
        super().__init__()
        self.m, self.K, self.r = m, K, r
        self.enc = nn.Sequential(nn.Linear(n_geo, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rate = nn.Linear(width, r)
        self.amp = nn.Linear(width, K * 2 * r * m)
        rate_init(self.rate, r)
        with torch.no_grad():
            self.amp.weight.mul_(0.05)
            self.amp.bias.zero_()

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        h = self.enc(p["xg"][idx])
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(h))
        ab = self.amp(h).view(-1, self.K, 2, self.r, self.m) * p["sig_n"][idx][:, :, None, None, None]
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        tau = p["tau"][idx]
        A0 = torch.einsum("bk,bkim->bim", tau, ab[:, :, 0])
        A1 = torch.einsum("bk,bkim->bim", tau, ab[:, :, 1])
        plateau = torch.einsum("bk,bkm->bm", tau, p["ubk_n"][idx])
        return plateau[:, None, :] + torch.einsum("bin,bim->bnm", E, A0) + torch.einsum("bin,bim->bnm", F, A1)


class GenericTurnpike(nn.Module):
    def __init__(self, m, n_in=8, r=4, width=256):
        super().__init__()
        self.m, self.r = m, r
        self.enc = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rate = nn.Linear(width, r)
        self.amp = nn.Linear(width, 2 * r * m)
        rate_init(self.rate, r)
        with torch.no_grad():
            self.amp.weight.mul_(0.05)
            self.amp.bias.zero_()

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        h = self.enc(p["x"][idx])
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(h))
        ab = self.amp(h).view(-1, 2, self.r, self.m)
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        return p["ubar_n"][idx][:, None, :] + torch.einsum("bin,bim->bnm", E, ab[:, 0]) + torch.einsum("bin,bim->bnm", F, ab[:, 1])


class HybridTauTurnpike(nn.Module):
    """Semilinear case: exact nonlinear plateau u_bar(s) (static OCP) + layers whose amplitudes are tau-linear (exact as
    gamma -> 0; encoder on geometry/alpha/beta/gamma) plus a small correction from all inputs; shared rates >= lam1."""

    def __init__(self, m, K, n_geo, n_in, r=4, width=256):
        super().__init__()
        self.m, self.K, self.r = m, K, r
        self.enc = nn.Sequential(nn.Linear(n_geo, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.enc_all = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rate = nn.Linear(width, r)
        self.amp = nn.Linear(width, K * 2 * r * m)
        self.corr = nn.Linear(width, 2 * r * m)
        rate_init(self.rate, r)
        with torch.no_grad():
            for lin in (self.amp, self.corr):
                lin.weight.mul_(0.05)
                lin.bias.zero_()

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        h = self.enc(p["xg"][idx])
        lam = p["floor"][idx, None] + nn.functional.softplus(self.rate(h))
        ab = self.amp(h).view(-1, self.K, 2, self.r, self.m) * p["sig_n"][idx][:, :, None, None, None]
        cr = self.corr(self.enc_all(p["x"][idx])).view(-1, 2, self.r, self.m)
        tau = p["tau"][idx]
        A0 = torch.einsum("bk,bkim->bim", tau, ab[:, :, 0]) + cr[:, 0]
        A1 = torch.einsum("bk,bkim->bim", tau, ab[:, :, 1]) + cr[:, 1]
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        return p["ubar_n"][idx][:, None, :] + torch.einsum("bin,bim->bnm", E, A0) + torch.einsum("bin,bim->bnm", F, A1)


class CNNmulti(nn.Module):
    def __init__(self, m, n_in=8, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.lift = nn.Conv1d(n_in + 1, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))

    def forward(self, p, idx, T):
        N = int(round(T / p["dt"]))
        x = p["x"][idx]
        h = self.lift(torch.cat([x[:, :, None].expand(-1, -1, N), torch.ones(len(idx), 1, N)], dim=1))
        for c in self.convs:
            h = h + nn.functional.gelu(c(h))
        return self.proj(h).transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=["tau_turnpike", "turnpike", "hybrid_tau", "cnn"], required=True)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=560)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None, help="directory for test predictions u(s,t) at T = 1, 8 (float32)")
    ap.add_argument("--r", type=int, default=4, help="number of exponential modes (turnpike models)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float32 if args.fp32 else torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load(args.data)
    if args.fp32:
        p = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in p.items()}
    te = p["te"]
    K, n_geo, n_in = p["tau"].shape[1], p["xg"].shape[1], p["x"].shape[1]
    make = {"tau_turnpike": lambda: TauTurnpike(p["m"], K=K, r=args.r, n_geo=n_geo), "turnpike": lambda: GenericTurnpike(p["m"], n_in=n_in, r=args.r),
            "hybrid_tau": lambda: HybridTauTurnpike(p["m"], K=K, n_geo=n_geo, n_in=n_in, r=args.r),
            "cnn": lambda: CNNmulti(p["m"], n_in=n_in)}[args.model]
    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = make()
        n_par = sum(q.numel() for q in model.parameters())
        t0 = time.perf_counter()
        print(f"{args.model} seed {seed} ({n_par} params, train T={args.train_T})", flush=True)
        model = train(model, p, args.train_T, args.epochs, args.lr, args.batch, wd=args.weight_decay)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            res = {str(T): summarize(rel(model(p, te, T), p[f"Y{T}"][te]).cpu().numpy()) for T in T_EVAL}
            if args.save_pred is not None:
                args.save_pred.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_pred / f"pred_{args.model}_r{args.r}_seed{seed}.npz", test_idx=te.cpu().numpy(),
                         **{f"U_T{T}": (model(p, te, T) * p["rho"][te, None, None]).cpu().numpy().astype(np.float32) for T in (1, 8)})
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

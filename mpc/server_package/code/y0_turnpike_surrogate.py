#!/usr/bin/env python3
"""Turnpike surrogates with a nonzero initial state y0 (data from gen_y0_data.py).

    u(t) / u_bar = 1 + sum_i a_i e^{-lam_i t} + sum_i b_i e^{-kap_i (T - t)}
Plateau u_bar(mu) and the terminal layer (b, kap) depend on mu only (V4: the terminal layer is y0-independent);
the initial layer depends on z = cosine coefficients of y0 - y_bar(mu). Rates >= lam1 (stored floor).
    tp_lin   a = a0(mu) + A(mu) z, lam = lam(mu)        (exact structure of the LQ problem: a learned rank-r feedback)
    tp_hyb   tp_lin + MLP(mu, z) correction of a        (linear part kept; nonlinear correction for the semilinear problem)
    tp_mlp   (a, lam) = MLP(mu, z)                      (general; needed for the semilinear problem)
    tp_noz   tp_mlp with z = 0                          (ablation: ignores y0)
    cnn      dilated CNN in time, no coordinates, input [mu, z] broadcast + ones
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


def load(path, floor_init="lam1"):
    d = np.load(path)
    tr = d["train_idx"]
    X, Z = d["X"], d["Z"]
    norm = {"x_mean": X[tr].mean(0), "x_std": X[tr].std(0), "z_scale": np.sqrt((Z[tr] ** 2).mean(0))}
    p = {"dt": float(d["dt"]), "tr": torch.tensor(tr), "va": torch.tensor(d["val_idx"]), "te": torch.tensor(d["test_idx"]),
         "x": torch.tensor((X - norm["x_mean"]) / norm["x_std"]), "z": torch.tensor(Z / norm["z_scale"]),
         "scale": torch.tensor(d["u_bar"]), "floor": torch.tensor(d["lam1"]), "norm": norm}
    p["Y"] = {T: torch.tensor(d[f"U_T{T}"]) / p["scale"][:, None] for T in T_EVAL}
    if floor_init == "open":  # initial layer far from the turnpike: open-loop lambda_1(beta) as its floor
        b_u, inv = np.unique(d["beta"], return_inverse=True)
        fo = np.array([first_eig(b) for b in b_u])[inv]
        p["floor"] = torch.stack([torch.tensor(fo), p["floor"]], 1)
    return p


def rate_head(width, r):
    lin = nn.Linear(width, r)
    lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
    with torch.no_grad():
        lin.weight.mul_(0.05)
        lin.bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))
    return lin


def amp_head(width, k):
    lin = nn.Linear(width, k)
    with torch.no_grad():
        lin.weight.mul_(0.05)
        lin.bias.zero_()
    return lin


def enc(i, width):
    return nn.Sequential(nn.Linear(i, width), nn.GELU(), nn.Linear(width, width), nn.GELU())


class Y0Turnpike(nn.Module):
    def __init__(self, n_mu, n_z, r=8, width=128, mode="lin"):
        super().__init__()
        self.r, self.n_z, self.mode = r, n_z, mode
        self.enc_mu = enc(n_mu, width)
        self.kap, self.b = rate_head(width, r), amp_head(width, r)
        if mode in ("lin", "hyb"):
            self.lam, self.a0, self.A = rate_head(width, r), amp_head(width, r), amp_head(width, r * n_z)
            if mode == "hyb":
                self.enc_init, self.a_corr = enc(n_mu + n_z, width), amp_head(width, r)
        else:
            self.enc_init = enc(n_mu + n_z, width)
            self.lam, self.a = rate_head(width, r), amp_head(width, r)

    def forward(self, x, z, t, s, fl):
        # fl (B,): one floor; fl (B, 2): (initial-layer floor, terminal-layer floor)
        fl_i, fl = (fl[:, 0], fl[:, 1]) if fl.dim() == 2 else (fl, fl)
        h = self.enc_mu(x)
        kap = fl[:, None] + nn.functional.softplus(self.kap(h))
        b = self.b(h)
        if self.mode in ("lin", "hyb"):
            lam = fl_i[:, None] + nn.functional.softplus(self.lam(h))
            a = self.a0(h) + torch.einsum("brk,bk->br", self.A(h).view(-1, self.r, self.n_z), z)
            if self.mode == "hyb":
                a = a + self.a_corr(self.enc_init(torch.cat([x, z], 1)))
        else:
            zz = torch.zeros_like(z) if self.mode == "noz" else z
            g = self.enc_init(torch.cat([x, zz], 1))
            lam = fl_i[:, None] + nn.functional.softplus(self.lam(g))
            a = self.a(g)
        L0 = torch.einsum("bi,bin->bn", a, torch.exp(-lam[:, :, None] * t[None, None, :]))
        L1 = torch.einsum("bi,bin->bn", b, torch.exp(-kap[:, :, None] * s[None, None, :]))
        return 1.0 + L0 + L1


class CNNz(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = DilatedCNN(n_in)

    def forward(self, x, z, t, s, fl):
        xz = torch.cat([x, z], 1)
        return self.net(xz, torch.ones(len(xz), len(t)), t, s)


def predict(model, p, idx, T):
    t, s = grid(T, p["dt"])
    return model(p["x"][idx], p["z"][idx], t, s, p["floor"][idx])


def train(model, p, train_T, epochs, lr, batch):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = int(np.ceil(len(p["tr"]) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=lr * 1e-3)
    best, best_state, v = float("inf"), copy.deepcopy(model.state_dict()), float("nan")
    for ep in range(1, epochs + 1):
        perm = p["tr"][torch.randperm(len(p["tr"]), device=p["tr"].device)]
        for b in range(steps):
            idx = perm[b * batch : (b + 1) * batch]
            opt.zero_grad()
            loss = sum((rel_err(predict(model, p, idx, T), p["Y"][T][idx]) ** 2).mean() for T in train_T) / len(train_T)
            loss.backward()
            opt.step()
            sched.step()
        if ep % max(1, epochs // 40) == 0 or ep == epochs:
            with torch.no_grad():
                v = float(np.mean([float(rel_err(predict(model, p, p["va"], T), p["Y"][T][p["va"]]).median()) for T in train_T]))
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
    ap.add_argument("--model", choices=["tp_lin", "tp_hyb", "tp_mlp", "tp_noz", "cnn"], required=True)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None)
    ap.add_argument("--save-model", type=pathlib.Path, default=None, help="directory for state_dict + normalization (per seed)")
    ap.add_argument("--floor-init", choices=["lam1", "open"], default="lam1")
    ap.add_argument("--tag", default=None, help="file-name tag for predictions / models (default: model name)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float32 if args.fp32 else torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load(args.data, args.floor_init)
    tag = args.tag or args.model
    if args.fp32:
        p = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in p.items()}
        p["Y"] = {T: v.float() for T, v in p["Y"].items()}
    te = p["te"]
    n_mu, n_z = p["x"].shape[1], p["z"].shape[1]
    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        if args.model == "cnn":
            model = CNNz(n_mu + n_z)
        else:
            model = Y0Turnpike(n_mu, n_z, r=args.r, mode={"tp_lin": "lin", "tp_hyb": "hyb", "tp_mlp": "mlp", "tp_noz": "noz"}[args.model])
        n_par = sum(q.numel() for q in model.parameters())
        print(f"{args.model} seed {seed} ({n_par} params, train T={args.train_T})", flush=True)
        t0 = time.perf_counter()
        model = train(model, p, args.train_T, args.epochs, args.lr, args.batch)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            res = {str(T): summarize(rel_err(predict(model, p, te, T), p["Y"][T][te]).cpu().double().numpy()) for T in T_EVAL}
            if args.save_pred is not None:
                args.save_pred.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_pred / f"pred_{tag}_seed{seed}.npz", test_idx=te.cpu().numpy(),
                         **{f"U_T{T}": (predict(model, p, te, T) * p["scale"][te, None]).cpu().double().numpy() for T in T_EVAL})
        if args.save_model is not None:
            args.save_model.mkdir(parents=True, exist_ok=True)
            torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "norm": p["norm"], "model": args.model,
                        "r": args.r, "n_mu": n_mu, "n_z": n_z, "fp32": args.fp32, "floor_init": args.floor_init},
                       args.save_model / f"model_{tag}_seed{seed}.pt")
        runs.append({"seed": seed, "params": n_par, "train_seconds": secs, "test": res})
        print(f"  {args.model}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in T_EVAL)
              + f" ({secs:.0f}s)", flush=True)
    rep = {"model": args.model, "tag": tag, "floor_init": args.floor_init, "r": args.r, "train_T": args.train_T, "epochs": args.epochs, "runs": runs,
           "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in T_EVAL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

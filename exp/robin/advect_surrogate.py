#!/usr/bin/env python3
"""Surrogates for the downstream-observation advection dataset (gen_advect_data.py), u(s, t), m = 99.
Trained on T = 2 only, evaluated on T = 2 and T = 4 test sets. All outputs are normalized by rho = RMS of the exact
periodic turnpike P (harmonic balance, computed per sample like an input).

    tp_transport  u = P + w(t) L_I(s, t) + w(T - t) L_E(s, T - t): exact periodic part + two LOCAL, FINITE-WINDOW
                  layers (transport: the boundary layers last about one transit time and do not decay exponentially).
                  L_I / L_E: 2D CNNs over (time in window, s), dilated along time (receptive field ~ transit time), with
                  channels [P, target, bx(s), transit time tau(s), time in window, mu]; window length tau_w, cosine taper.
    per_only      u = P (no learning)
    cnn           dilated 1D CNN over t; channels mu, cos/sin theta, the target at the m nodes, ones (no P)
    cnn_per       the same with P as m extra input channels, output added to P (generic correction of the exact part)
    deeponet      branch mu -> p coefficients; trunk (s, cos theta, sin theta, t / 2, (T - t) / 2) -> p basis values
    fno           2D FNO over (s, t), channels mu, target, cos/sin theta, t / 2, (T - t) / 2
mu = (gamma, log10 eps, cos phi0, sin phi0, a, log10 sigma), standardized on the training set.
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
from turnpike_surrogate import summarize  # noqa: E402

T_TRAIN, T_EVAL = 2, (2, 4)


def rel(pred, y):
    return torch.linalg.norm((pred - y).flatten(1), dim=1) / torch.linalg.norm(y.flatten(1), dim=1)


def tgrid(T, dt):
    N = int(round(T / dt))
    return torch.arange(1, N + 1) * dt


def load(path, device):
    d = np.load(path)
    fix = json.loads(str(d["fixed"]))
    X = d["X"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    feat = np.stack([X[:, 0], X[:, 1], np.cos(X[:, 2]), np.sin(X[:, 2]), X[:, 3], X[:, 4]], 1)
    nH = int(d["n_harm"].max()) + 1
    uhat = torch.tensor(d["uhat"][:, :nH].astype(np.complex128))
    phi0 = torch.tensor(X[:, 2])
    omega = torch.full((len(X),), float(fix["omega"]), dtype=torch.float64)
    dt, m = float(d["dt"]), int(d["m"])
    sg = torch.arange(1, m + 1, dtype=torch.float64) / (m + 1)
    bx = float(fix["bx"]) * (1.0 + torch.tensor(X[:, 0])[:, None] * (sg[None, :] - 0.5))  # (n, m)
    p = {"mu": torch.tensor((feat - feat[tr].mean(0)) / feat[tr].std(0)), "phi0": phi0, "omega": omega, "dt": dt, "m": m,
         "a": torch.tensor(X[:, 3]), "sigma": torch.tensor(10.0 ** X[:, 4]), "bx": bx / float(fix["bx"]),
         "tau": (float(fix["xo"]) + 0.5 * float(fix["w"])) / bx,  # transit time from the edge to the strip centre
         "tr": torch.tensor(tr), "va": torch.tensor(va), "te": torch.tensor(te), "row": {}, "P": {}, "Y": {}, "U": {}}
    t2 = tgrid(T_TRAIN, dt)
    rho = torch.sqrt((torch.cat([periodic(uhat[k:k + 50], omega[k:k + 50], phi0[k:k + 50], t2) for k in range(0, len(X), 50)]) ** 2)
                     .mean(dim=(1, 2)))
    p["rho"] = rho
    p["uhat"] = uhat.to(torch.complex64)
    for T in T_EVAL:
        t = tgrid(T, dt)
        rows = torch.arange(len(X)) if T == T_TRAIN else torch.tensor(te)
        rowmap = torch.full((len(X),), -1, dtype=torch.long)
        rowmap[rows] = torch.arange(len(rows))
        p["row"][T] = rowmap
        U = torch.tensor((d["U_T2"] if T == T_TRAIN else d[f"U_T{T}_test"]).astype(np.float64))
        p["U"][T] = U
        p["Y"][T] = U / rho[rows, None, None]
        p["P"][T] = torch.cat([periodic(uhat[rows[k:k + 50]], omega[rows[k:k + 50]], phi0[rows[k:k + 50]], t)
                               for k in range(0, len(rows), 50)]) / rho[rows, None, None]
    conv = lambda v: v.float().to(device) if isinstance(v, torch.Tensor) and v.is_floating_point() else (  # noqa: E731
        v.to(device) if isinstance(v, torch.Tensor) else v)
    return {k: ({kk: conv(vv) for kk, vv in v.items()} if isinstance(v, dict) else conv(v)) for k, v in p.items()}


def periodic_dev(uhat, omega, phi0, t):
    """moving_surrogate.periodic on t's device."""
    n = torch.arange(uhat.shape[1], dtype=t.dtype, device=t.device)
    E = torch.exp(1j * (omega[:, None, None] * t[None, :, None] + phi0[:, None, None]) * n[None, None, :])
    E[:, :, 1:] *= 2.0
    return torch.real(torch.einsum("bnh,bhm->bnm", E, uhat))


def target_field(p, idx, t):
    """Moving hot spot at the m control nodes: (B, N, m)."""
    sg = torch.arange(1, p["m"] + 1, device=t.device, dtype=t.dtype) / (p["m"] + 1)
    th = p["omega"][idx, None] * t[None, :] + p["phi0"][idx, None]
    c = 0.5 + p["a"][idx, None] * torch.sin(th)
    return torch.exp(-0.5 * ((sg[None, None, :] - c[:, :, None]) / p["sigma"][idx, None, None]) ** 2), th


def taper(tau, tau_w, delta):
    """1 for tau <= tau_w - delta, cosine down to 0 at tau_w."""
    x = torch.clamp((tau - (tau_w - delta)) / delta, 0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * x))


class LayerNet(nn.Module):
    def __init__(self, c_in, width=32, dil_t=(1, 2, 4, 8, 16, 32, 64), dil_s=(1, 2, 4, 1, 2, 4, 1)):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(width, width, 3, padding=(dt_, ds), dilation=(dt_, ds)) for dt_, ds in zip(dil_t, dil_s)])
        self.out = nn.Conv2d(width, 1, 1)
        with torch.no_grad():
            self.out.weight.mul_(0.05)
            self.out.bias.zero_()

    def forward(self, h):  # (B, C, Nw, m) -> (B, Nw, m)
        h = nn.functional.gelu(self.lift(h))
        for conv in self.blocks:
            h = h + nn.functional.gelu(conv(h))
        return self.out(h)[:, 0]


class TransportTP(nn.Module):
    """mirror: the windows also see the periodic control just OUTSIDE the horizon, P(-tau) before the start and P(T + tau)
    after the end. At t = 0 the domain is empty while the periodic orbit assumes the heat injected during the last transit
    time is already in flight; the initial burst replaces that missing heat (symmetrically for the end)."""

    def __init__(self, m, tau_w=1.0, delta=0.1, width=32, mirror=False):
        super().__init__()
        self.m, self.tau_w, self.delta, self.mirror = m, tau_w, delta, mirror
        c_in = 5 + 6 + int(mirror)
        self.net_i, self.net_e = LayerNet(c_in, width), LayerNet(c_in, width)

    def window(self, net, p, idx, P, g, tw, extra):
        B, Nw, m = P.shape
        ch = [P, g, p["bx"][idx, None, :].expand(B, Nw, m), p["tau"][idx, None, :].expand(B, Nw, m), tw[None, :, None].expand(B, Nw, m)]
        ch += [p["mu"][idx, k, None, None].expand(B, Nw, m) for k in range(p["mu"].shape[1])] + extra
        return net(torch.stack(ch, 1))

    def forward(self, p, idx, T):
        t = tgrid(T, p["dt"]).to(p["mu"])
        P = p["P"][T][p["row"][T][idx]]
        g, _ = target_field(p, idx, t)
        Nw = int(round(self.tau_w / p["dt"]))
        ti, te_ = t[:Nw], (T - t)[-Nw:].flip(0)  # time since start / time to the end, both increasing
        xi, xe = [], []
        if self.mirror:
            sc = p["rho"][idx, None, None]
            xi = [periodic_dev(p["uhat"][idx], p["omega"][idx], p["phi0"][idx], -ti) / sc]
            xe = [periodic_dev(p["uhat"][idx], p["omega"][idx], p["phi0"][idx], T + te_) / sc]
        Li = self.window(self.net_i, p, idx, P[:, :Nw], g[:, :Nw], ti, xi) * taper(ti, self.tau_w, self.delta)[None, :, None]
        Le = self.window(self.net_e, p, idx, P[:, -Nw:].flip(1), g[:, -Nw:].flip(1), te_, xe) * taper(te_, self.tau_w, self.delta)[None, :, None]
        out = P.clone()
        out[:, :Nw] = out[:, :Nw] + Li
        out[:, -Nw:] = out[:, -Nw:] + Le.flip(1)
        return out


class PerOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, p, idx, T):
        return p["P"][T][p["row"][T][idx]] + 0 * self.dummy


class CNNt(nn.Module):
    def __init__(self, m, use_P, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.m, self.use_P = m, use_P
        self.lift = nn.Conv1d(6 + 2 + m + 1 + (m if use_P else 0), width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=dd, dilation=dd) for dd in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))
        if use_P:
            with torch.no_grad():
                self.proj[-1].weight.mul_(0.05)
                self.proj[-1].bias.zero_()

    def forward(self, p, idx, T):
        t = tgrid(T, p["dt"]).to(p["mu"])
        B, N = len(idx), len(t)
        g, th = target_field(p, idx, t)
        h = [p["mu"][idx, :, None].expand(-1, -1, N), torch.cos(th)[:, None], torch.sin(th)[:, None], g.transpose(1, 2),
             torch.ones(B, 1, N, device=t.device)]
        if self.use_P:
            P = p["P"][T][p["row"][T][idx]]
            h.append(P.transpose(1, 2))
        h = self.lift(torch.cat(h, 1))
        for conv in self.convs:
            h = h + nn.functional.gelu(conv(h))
        out = self.proj(h).transpose(1, 2)
        return out + P if self.use_P else out


class DeepONet(nn.Module):
    def __init__(self, pdim=128, width=256):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, pdim))
        self.trunk = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, pdim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, p, idx, T):
        t = tgrid(T, p["dt"]).to(p["mu"])
        B, N, m = len(idx), len(t), p["m"]
        sg = torch.arange(1, m + 1, device=t.device, dtype=t.dtype) / (m + 1)
        th = p["omega"][idx, None] * t[None, :] + p["phi0"][idx, None]
        z = torch.stack([sg[None, None, :].expand(B, N, m), torch.cos(th)[..., None].expand(B, N, m), torch.sin(th)[..., None].expand(B, N, m),
                         (t / 2)[None, :, None].expand(B, N, m), ((T - t) / 2)[None, :, None].expand(B, N, m)], -1)
        return torch.einsum("bnmk,bk->bnm", self.trunk(z), self.branch(p["mu"][idx])) + self.bias


class SpectralConv2d(nn.Module):
    def __init__(self, ci, co, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        sc = 1.0 / (ci * co)
        self.w1 = nn.Parameter(sc * torch.randn(ci, co, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(sc * torch.randn(ci, co, m1, m2, dtype=torch.cfloat))

    def forward(self, x):  # (B, C, H = m, W = N)
        xh = torch.fft.rfft2(x)
        out = torch.zeros(x.shape[0], self.w1.shape[1], x.shape[-2], x.shape[-1] // 2 + 1, dtype=xh.dtype, device=x.device)
        m2 = min(self.m2, xh.shape[-1])
        out[:, :, : self.m1, :m2] = torch.einsum("bixy,ioxy->boxy", xh[:, :, : self.m1, :m2], self.w1[..., :m2])
        out[:, :, -self.m1 :, :m2] = torch.einsum("bixy,ioxy->boxy", xh[:, :, -self.m1 :, :m2], self.w2[..., :m2])
        return torch.fft.irfft2(out, s=x.shape[-2:])


class FNO2d(nn.Module):
    def __init__(self, width=32, m1=12, m2=32, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(6 + 5, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, m1, m2) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 4 * width, 1), nn.GELU(), nn.Conv2d(4 * width, 1, 1))

    def forward(self, p, idx, T):
        t = tgrid(T, p["dt"]).to(p["mu"])
        B, N, m = len(idx), len(t), p["m"]
        g, th = target_field(p, idx, t)
        ch = [p["mu"][idx, k, None, None].expand(B, m, N) for k in range(p["mu"].shape[1])]
        ch += [g.transpose(1, 2), torch.cos(th)[:, None, :].expand(B, m, N), torch.sin(th)[:, None, :].expand(B, m, N),
               (t / 2)[None, None, :].expand(B, m, N), ((T - t) / 2)[None, None, :].expand(B, m, N)]
        h = self.lift(torch.stack(ch, 1))
        h = nn.functional.pad(h, (0, 64, 0, 8))  # non-periodic padding in t and s
        for k, (sc, sk) in enumerate(zip(self.spec, self.skip)):
            h = sc(h) + sk(h)
            if k < len(self.spec) - 1:
                h = nn.functional.gelu(h)
        return self.proj(h[..., :m, :N])[:, 0].transpose(1, 2)


def train(model, p, epochs, lr, batch, log_every):
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
            loss = (rel(model(p, idx, T_TRAIN), p["Y"][T_TRAIN][p["row"][T_TRAIN][idx]]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        if ep % max(1, epochs // 40) == 0 or ep == epochs:
            with torch.no_grad():
                va = p["va"]
                v = float(torch.cat([rel(model(p, va[k:k + batch], T_TRAIN), p["Y"][T_TRAIN][p["row"][T_TRAIN][va[k:k + batch]]])
                                     for k in range(0, len(va), batch)]).median())
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
        if ep % log_every == 0:
            print(f"    ep {ep} loss {float(loss.detach()):.3e} val {v:.3e}", flush=True)
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=["tp_transport", "per_only", "cnn", "cnn_per", "deeponet", "fno"], required=True)
    ap.add_argument("--tau-w", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=32, help="tp_transport layer-CNN width")
    ap.add_argument("--mirror", action="store_true", help="tp_transport: also feed P(-tau) / P(T + tau) (in-flight heat)")
    ap.add_argument("--tag", default="", help="suffix for the output / prediction names")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    p = load(args.data, args.device)
    te = p["te"]
    runs = []
    for seed in (args.seeds if args.model != "per_only" else [0]):
        torch.manual_seed(seed)
        model = {"tp_transport": lambda: TransportTP(p["m"], tau_w=args.tau_w, width=args.width, mirror=args.mirror), "per_only": PerOnly,
                 "cnn": lambda: CNNt(p["m"], False), "cnn_per": lambda: CNNt(p["m"], True), "deeponet": DeepONet,
                 "fno": FNO2d}[args.model]().to(args.device)
        n_par = sum(q.numel() * (2 if q.is_complex() else 1) for q in model.parameters())
        print(f"{args.model} seed {seed} ({n_par} params, train T = {T_TRAIN})", flush=True)
        t0 = time.perf_counter()
        model = train(model, p, args.epochs if args.model != "per_only" else 0, args.lr, args.batch, max(1, args.epochs // 6))
        secs = time.perf_counter() - t0
        res, preds = {}, {}
        with torch.no_grad():
            for T in T_EVAL:
                out = torch.cat([model(p, te[k : k + 8], T) for k in range(0, len(te), 8)])
                rows = p["row"][T][te]
                u = out * p["rho"][te, None, None]
                res[str(T)] = summarize(rel(u, p["U"][T][rows]).cpu().double().numpy())
                preds[f"U_T{T}"] = u.cpu().numpy().astype(np.float32)
                if T == 4 and args.model == "tp_transport":  # window floor: the exact layers cut to the windows
                    t = tgrid(T, p["dt"]).to(p["mu"])
                    w = taper(t, args.tau_w, model.delta) + taper(T - t, args.tau_w, model.delta)
                    Ue, Pe = p["U"][T][rows], p["P"][T][rows] * p["rho"][te, None, None]
                    res["window_floor_T4"] = summarize(rel(Pe + (Ue - Pe) * w[None, :, None], Ue).cpu().double().numpy())
        if args.save_pred is not None:
            args.save_pred.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_pred / f"pred_{args.model}{args.tag}_seed{seed}.npz", test_idx=te.cpu().numpy(), **preds)
        runs.append({"seed": seed, "params": n_par, "train_seconds": secs, "test": res})
        print(f"  {args.model}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} p90 {res[str(T)]['p90']:.2e} max {res[str(T)]['max']:.2e}"
                                         for T in T_EVAL) + f" ({secs:.0f}s)", flush=True)
    rep = {"model": args.model, "tau_w": args.tau_w, "epochs": args.epochs, "runs": runs,
           "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in T_EVAL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

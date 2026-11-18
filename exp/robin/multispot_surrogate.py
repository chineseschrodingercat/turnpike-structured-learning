#!/usr/bin/env python3
"""Surrogates for the two-spot quasi-periodic target (gen_multispot_data.py), u(s, t), m = 99, linear (--kind lin) or
semilinear (--kind nl) data. P = exact quasi-periodic LINEAR reference from the stored harmonics,
    P(t) = u0 + sum_j 2 Re sum_{k>=1} U_jk e^{i k theta_j(t)},  theta_j(t) = omega_j t + phi_j,
normalized by rho = RMS(P) on T = 1. Time enters only through theta_j(t), t and T - t.
    tp        u/rho = P + L_I + L_E                              (exact reference: the linear problem)
    tp_local  u/rho = P + Delta + L_I + L_E, Delta from a 1D CNN over s applied at every time to the local channels
              [P, dP/dt, target, mu] (moving-target tp_local with the phase derivative replaced by the time derivative)
    tp_tc     tp_local with end-snapshot layers and a Delta with temporal context: a 2D CNN over (t, s) with valid
              convolutions in t, applied to the inputs evaluated on the extended grid [-W dt, T + W dt] (they are analytic
              in t), so Delta stays time-shift equivariant and independent of T
    tp_tce    tp_tc plus the clipped distances to the ends, max(0, 1 - t/L) and max(0, 1 - (T - t)/L), as input channels,
              so the temporal CNN can also correct the end layers; for T >= 2 (L + W dt) no output sees both ends
    tp_tce_nolayer  tp_tce without the snapshot layers: the whole correction is the temporal CNN
    ref_only  u/rho = P
    cnn       dilated CNN in t, inputs mu, cos/sin theta_j(t), the target at the m nodes, ones
    cnn_res   cnn plus the channels P and dP/dt, output P + CNN
    cnn_res_tc  cnn_res made horizon-independent: valid convolutions in t on the extended grid plus the clipped
              distance-to-end channels of tp_tce (receptive half-width sum(tdil) instead of zero padding)
Layers: r exponential modes with rates lambda_1(beta) + softplus(.), amplitudes modulated by (1, cos theta_j, sin theta_j);
initial amplitudes from (mu, cos/sin phi_j), terminal ones from (mu, cos/sin theta_j(T)). Reports whole-horizon errors and
end-window errors (tau = 0.5) with transfer ratios T = 8 vs T = 1.
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
SPOT = 4  # first spot column; per spot: A, c, a, omega, log10_sigma, phi
TDIL = (1, 2, 4, 8, 16, 32)  # time dilations of the temporal correction; receptive half-width sum(TDIL) steps
SDIL = (1, 2, 4, 8, 1, 2, 4, 8)  # matching space dilations, block by block
EDGE_L = 0.3  # length of the clipped distance-to-end ramps of tp_tce


def rel(pred, y):
    return torch.linalg.norm((pred - y).flatten(1), dim=1) / torch.linalg.norm(y.flatten(1), dim=1)


def reference(u0, uhat, om, ph, t, deriv=False):
    """u0 (B, m), uhat (B, 2, K, m) complex, om/ph (B, 2), t (N,) -> (B, N, m)."""
    k = torch.arange(uhat.shape[2], dtype=t.dtype)
    out = torch.zeros(u0.shape[0], len(t), u0.shape[1], dtype=t.dtype) if deriv else u0[:, None, :].expand(-1, len(t), -1).clone()
    for j in range(uhat.shape[1]):
        th = om[:, j, None, None] * t[None, :, None] + ph[:, j, None, None]  # (B, N, 1)
        E = torch.exp(1j * th * k[None, None, :])  # (B, N, K)
        E[..., 0] = 0
        if deriv:
            E = E * (1j * k * om[:, j, None])[:, None, :]
        out = out + 2 * torch.real(torch.einsum("bnk,bkm->bnm", E, uhat[:, j]))
    return out


def load(path, kind, ext=0):
    """ext > 0 also stores P, dP/dt on the extended grid k = 1 - ext, ..., N + ext (k dt), for the temporal correction."""
    d = np.load(path)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    X = d["X"]
    cols = [0, 1, 2] + ([3] if kind == "nl" else []) + [SPOT + 6 * j + q for j in range(2) for q in range(5)]
    mu = X[:, cols]
    om = torch.tensor(X[:, [SPOT + 3, SPOT + 9]])
    ph = torch.tensor(X[:, [SPOT + 5, SPOT + 11]])
    K = int(d["nharm"].max()) + 1
    u0, uhat = torch.tensor(d["u0"]), torch.tensor(d["uhat"][:, :, :K].astype(np.complex128))
    dt = float(d["dt"])
    p = {"mu": torch.tensor((mu - mu[tr].mean(0)) / mu[tr].std(0)), "om": om, "ph": ph, "floor": torch.tensor(d["lam1"]),
         "A": torch.tensor(X[:, [SPOT, SPOT + 6]]), "c": torch.tensor(X[:, [SPOT + 1, SPOT + 7]]), "a": torch.tensor(X[:, [SPOT + 2, SPOT + 8]]),
         "sig": torch.tensor(10.0 ** X[:, [SPOT + 4, SPOT + 10]]), "dt": dt, "m": int(d["m"]), "n_mu": mu.shape[1],
         "tr": torch.tensor(tr), "va": torch.tensor(va), "te": torch.tensor(te), "row": {}, "P": {}, "dP": {}, "Y": {}, "U": {},
         "ext": ext, "Pext": {}, "dPext": {}}
    t1, _ = grid(1, dt)
    p["rho"] = torch.cat([torch.sqrt((reference(u0[b:b + 50], uhat[b:b + 50], om[b:b + 50], ph[b:b + 50], t1) ** 2).mean(dim=(1, 2)))
                          for b in range(0, len(X), 50)])
    for T in T_EVAL:
        t, _ = grid(T, dt)
        rows = torch.arange(len(X)) if T <= 2 else torch.tensor(te)
        rowmap = torch.full((len(X),), -1, dtype=torch.long)
        rowmap[rows] = torch.arange(len(rows))
        p["row"][T] = rowmap
        U = torch.tensor((d[f"U_{kind}_T{T}"] if T <= 2 else d[f"U_{kind}_T{T}_test"]).astype(np.float64))
        rh = p["rho"][rows, None, None]
        p["U"][T], p["Y"][T] = U, U / rh
        P, dP = [], []
        for b in range(0, len(rows), 50):
            r = rows[b:b + 50]
            P.append(reference(u0[r], uhat[r], om[r], ph[r], t))
            dP.append(reference(u0[r], uhat[r], om[r], ph[r], t, deriv=True))
        p["P"][T] = torch.cat(P) / rh
        p["dP"][T] = torch.cat(dP) / (20.0 * math.pi * rh)
        if ext > 0:
            te_ = torch.arange(1 - ext, len(t) + ext + 1, dtype=t.dtype) * dt
            P, dP = [], []
            for b in range(0, len(rows), 50):
                r = rows[b:b + 50]
                P.append(reference(u0[r], uhat[r], om[r], ph[r], te_))
                dP.append(reference(u0[r], uhat[r], om[r], ph[r], te_, deriv=True))
            p["Pext"][T] = torch.cat(P) / rh
            p["dPext"][T] = torch.cat(dP) / (20.0 * math.pi * rh)
    return p


def mlp(i, width, o, zero_last=True):
    net = nn.Sequential(nn.Linear(i, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, o))
    if zero_last:
        with torch.no_grad():
            net[-1].weight.mul_(0.05)
            net[-1].bias.zero_()
    return net


def phases(p, idx, t):
    return p["om"][idx, None, :] * t[None, :, None] + p["ph"][idx, None, :]  # (B, N, 2)


def target(p, idx, th, m):
    s = torch.arange(1, m + 1, dtype=th.dtype) / (m + 1)
    c = p["c"][idx, None, :] + p["a"][idx, None, :] * torch.sin(th)  # (B, N, 2)
    g = p["A"][idx, None, :, None] * torch.exp(-0.5 * ((s[None, None, None, :] - c[..., None]) / p["sig"][idx, None, :, None]) ** 2)
    return g.sum(2)  # (B, N, m)


class SpatialHead(nn.Module):
    """Layer amplitudes from end snapshots on the s grid: a 1D CNN over s, (B, c_in, m) -> (B, c_out, m)."""

    def __init__(self, c_in, c_out, width=64):
        super().__init__()
        self.lift = nn.Conv1d(c_in, width, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8)])
        self.out = nn.Conv1d(width, c_out, 1)
        with torch.no_grad():
            self.out.weight.mul_(0.05)
            self.out.bias.zero_()

    def forward(self, h):
        h = nn.functional.gelu(self.lift(h))
        for conv in self.blocks:
            h = h + nn.functional.gelu(conv(h))
        return self.out(h)


class TPMulti(nn.Module):
    def __init__(self, m, n_mu, local: bool, r=8, width_cnn=48, width_mlp=256, bnd=False, phase=False, tc=False, edge=False, layers=True,
                 tdil=TDIL, edge_L=EDGE_L):
        super().__init__()
        self.m, self.r, self.local, self.bnd, self.phase, self.tc, self.edge, self.layers = m, r, local, bnd, phase, tc, edge, layers
        self.tdil, self.edge_L = tuple(tdil), edge_L
        if local and tc:  # Delta with temporal context: 2D CNN over (t, s), valid in t, run on the extended grid
            self.lift = nn.Conv2d(3 + n_mu + 4 + (2 if edge else 0), width_cnn, (1, 5), padding=(0, 2))
            self.blocks = nn.ModuleList([nn.Conv2d(width_cnn, width_cnn, (3, 5), padding=(0, 2 * ds), dilation=(d, ds))
                                         for d, ds in zip(self.tdil, SDIL)])
            self.out = nn.Conv2d(width_cnn, 1, 1)
            with torch.no_grad():
                self.out.weight.mul_(0.05)
                self.out.bias.zero_()
        elif local:
            self.lift = nn.Conv1d(3 + n_mu + (4 if phase else 0), width_cnn, 5, padding=2)
            self.blocks = nn.ModuleList([nn.Conv1d(width_cnn, width_cnn, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8)])
            self.out = nn.Conv1d(width_cnn, 1, 1)
            with torch.no_grad():
                self.out.weight.mul_(0.05)
                self.out.bias.zero_()
        if not layers:
            return
        self.rate = mlp(n_mu, 128, r, zero_last=False)
        lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
        with torch.no_grad():
            self.rate[-1].weight.mul_(0.05)
            self.rate[-1].bias.copy_(inv_softplus(torch.clamp(lam0 - 3.0, min=0.5)))
        self.nb = 5
        if bnd:  # amplitudes from the snapshots (P, dP/dt, target) at t = dt (initial) and t = T (terminal), plus mu and phases
            self.lay_a = SpatialHead(3 + n_mu + 4, r * self.nb)
            self.lay_b = SpatialHead(3 + n_mu + 4, r * self.nb)
        else:
            self.lay_a = mlp(n_mu + 4, width_mlp, r * self.nb * m)
            self.lay_b = mlp(n_mu + 4, width_mlp, r * self.nb * m)

    def forward(self, p, idx, T):
        t, s = grid(T, p["dt"])
        mu = p["mu"][idx]
        B, N = len(idx), len(t)
        rows = p["row"][T][idx]
        P = p["P"][T][rows]
        th = phases(p, idx, t)
        per = P
        if self.local and self.tc:
            # inputs are analytic in t (through theta_j(t)), so they extend past [0, T]; valid convolutions in t then give
            # exactly N outputs with no padding, i.e. Delta is time-shift equivariant and horizon-independent
            W = sum(self.tdil)
            assert p["ext"] == W, "load(..., ext) must equal the receptive half-width sum(tdil)"
            te_ =torch.arange(1 - W, N + W + 1, dtype=t.dtype) * p["dt"]
            the = phases(p, idx, te_)  # (B, Ne, 2)
            Ne = len(te_)
            g = target(p, idx, the, self.m) / p["rho"][idx, None, None]
            trig = torch.cat([torch.cos(the), torch.sin(the)], -1).permute(0, 2, 1)[..., None].expand(-1, -1, -1, self.m)
            h = torch.cat([torch.stack([p["Pext"][T][rows], p["dPext"][T][rows], g], 1), mu[:, :, None, None].expand(-1, -1, Ne, self.m), trig], 1)
            if self.edge:  # clipped distances to t = 0 and t = T; they keep growing past the ends, so both sides are told apart
                L = self.edge_L
                ramp = torch.stack([torch.clamp(1 - te_ / L, min=0), torch.clamp(1 - (N * p["dt"] - te_) / L, min=0)])
                h = torch.cat([h, ramp[None, :, :, None].expand(B, -1, -1, self.m)], 1)
            h = nn.functional.gelu(self.lift(h))
            for conv, d in zip(self.blocks, self.tdil):
                h = h[:, :, d:-d] + nn.functional.gelu(conv(h))
            per = P + self.out(h)[:, 0]
        elif self.local:
            g = target(p, idx, th, self.m) / p["rho"][idx, None, None]
            h = torch.stack([P, p["dP"][T][rows], g], 2).reshape(B * N, 3, self.m)
            h = torch.cat([h, mu[:, None, :, None].expand(B, N, mu.shape[1], self.m).reshape(B * N, mu.shape[1], self.m)], 1)
            if self.phase:  # torus coordinates (theta_1, theta_2): the semilinear quasi-periodic optimum is a function of them
                tr = torch.cat([torch.cos(th), torch.sin(th)], -1).reshape(B * N, 4, 1).expand(-1, -1, self.m)
                h = torch.cat([h, tr], 1)
            h = nn.functional.gelu(self.lift(h))
            for conv in self.blocks:
                h = h + nn.functional.gelu(conv(h))
            per = P + self.out(h).view(B, N, self.m)
        if not self.layers:
            return per
        lam =p["floor"][idx, None] + nn.functional.softplus(self.rate(mu))
        Pl = torch.cat([torch.ones(B, N, 1, dtype=t.dtype), torch.cos(th), torch.sin(th)], -1)  # (B, N, 5)
        th0, thT = p["ph"][idx], p["om"][idx] * T + p["ph"][idx]
        z0, zT = torch.cat([mu, torch.cos(th0), torch.sin(th0)], 1), torch.cat([mu, torch.cos(thT), torch.sin(thT)], 1)
        if self.bnd:
            dP = p["dP"][T][rows]
            gb = target(p, idx, th[:, [0, -1]], self.m) / p["rho"][idx, None, None]  # (B, 2, m)
            f0 = torch.cat([torch.stack([P[:, 0], dP[:, 0], gb[:, 0]], 1), z0[:, :, None].expand(-1, -1, self.m)], 1)
            fT = torch.cat([torch.stack([P[:, -1], dP[:, -1], gb[:, 1]], 1), zT[:, :, None].expand(-1, -1, self.m)], 1)
            ca = self.lay_a(f0).view(B, self.r, self.nb, self.m)
            cb = self.lay_b(fT).view(B, self.r, self.nb, self.m)
        else:
            ca = self.lay_a(z0).view(B, self.r, self.nb, self.m)
            cb = self.lay_b(zT).view(B, self.r, self.nb, self.m)
        E = torch.exp(-lam[:, :, None] * t[None, None, :])
        F = torch.exp(-lam[:, :, None] * s[None, None, :])
        return per + torch.einsum("brn,bnk,brkm->bnm", E, Pl, ca) + torch.einsum("brn,bnk,brkm->bnm", F, Pl, cb)


class RefOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, p, idx, T):
        return p["P"][T][p["row"][T][idx]] + 0 * self.dummy


class CNNMulti(nn.Module):
    def __init__(self, m, n_mu, residual: bool, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.m, self.residual = m, residual
        self.lift = nn.Conv1d(n_mu + 4 + m + 1 + (2 * m if residual else 0), width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=dd, dilation=dd) for dd in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))
        if residual:
            with torch.no_grad():
                self.proj[-1].weight.mul_(0.05)
                self.proj[-1].bias.zero_()

    def forward(self, p, idx, T):
        t, _ = grid(T, p["dt"])
        B, N = len(idx), len(t)
        th = phases(p, idx, t)
        g = target(p, idx, th, self.m)
        h = [p["mu"][idx, :, None].expand(-1, -1, N), torch.cos(th).transpose(1, 2), torch.sin(th).transpose(1, 2), g.transpose(1, 2),
             torch.ones(B, 1, N, dtype=t.dtype)]
        if self.residual:
            rows = p["row"][T][idx]
            P = p["P"][T][rows]
            h += [P.transpose(1, 2), p["dP"][T][rows].transpose(1, 2)]
        h = self.lift(torch.cat(h, 1))
        for conv in self.convs:
            h = h + nn.functional.gelu(conv(h))
        out = self.proj(h).transpose(1, 2)
        return P + out if self.residual else out


class CNNTC(nn.Module):
    """cnn_res with the same trunk (the m nodes as channels, 1D convolutions in t), but valid in t on the extended grid
    [-W dt, T + W dt] and with the clipped distances to the ends as two more channels, so it is horizon-independent."""

    def __init__(self, m, n_mu, width=64, tdil=TDIL, edge_L=EDGE_L):
        super().__init__()
        self.m, self.tdil, self.edge_L = m, tuple(tdil), edge_L
        self.lift = nn.Conv1d(n_mu + 4 + m + 1 + 2 * m + 2, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=dd) for dd in self.tdil])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, m, 1))
        with torch.no_grad():
            self.proj[-1].weight.mul_(0.05)
            self.proj[-1].bias.zero_()

    def forward(self, p, idx, T):
        t, _ = grid(T, p["dt"])
        B, N, W = len(idx), len(t), sum(self.tdil)
        assert p["ext"] == W, "load(..., ext) must equal the receptive half-width sum(tdil)"
        te_ = torch.arange(1 - W, N + W + 1, dtype=t.dtype) * p["dt"]
        Ne = len(te_)
        th = phases(p, idx, te_)
        rows = p["row"][T][idx]
        L = self.edge_L
        ramp = torch.stack([torch.clamp(1 - te_ / L, min=0), torch.clamp(1 - (N * p["dt"] - te_) / L, min=0)])
        h = [p["mu"][idx, :, None].expand(-1, -1, Ne), torch.cos(th).transpose(1, 2), torch.sin(th).transpose(1, 2),
             target(p, idx, th, self.m).transpose(1, 2), torch.ones(B, 1, Ne, dtype=t.dtype),
             p["Pext"][T][rows].transpose(1, 2), p["dPext"][T][rows].transpose(1, 2), ramp[None].expand(B, -1, -1)]
        h = self.lift(torch.cat(h, 1))
        for conv, dd in zip(self.convs, self.tdil):
            h = h[:, :, dd:-dd] + nn.functional.gelu(conv(h))
        return p["P"][T][rows] + self.proj(h).transpose(1, 2)


def train(model, p, train_T, epochs, lr, batch, wd=0.0):
    if epochs == 0:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if wd > 0 else torch.optim.Adam(model.parameters(), lr=lr)
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
                v = float(np.mean([float(torch.cat([rel(model(p, p["va"][k:k + 50], T), p["Y"][T][p["row"][T][p["va"][k:k + 50]]])
                                                   for k in range(0, len(p["va"]), 50)]).median()) for T in train_T]))
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
        if ep % max(1, epochs // 10) == 0:
            print(f"    ep {ep} loss {float(loss.detach()):.3e} val {v:.3e}", flush=True)
    model.load_state_dict(best_state)
    return model


def windows(u, y, dt, tau=0.5):
    """Per-sample relative errors on [0, tau], [T - tau, T] and the interior."""
    n = int(round(tau / dt))
    out = {}
    for k, sl in (("init", slice(0, n)), ("term", slice(u.shape[1] - n, None)), ("interior", slice(n, u.shape[1] - n))):
        if sl == slice(n, u.shape[1] - n) and u.shape[1] <= 2 * n:
            continue
        out[k] = rel(u[:, sl], y[:, sl])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--kind", choices=["nl", "lin"], required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=["tp", "tp_local", "tp_bnd", "tp_local_bnd", "tp_qp", "tp_tc", "tp_tce", "tp_tce_nolayer",
                                         "ref_only", "cnn", "cnn_res", "cnn_res_tc"], required=True)
    ap.add_argument("--tag", default="", help="suffix of the prediction file names, to keep variants of one model apart")
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay (0: Adam)")
    ap.add_argument("--tdil", type=int, nargs="+", default=list(TDIL), help="time dilations of the tp_tc* temporal CNN")
    ap.add_argument("--edge-L", type=float, default=EDGE_L, help="ramp length of the tp_tce* distance-to-end channels")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    p = load(args.data, args.kind, ext=sum(args.tdil) if args.model.startswith("tp_tc") or args.model == "cnn_res_tc" else 0)
    tc = dict(tc=True, tdil=args.tdil, edge_L=args.edge_L)
    if args.fp32:
        conv = lambda v: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v  # noqa: E731
        p = {k: ({kk: conv(vv) for kk, vv in v.items()} if isinstance(v, dict) else conv(v)) for k, v in p.items()}
        torch.set_default_dtype(torch.float32)
    dev = torch.device(args.device)
    move = lambda v: v.to(dev) if isinstance(v, torch.Tensor) else v  # noqa: E731
    p = {k: ({kk: move(vv) for kk, vv in v.items()} if isinstance(v, dict) else move(v)) for k, v in p.items()}
    torch.set_default_device(dev)
    te, dt = p["te"], p["dt"]
    runs = []
    for seed in (args.seeds if args.model != "ref_only" else [0]):
        torch.manual_seed(seed)
        model = {"tp": lambda: TPMulti(p["m"], p["n_mu"], local=False, r=args.r), "tp_local": lambda: TPMulti(p["m"], p["n_mu"], local=True, r=args.r),
                 "tp_bnd": lambda: TPMulti(p["m"], p["n_mu"], local=False, r=args.r, bnd=True),
                 "tp_local_bnd": lambda: TPMulti(p["m"], p["n_mu"], local=True, r=args.r, bnd=True),
                 "tp_qp": lambda: TPMulti(p["m"], p["n_mu"], local=True, r=args.r, bnd=True, phase=True),
                 "tp_tc": lambda: TPMulti(p["m"], p["n_mu"], local=True, r=args.r, bnd=True, **tc),
                 "tp_tce": lambda: TPMulti(p["m"], p["n_mu"], local=True, r=args.r, bnd=True, edge=True, **tc),
                 "tp_tce_nolayer": lambda: TPMulti(p["m"], p["n_mu"], local=True, r=args.r, edge=True, layers=False, **tc),
                 "ref_only": RefOnly, "cnn": lambda: CNNMulti(p["m"], p["n_mu"], residual=False),
                 "cnn_res": lambda: CNNMulti(p["m"], p["n_mu"], residual=True),
                 "cnn_res_tc": lambda: CNNTC(p["m"], p["n_mu"], tdil=args.tdil, edge_L=args.edge_L)}[args.model]()
        n_par = sum(q.numel() for q in model.parameters())
        print(f"{args.kind}/{args.model} seed {seed} ({n_par} params, train T={args.train_T})", flush=True)
        t0 = time.perf_counter()
        model = train(model, p, args.train_T, args.epochs if args.model != "ref_only" else 0, args.lr, args.batch, args.wd)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            res, preds, win = {}, {}, {}
            for T in T_EVAL:
                out = torch.cat([model(p, te[k : k + 10], T) for k in range(0, len(te), 10)])
                Uref = p["U"][T][p["row"][T][te]]
                u = out * p["rho"][te, None, None]
                res[str(T)] = summarize(rel(u, Uref).cpu().double().numpy())
                win[T] = windows(u, Uref, dt)
                res[str(T)].update({f"{k}_median": float(v.median()) for k, v in win[T].items()})
                if T in (1, 8):
                    preds[f"U_T{T}"] = u.cpu().numpy().astype(np.float32)
            for k in ("init", "term"):
                ratio = (win[8][k] / win[1][k]).cpu().double().numpy()
                res[f"transfer_{k}"] = {"median": float(np.median(ratio)), "p90": float(np.percentile(ratio, 90))}
            if args.save_pred is not None:
                args.save_pred.mkdir(parents=True, exist_ok=True)
                np.savez(args.save_pred / f"pred_{args.kind}_{args.model}{args.tag}_seed{seed}.npz", test_idx=te.cpu().numpy(), **preds)
        runs.append({"seed": seed, "params": n_par, "train_seconds": secs, "test": res})
        print(f"  {args.kind}/{args.model}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in T_EVAL)
              + f"  transfer init {res['transfer_init']['median']:.2f} term {res['transfer_term']['median']:.2f} ({secs:.0f}s)", flush=True)
    rep = {"kind": args.kind, "model": args.model, "train_T": args.train_T, "r": args.r, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "wd": args.wd,
           "tdil": args.tdil, "edge_L": args.edge_L, "pred_prefix": f"pred_{args.kind}_{args.model}{args.tag}",
           "runs": runs, "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")}
                                             for T in T_EVAL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Which structural priors make the T=1 exponential fit extrapolate in the horizon?

Per test sample, fit u/u_bar - 1 on T = 1 with
    sum_i a_i e^{-lam_i t} + sum_i b_i e^{-kap_i (1 - t)}
(rates by Adam, amplitudes by least squares) under the variants
    separate   : lam and kap independent
    shared     : kap = lam (Hamiltonian +/- symmetry of the LQ optimality system)
    +floor     : rates >= lam_1(beta), the smallest generalized eigenvalue of (K_beta, M)
and evaluate on T = 2, 4, 8.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import scipy.linalg as sla
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators  # noqa: E402

torch.set_default_dtype(torch.float64)


def grid(T, dt):
    N = int(round(T / dt))
    t = torch.arange(1, N + 1) * dt
    return t, T - t


def first_eig(beta: float) -> float:
    ops = assemble_operators(RobinConfig(beta=beta))
    return float(sla.eigh(ops.K_beta.toarray(), ops.M.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0])


def fit(Y1, dt, r, shared, floor, iters=4000, lr=0.02):
    t, s = grid(1, dt)
    target = Y1 - 1.0
    B = Y1.shape[0]
    lam0 = torch.logspace(math.log10(2.0), math.log10(400.0), r)
    base = floor[:, None] if floor is not None else torch.zeros(B, 1)
    z0 = torch.log(torch.clamp(lam0[None, :] - base, min=0.5))
    z = nn.Parameter(z0.clone() if shared else z0.repeat(1, 2))

    def rates(zz):
        rr = base + torch.exp(zz)
        return (rr, rr) if shared else (rr[:, :r], rr[:, r:])

    def design(zz, tt, ss):
        lam, kap = rates(zz)
        return torch.cat([torch.exp(-lam[:, :, None] * tt[None, None, :]),
                          torch.exp(-kap[:, :, None] * ss[None, None, :])], dim=1).transpose(1, 2)

    def solve(Phi):
        G = Phi.transpose(1, 2) @ Phi
        ridge = 1e-12 * torch.diagonal(G, dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(2 * r)
        return torch.linalg.solve(G + ridge, Phi.transpose(1, 2) @ target[:, :, None])

    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        Phi = design(z, t, s)
        loss = ((Phi @ solve(Phi)).squeeze(-1) - target).pow(2).sum(1).div(Y1.pow(2).sum(1)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        c = solve(design(z, t, s))
    return lambda T: (1.0 + (design(z, *grid(T, dt)) @ c).squeeze(-1)).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d = np.load(args.data)
    te = d["test_idx"]
    dt = float(d["dt"])
    Y = {T: torch.tensor(d[f"U_T{T}"][te] / d["u_bar"][te, None]) for T in (1, 2, 4, 8)}
    lam1 = torch.tensor([first_eig(b) for b in d["beta"][te]])
    print(f"lambda_1(beta) over test set: {float(lam1.min()):.3f} .. {float(lam1.max()):.3f}", flush=True)
    out = {}
    for r in (4, 8):
        for shared in (False, True):
            for use_floor in (False, True):
                name = f"r{r}_{'shared' if shared else 'separate'}{'_floor' if use_floor else ''}"
                pred = fit(Y[1], dt, r, shared, lam1 if use_floor else None)
                res = {}
                for T in (1, 2, 4, 8):
                    e = (torch.linalg.norm(pred(T) - Y[T], dim=1) / torch.linalg.norm(Y[T], dim=1)).numpy()
                    res[str(T)] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(e.max())}
                out[name] = res
                print(name + ": " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in (1, 2, 4, 8)), flush=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

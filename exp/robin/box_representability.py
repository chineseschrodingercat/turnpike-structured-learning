#!/usr/bin/env python3
"""Is the worst-case error of the box surrogate a representation limit or a learning limit?

For every test sample, fit the surrogate's own functional form PER SAMPLE (no learning across mu)
to the exact T=8 adjoint trace: q/u_bar = q_bar/u_bar + sum a e^{-lam t} + sum b e^{-lam (T-t)},
rates >= lambda_1(beta) (shared between ends), r in {4, 6, 8}; amplitudes by least squares, rates by Adam.
Report the error of u = clip(q) at T = 8 (and T = 1 via the same fitted layers).
Small per-sample error on the worst samples => the learned model's worst case is a learning problem.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from turnpike_surrogate import first_eig, grid  # noqa: E402

torch.set_default_dtype(torch.float64)


def fit(Q, qbar, floor, T, dt, r, iters=3000, lr=0.02):
    t, s = grid(T, dt)
    target = Q - qbar[:, None]
    B = Q.shape[0]
    z = nn.Parameter(torch.log(torch.clamp(torch.logspace(math.log10(2.0), math.log10(400.0), r)[None, :] - floor[:, None], min=0.5)))
    opt = torch.optim.Adam([z], lr=lr)

    def design(zz, tt, ss):
        lam = floor[:, None] + torch.exp(zz)
        return torch.cat([torch.exp(-lam[:, :, None] * tt[None, None, :]), torch.exp(-lam[:, :, None] * ss[None, None, :])], 1).transpose(1, 2)

    def solve(Phi):
        G = Phi.transpose(1, 2) @ Phi
        ridge = 1e-12 * torch.diagonal(G, dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(2 * r)
        return torch.linalg.solve(G + ridge, Phi.transpose(1, 2) @ target[:, :, None])

    for _ in range(iters):
        opt.zero_grad()
        Phi = design(z, t, s)
        loss = ((Phi @ solve(Phi)).squeeze(-1) - target).pow(2).sum(1).div(Q.pow(2).sum(1)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        c = solve(design(z, t, s))
    return lambda TT: (qbar[:, None] + (design(z, *grid(TT, dt)) @ c).squeeze(-1)).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    d = np.load(args.data)
    te = d["test_idx"]
    ub = d["u_bar"][te]
    dt = float(d["dt"])
    qbar = torch.tensor(d["q_bar"][te] / ub)
    lo, hi = torch.tensor(d["X"][te, 4]), torch.tensor(d["X"][te, 5])
    floor = torch.tensor([first_eig(b) for b in d["beta"][te]])
    Q8 = torch.tensor(d["Q_T8"][te] / ub[:, None])
    U = {T: torch.tensor(d[f"U_T{T}"][te] / ub[:, None]) for T in (1, 8)}
    rep = {}
    for r in (4, 6, 8):
        pred = fit(Q8, qbar, floor, 8, dt, r)
        res = {}
        for T in (1, 8):
            u = torch.minimum(torch.maximum(pred(T), lo[:, None]), hi[:, None])
            e = (torch.linalg.norm(u - U[T], dim=1) / torch.linalg.norm(U[T], dim=1)).numpy()
            res[str(T)] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(e.max()),
                           "worst5_idx": [int(i) for i in np.argsort(e)[-5:][::-1]], "worst5_err": [float(x) for x in np.sort(e)[-5:][::-1]]}
        rep[f"r{r}"] = res
        print(f"r={r}: per-sample fit of exact T=8 layers, u-error  T=8 med {res['8']['median']:.2e} max {res['8']['max']:.2e} | "
              f"T=1 (same layers) med {res['1']['median']:.2e} max {res['1']['max']:.2e}", flush=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stronger baselines: DeepONet, FNO-1D and a dilated CNN trained on horizons T in {1, 2}; tested on T = 1, 2, 4, 8.

All models predict the control normalized by the closed-form plateau scale
(u_bar for the constant target, 1.25 u_bar for the oscillatory target) and are
selected on validation error at the training horizons only.

    deeponet  branch(params) . trunk(t, T - t) + bias
    fno       FNO-1D in time; input channels [params (broadcast), y_d(t), t, T - t]; right zero-padding N/4
    cnn       dilated residual CNN, no absolute coordinates: [params (broadcast), y_d(t), ones]
              with zero padding (edges are visible only through the padding); receptive field +-1023 steps
    turnpike  TurnpikeNet (constant) / OscTurnpikeNet (oscillatory), r = 4, rate floor lambda_1(beta)
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from turnpike_surrogate import TurnpikeNet, first_eig, grid, mlp, rel_err, summarize  # noqa: E402
from osc_turnpike_surrogate import C_OFF, OscTurnpikeNet  # noqa: E402
from osc_turnpike_surrogate import load as load_osc_raw  # noqa: E402

T_EVAL = [1, 2, 4, 8]


def std_cols(Z, tr):
    return torch.tensor((Z - Z[tr].mean(0)) / Z[tr].std(0))


def load_problem(kind: str, path: pathlib.Path) -> dict:
    d = np.load(path)
    tr = d["train_idx"]
    p = {"kind": kind, "dt": float(d["dt"]), "tr": torch.tensor(tr), "va": torch.tensor(d["val_idx"]), "te": torch.tensor(d["test_idx"])}
    if kind == "const":
        p["x"] = std_cols(d["X"], tr)
        p["scale"] = torch.tensor(d["u_bar"])
        p["Y"] = {T: torch.tensor(d[f"U_T{T}"]) / p["scale"][:, None] for T in T_EVAL}
        p["yd"] = lambda T, idx: torch.ones(len(idx), int(round(T / p["dt"])))
        # 2D data store the (separable) first eigenvalue; 1D data recompute it
        p["floor"] = torch.tensor(d["lam1"]) if "lam1" in d.files else torch.tensor([first_eig(b) for b in d["beta"]])
        p["beta"] = d["beta"] if "beta" in d.files else None
    else:
        p["x"] = std_cols(d["X"], tr)  # w, c_rel, log a, log b, omega, A
        p["scale"] = C_OFF * torch.tensor(d["u_bar"])
        om, A = torch.tensor(d["omega"]), torch.tensor(d["A"])
        p["Y"] = {T: (C_OFF * torch.tensor(d[f"U_const_T{T}"]) + A[:, None] * torch.tensor(d[f"U_sin_T{T}"])) / p["scale"][:, None] for T in T_EVAL}

        def yd(T, idx):
            t, _ = grid(T, p["dt"])
            return (C_OFF + A[idx, None] * torch.sin(om[idx, None] * t[None, :])) / C_OFF

        p["yd"] = yd
        p["osc_raw"] = load_osc_raw(path)
    return p


class DeepONet(nn.Module):
    def __init__(self, n_par, p=32, width=128):
        super().__init__()
        self.branch, self.trunk = mlp(n_par, width, p), mlp(2, width, p, act=nn.Tanh)
        self.bias = nn.Parameter(torch.ones(()))

    def forward(self, x, yd, t, s):
        return self.branch(x) @ self.trunk(torch.stack([t, s], -1)).T + self.bias


class SpectralConv1d(nn.Module):
    def __init__(self, ci, co, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter((1.0 / (ci * co)) * torch.randn(ci, co, modes, dtype=torch.complex128))

    def forward(self, x):
        xh = torch.fft.rfft(x)
        m = min(self.modes, xh.shape[-1])
        out = torch.zeros(x.shape[0], self.w.shape[1], xh.shape[-1], dtype=xh.dtype, device=x.device)
        out[..., :m] = torch.einsum("bim,iom->bom", xh[..., :m], self.w[..., :m])
        return torch.fft.irfft(out, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, n_par, width=32, modes=32, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(n_par + 3, width, 1)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, 4 * width, 1), nn.GELU(), nn.Conv1d(4 * width, 1, 1))

    def forward(self, x, yd, t, s):
        B, N = yd.shape
        h = torch.cat([x[:, :, None].expand(-1, -1, N), yd[:, None, :],
                       t[None, None, :].expand(B, 1, N), s[None, None, :].expand(B, 1, N)], dim=1)
        h = self.lift(h)
        pad = max(16, N // 4)
        h = nn.functional.pad(h, (0, pad))
        for k, (sc, sk) in enumerate(zip(self.spec, self.skip)):
            h = sc(h) + sk(h)
            if k < len(self.spec) - 1:
                h = nn.functional.gelu(h)
        return self.proj(h[..., :N]).squeeze(1)


class DilatedCNN(nn.Module):
    def __init__(self, n_par, width=32, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.lift = nn.Conv1d(n_par + 2, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations])
        self.proj = nn.Sequential(nn.Conv1d(width, 4 * width, 1), nn.GELU(), nn.Conv1d(4 * width, 1, 1))

    def forward(self, x, yd, t, s):
        B, N = yd.shape
        h = self.lift(torch.cat([x[:, :, None].expand(-1, -1, N), yd[:, None, :], torch.ones(B, 1, N)], dim=1))
        for conv in self.convs:
            h = h + nn.functional.gelu(conv(h))
        return self.proj(h).squeeze(1)


class TurnpikeConst(nn.Module):
    """floor: 'lam1' = stored lam1 (linearized floor for the semilinear data), 'open' = open-loop lambda_1(beta),
    'split' = open-loop floor for the initial rates and lam1 for the terminal rates (unshared rates)."""

    def __init__(self, p, r=4, floor="lam1"):
        super().__init__()
        self.net = TurnpikeNet(r=r, shared=floor != "split", use_floor=True, n_in=p["x"].shape[1])
        if floor == "lam1":
            self.floor = p["floor"]
        else:
            fo = torch.tensor([first_eig(b) for b in p["beta"]])
            self.floor = fo if floor == "open" else torch.stack([fo, p["floor"]], 1)
        self._idx = None

    def forward(self, x, yd, t, s, idx=None):
        return self.net(x, t, s, self.floor[idx])


class TurnpikeOsc(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net, self.raw = OscTurnpikeNet(), p["osc_raw"]

    def forward(self, x, yd, t, s, idx=None):
        T = float(t[-1] + s[-1])
        u = self.net.predict(self.raw, idx, T)
        return u / (C_OFF * self.raw["ubar"][idx, None])


def predict(model, p, idx, T):
    t, s = grid(T, p["dt"])
    x, yd = p["x"][idx], p["yd"](T, idx)
    if isinstance(model, (TurnpikeConst, TurnpikeOsc)):
        return model(x, yd, t, s, idx=idx)
    return model(x, yd, t, s)


def loss_on(model, p, idx, T):
    """Relative squared error of the normalized control; TurnpikeOsc uses its component loss (targets 1, sin, cos)."""
    if isinstance(model, TurnpikeOsc):
        raw = model.raw
        uc, Ur, Ui = model.net(raw, idx, T)
        num = (Ur - raw[f"k{T}"][idx]).pow(2).sum(1) + (Ui - raw[f"s{T}"][idx]).pow(2).sum(1)
        den = raw[f"k{T}"][idx].pow(2).sum(1) + raw[f"s{T}"][idx].pow(2).sum(1)
        return (rel_err(uc, raw[f"c{T}"][idx]) ** 2).mean() + (num / den).mean()
    return (rel_err(predict(model, p, idx, T), p["Y"][T][idx]) ** 2).mean()


def train(model, p, train_T, epochs, lr, batch):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = int(np.ceil(len(p["tr"]) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=lr * 1e-3)
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for ep in range(1, epochs + 1):
        perm = p["tr"][torch.randperm(len(p["tr"]), device=p["tr"].device)]
        for b in range(steps_per_epoch):
            idx = perm[b * batch : (b + 1) * batch]
            opt.zero_grad()
            loss = sum(loss_on(model, p, idx, T) for T in train_T) / len(train_T)
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
    ap.add_argument("--kind", choices=["const", "osc"], required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--models", nargs="+", default=["turnpike", "deeponet", "fno", "cnn"])
    ap.add_argument("--train-T", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None, help="directory for test-set predictions (unnormalized)")
    ap.add_argument("--tp", type=json.loads, default='{"r": 4, "floor": "lam1"}', help="TurnpikeConst options")
    ap.add_argument("--epochs", type=json.loads, default='{"turnpike": 6000, "deeponet": 6000, "fno": 300, "cnn": 300}')
    ap.add_argument("--batch", type=json.loads, default='{"turnpike": 700, "deeponet": 700, "fno": 50, "cnn": 50}')
    ap.add_argument("--lr", type=json.loads, default='{"turnpike": 2e-3, "deeponet": 2e-3, "fno": 1e-3, "cnn": 1e-3}')
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load_problem(args.kind, args.data)
    n_par = p["x"].shape[1]
    makers = {
        "deeponet": lambda: DeepONet(n_par),
        "fno": lambda: FNO1d(n_par),
        "cnn": lambda: DilatedCNN(n_par),
        "turnpike": lambda: TurnpikeConst(p, **args.tp) if args.kind == "const" else TurnpikeOsc(p),
    }
    report = {"kind": args.kind, "train_T": args.train_T, "models": {}}
    if args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
    te = p["te"]
    for name in args.models:
        runs = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = makers[name]()
            n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
            t0 = time.perf_counter()
            print(f"{args.kind} {name} seed {seed} ({n_params} params, train T={args.train_T})", flush=True)
            model = train(model, p, args.train_T, args.epochs[name], args.lr[name], args.batch[name])
            secs = time.perf_counter() - t0
            with torch.no_grad():
                res = {str(T): summarize(rel_err(predict(model, p, te, T), p["Y"][T][te]).cpu().numpy()) for T in T_EVAL}
                t1 = time.perf_counter()
                predict(model, p, te, 8)
                infer = (time.perf_counter() - t1) / len(te)
                if args.save_pred is not None:
                    args.save_pred.mkdir(parents=True, exist_ok=True)
                    np.savez(args.save_pred / f"pred_{name}_seed{seed}.npz", test_idx=te.cpu().numpy(), train_T=np.array(args.train_T),
                             **{f"U_T{T}": (predict(model, p, te, T) * p["scale"][te, None]).cpu().numpy() for T in T_EVAL})
            runs.append({"seed": seed, "params": n_params, "train_seconds": secs, "infer_seconds_per_sample_T8": infer, "test": res})
            print(f"  {name}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}" for T in T_EVAL)
                  + f"  ({secs:.0f}s)", flush=True)
        report["models"][name] = {
            "runs": runs,
            "mean_over_seeds": {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in runs])) for k in ("median", "p90", "max")} for T in T_EVAL},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""1D test of the convolutional, horizon-independent implementation of the turnpike surrogate (log section 32).

Question: cnn_res_tc (E20) represents the end layers only inside a finite window L + W dt <= T_train / 2. In 1D the
slowest layer rates are lambda_1 in [2.7, 8.2] (E01), so with T_train = 1 the layers may outlive the window. Does the
convolutional implementation still work, how long a training horizon does it need, and does a hybrid (explicit
exponential modes + convolutional correction) keep the advantages of both?

Data: E01 (linear Robin problem, constant target, runs/turnpike/turnpike_data.npz; every sample solved on T = 1, 2, 4, 8)
and E08 (semilinear, runs/semilinear/data_semilinear.npz; training/validation samples on T = 1, 2 only). Every model
predicts u / u_bar with the exact plateau u_bar, as in E01/E08, and uses the E01/E08 splits and input columns.

  tp     TurnpikeNet: 1 + sum_i a_i(mu) e^{-lam_i t} + sum_i b_i(mu) e^{-lam_i (T - t)}, r = 4, shared rates,
         rates >= floor (E01: lambda_1(beta); E08: the linearized floor lam1), exactly as in E01/E08.
  tc     cnn_res_tc in 1D: channels [mu (broadcast), ones, max(0, 1 - t/L), max(0, 1 - (T - t)/L)] on the extended
         grid t_k = k dt, k = 1 - W, ..., N + W; 1x1 lift to width 64; residual blocks h <- crop_d(h) + GELU(conv(h))
         with kernel 3, dilation d and no padding (valid), d in tdil, W = sum(tdil); head 64 -> 128 -> 1 (last layer
         scaled by 0.05, bias 0); output 1 + head on t_1..t_N. The output at t depends on the inputs on
         [t - W dt, t + W dt] only, so the model is exactly horizon independent on T >= 2 (L + W dt).
  hyb    tp + tc correction with a short window (L + W dt < T_train / 2): the exponential modes carry the slow tail
         beyond the window, the convolution corrects the shape near the ends. Trained jointly.
  hyb2   the same model trained in two stages: first tp alone (the tp protocol, --tp-*), then tp frozen and only the
         convolutional correction trained on the residual (the tc protocol).
  local_oracle  no learning: the exact u_{T_train} on its first and last T_train / 2, continued in between by its
         midpoint value. This is what a perfectly trained tc with a window of T_train / 2 predicts (structural floor).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines_multihorizon as bm  # noqa: E402
from baselines_multihorizon import TurnpikeConst, load_problem  # noqa: E402
from turnpike_surrogate import first_eig, grid, rel_err, summarize  # noqa: E402

T_EVAL = [1, 2, 4, 8]
TAU = 0.5  # end-window length for the window errors and transfer ratios


class CNNTC1D(nn.Module):
    def __init__(self, n_par, width=64, tdil=(1, 2, 4, 8, 16, 32, 64), edge_L=0.1):
        super().__init__()
        self.tdil, self.edge_L, self.W = tuple(tdil), edge_L, sum(tdil)
        self.lift = nn.Conv1d(n_par + 3, width, 1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d) for d in self.tdil])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, 1, 1))
        with torch.no_grad():
            self.proj[-1].weight.mul_(0.05)
            self.proj[-1].bias.zero_()

    def window(self, dt):
        """Half-width of the end region in which the output depends on the distance to that end."""
        return self.edge_L + self.W * dt

    def forward(self, x, yd, t, s):
        dt, N, W, L, B = float(t[0]), len(t), self.W, self.edge_L, x.shape[0]
        te = torch.arange(1 - W, N + W + 1, dtype=t.dtype) * dt
        ramp = torch.stack([torch.clamp(1 - te / L, min=0), torch.clamp(1 - (N * dt - te) / L, min=0)])
        h = torch.cat([x[:, :, None].expand(-1, -1, len(te)), torch.ones(B, 1, len(te), dtype=t.dtype),
                       ramp[None].expand(B, -1, -1)], 1)
        h = self.lift(h)
        for conv, d in zip(self.convs, self.tdil):
            h = h[:, :, d:-d] + nn.functional.gelu(conv(h))
        return 1.0 + self.proj(h).squeeze(1)


class Hybrid(TurnpikeConst):
    """Explicit exponential layers (TurnpikeConst) plus a short-window CNNTC1D correction."""

    def __init__(self, p, r=4, floor="lam1", **tc):
        super().__init__(p, r=r, floor=floor)
        self.tc = CNNTC1D(p["x"].shape[1], **tc)

    def forward(self, x, yd, t, s, idx=None):
        return self.net(x, t, s, self.floor[idx]) + self.tc(x, yd, t, s) - 1.0


def local_oracle(Y, Ttr, T, dt):
    n, N = int(round(Ttr / dt)), int(round(T / dt))
    h = n // 2
    out = np.empty((Y.shape[0], N))
    out[:, :h] = Y[:, :h]
    out[:, N - h:] = Y[:, n - h:]
    out[:, h:N - h] = Y[:, h - 1:h]
    return out


def windows(u, y, dt):
    n = int(round(TAU / dt))
    out = {"init": rel_err(u[:, :n], y[:, :n]), "term": rel_err(u[:, -n:], y[:, -n:])}
    if u.shape[1] > 2 * n:
        out["interior"] = rel_err(u[:, n:-n], y[:, n:-n])
    return out


def evaluate(pred_fn, p, T0):
    """pred_fn(T) -> normalized prediction on the test set; returns the result dict and the predictions."""
    te, res, win, preds = p["te"], {}, {}, {}
    for T in T_EVAL:
        u = pred_fn(T)
        if u is None:
            continue
        y = p["Y"][T][te]
        e = rel_err(u, y)
        res[str(T)] = summarize(e.cpu().double().numpy())
        res[str(T)]["per_sample"] = [float(v) for v in e.cpu().double().numpy()]
        win[T] = windows(u, y, p["dt"])
        res[str(T)].update({f"{k}_median": float(v.median()) for k, v in win[T].items()})
        preds[f"U_T{T}"] = (u * p["scale"][te, None]).cpu().numpy().astype(np.float32)
    for k in ("init", "term"):
        if T0 in win and 8 in win:
            ratio = (win[8][k] / win[T0][k]).cpu().double().numpy()
            res[f"transfer_{k}"] = {"median": float(np.median(ratio)), "p90": float(np.percentile(ratio, 90))}
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=["tp", "tc", "hyb", "hyb2", "local_oracle"], required=True)
    ap.add_argument("--tp-epochs", type=int, default=6000, help="hyb2 stage 1")
    ap.add_argument("--tp-batch", type=int, default=700, help="hyb2 stage 1")
    ap.add_argument("--tp-lr", type=float, default=2e-3, help="hyb2 stage 1")
    ap.add_argument("--train-T", type=int, nargs="+", default=[1])
    ap.add_argument("--tdil", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--edge-L", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--floor", default="lam1", help="TurnpikeConst floor: lam1 (stored or lambda_1(beta)), open, split")
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--save-pred", type=pathlib.Path, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    p = load_problem("const", args.data)
    if args.fp32:
        torch.set_default_dtype(torch.float32)
        for k, v in list(p.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                p[k] = v.float()
            elif isinstance(v, dict):
                p[k] = {kk: vv.float() for kk, vv in v.items()}
    d = np.load(args.data)
    te_np = d["test_idx"]
    lam_open = np.array([first_eig(b) for b in d["beta"][te_np]])
    lam_floor = (d["lam1"][te_np] if "lam1" in d.files else lam_open)
    T0, dt = args.train_T[0], p["dt"]
    name = args.model + args.tag
    rep = {"data": str(args.data), "model": args.model, "tag": args.tag, "train_T": args.train_T, "epochs": args.epochs,
           "batch": args.batch, "lr": args.lr, "fp32": args.fp32, "tdil": args.tdil, "edge_L": args.edge_L, "r": args.r,
           "floor": args.floor, "width": args.width, "lam_open_test": lam_open.tolist(), "lam_floor_test": lam_floor.tolist(),
           "runs": []}
    if args.model == "local_oracle":
        def pred_fn(T):
            if T < T0:
                return None
            Y = p["Y"][T0][p["te"]].cpu().double().numpy()
            return torch.tensor(local_oracle(Y, T0, T, dt), dtype=p["Y"][T].dtype)
        res, preds = evaluate(pred_fn, p, T0)
        rep["runs"].append({"seed": None, "params": 0, "train_seconds": 0.0, "test": res})
        seeds = []
    else:
        seeds = args.seeds
    n_par = p["x"].shape[1]
    tc_kw = dict(width=args.width, tdil=args.tdil, edge_L=args.edge_L)
    for seed in seeds:
        torch.manual_seed(seed)
        model = {"tp": lambda: TurnpikeConst(p, r=args.r, floor=args.floor),
                 "tc": lambda: CNNTC1D(n_par, **tc_kw),
                 "hyb": lambda: Hybrid(p, r=args.r, floor=args.floor, **tc_kw),
                 "hyb2": lambda: Hybrid(p, r=args.r, floor=args.floor, **tc_kw)}[args.model]()
        cnn = model if isinstance(model, CNNTC1D) else getattr(model, "tc", None)
        win_w = cnn.window(dt) if cnn is not None else None
        if win_w is not None:
            ok = all(T >= 2 * win_w - 1e-9 for T in args.train_T)
            print(f"window L + W dt = {win_w:.4f}; exactly horizon independent on T >= {2 * win_w:.4f}"
                  + ("" if ok else "  (WARNING: training horizon shorter than that)"), flush=True)
        n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
        print(f"{name} seed {seed} ({n_params} params, train T={args.train_T})", flush=True)
        t0 = time.perf_counter()
        stage1 = None
        if args.model == "hyb2":
            tp = bm.train(TurnpikeConst(p, r=args.r, floor=args.floor), p, args.train_T, args.tp_epochs, args.tp_lr, args.tp_batch)
            with torch.no_grad():
                stage1, _ = evaluate(lambda T: torch.cat([bm.predict(tp, p, p["te"][k:k + 50], T)
                                                          for k in range(0, len(p["te"]), 50)]), p, T0)
            print("  stage 1 (tp alone): " + "  ".join(f"T={T}: med {stage1[str(T)]['median']:.2e} max {stage1[str(T)]['max']:.2e}"
                                                    for T in T_EVAL), flush=True)
            model.net.load_state_dict(tp.net.state_dict())
            for q in model.net.parameters():
                q.requires_grad_(False)
        model = bm.train(model, p, args.train_T, args.epochs, args.lr, args.batch)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            def pred_fn(T):
                return torch.cat([bm.predict(model, p, p["te"][k:k + 50], T) for k in range(0, len(p["te"]), 50)])
            res, preds = evaluate(pred_fn, p, T0)
            t1 = time.perf_counter()
            bm.predict(model, p, p["te"][:1], 8)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            res["infer_seconds_one_sample_T8"] = time.perf_counter() - t1
        run = {"seed": seed, "params": n_params, "train_seconds": secs, "window": win_w, "test": res}
        if stage1 is not None:
            run["stage1_tp_test"] = {T: {k: v for k, v in r.items() if k != "per_sample"} if isinstance(r, dict) else r
                                     for T, r in stage1.items()}
        if isinstance(model, TurnpikeConst):
            with torch.no_grad():
                lam, kap, a, b = model.net.params(p["x"][p["te"]], model.floor[p["te"]])
            run["slowest_rate_test"] = lam.min(1).values.cpu().double().numpy().tolist()
        rep["runs"].append(run)
        if args.save_pred is not None and seed == seeds[0]:
            args.save_pred.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_pred / f"pred_{name}_seed{seed}.npz", test_idx=te_np, **preds)
        print(f"  {name}: " + "  ".join(f"T={T}: med {res[str(T)]['median']:.2e} max {res[str(T)]['max']:.2e}"
                                        for T in T_EVAL if str(T) in res)
              + f"  transfer init {res.get('transfer_init', {}).get('median', float('nan')):.2f}"
              + f" term {res.get('transfer_term', {}).get('median', float('nan')):.2f} ({secs:.0f}s)", flush=True)
    if args.model == "local_oracle" and args.save_pred is not None:
        args.save_pred.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_pred / f"pred_{name}.npz", test_idx=te_np, **preds)
    rep["mean_over_seeds"] = {str(T): {k: float(np.mean([r["test"][str(T)][k] for r in rep["runs"]]))
                                       for k in ("median", "p90", "max", "init_median", "term_median")}
                              for T in T_EVAL if str(T) in rep["runs"][0]["test"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Fit the matched-budget V1.2 semilinear panel and score paired endpoint windows."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paths import CODE_DIR, RAW_DATA, output_directory
from experiments.rebuild.rebuild_moving_full import add_horizon, prepare

sys.path.insert(0, str(CODE_DIR))
import baselines_multihorizon as scalar_models
import multi_turnpike_surrogate as strip_models
import semilinear_moving_surrogate as moving_models


DATASETS = {
    "2D-semi": "02 - Supplement/runs/semilinear2d/data_semilinear2d_strong.npz",
    "2D-semi-strip": "02 - Supplement/runs/semilinear2d_mi/data.npz",
    "2D-semi-move": "01 - Initial delivery/runs/semilinear_moving/data.npz",
}
UPDATES = {"2D-semi": 6000, "2D-semi-strip": 3000, "2D-semi-move": 20000}
COMMON = ("structured", "cnn_T1", "cnn_T12")
MOVING_EXTRA = ("global", "residual_T1", "residual_T12")
BATCH = 32
HORIZONS = (1, 2, 4, 8)
CHECKS = 40
SEEDS = (0, 1, 2)


def as_float(value):
    return torch.as_tensor(value, dtype=torch.float32)


def load_family(family, path):
    data = np.load(path)
    train, val, test = (data[key] for key in ("train_idx", "val_idx", "test_idx"))
    assert len(set(np.concatenate((train, val, test)))) == len(data["X"])
    assert len(val) == len(test) == 100
    if family == "2D-semi-move":
        p, _, raw_reference = prepare(data)
        add_horizon(p, raw_reference, 2, np.arange(len(data["X"])))
        p["Y"][2] = as_float(data["U_T2"]) / p["rho"][:, None, None]
        for horizon in (4, 8):
            add_horizon(p, raw_reference, horizon, test)
        scale = p["rho"]
    else:
        x = data["X"]
        p = {"x": as_float((x - x[train].mean(0)) / x[train].std(0)),
             "dt": float(data["dt"]), "floor": as_float(data["lam1"])}
        if family == "2D-semi":
            scale = as_float(data["u_bar"])
            p["yd"] = lambda horizon, idx: torch.ones(len(idx), round(horizon / p["dt"]))
        else:
            scale = as_float(np.sqrt(np.mean(data["u_bar"]**2, axis=1)))
            p["ubar_n"] = as_float(data["u_bar"]) / scale[:, None]
        view = (-1, 1) if family == "2D-semi" else (-1, 1, 1)
        p["Y"] = {h: as_float(data[f"U_T{h}"]) / scale.view(view) for h in (1, 2)}
    # Preserve the stored physical controls in their native precision for scoring.
    truth = {}
    for h in HORIZONS:
        key = f"U_T{h}"
        truth[h] = data[key + "_test"] if key + "_test" in data else data[key][test]
    return p, scale, truth, (train, val, test)


def make_model(family, arm, p):
    if family == "2D-semi":
        if arm == "structured":
            return scalar_models.TurnpikeConst(p, r=8)
        return scalar_models.DilatedCNN(p["x"].shape[1])
    if family == "2D-semi-strip":
        if arm == "structured":
            return strip_models.GenericTurnpike(99, n_in=10, r=8, width=256)
        return strip_models.CNNmulti(99, n_in=10)
    if arm == "structured":
        return moving_models.MovingTurnpikeLocal(99)
    if arm == "global":
        model = moving_models.MovingTurnpikeNL(99, True, r=8, K=24, K_layer=2)
        assert sum(w.numel() for w in model.parameters()) == 3561324
        return model
    if arm.startswith("residual"):
        return moving_models.CNNres(99)
    return moving_models.CNNmove(99)


def predict(model, family, p, ix, horizon):
    if family == "2D-semi":
        return scalar_models.predict(model, p, ix, horizon)
    return model(p, ix, horizon)


def relative(pred, truth):
    difference = (pred - truth).flatten(1)
    return difference.norm(dim=1) / truth.flatten(1).norm(dim=1)


def atomic_save(payload, path):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def fit(family, arm, seed, p, split, root):
    n_steps = UPDATES[family]
    horizons = (1, 2) if arm.endswith("T12") else (1,)
    torch.manual_seed(seed)
    model = make_model(family, arm, p)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=2e-6)
    train, validation, _ = split
    rng = np.random.default_rng(98173 + seed)
    draws = np.stack([rng.choice(train, BATCH, replace=False) for _ in range(n_steps)])
    draw_digest = hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()
    indices = torch.as_tensor(draws, dtype=torch.long)
    validation = torch.as_tensor(validation, dtype=torch.long)
    prefix = f"{family}_{arm}_seed{seed}"
    checkpoint = root / (prefix + ".pt")
    resume = root / (prefix + ".resume.pt")
    settings = dict(family=family, arm=arm, seed=seed, updates=n_steps, batch=BATCH,
                    train_horizons=list(horizons), parameter_draw_sha256=draw_digest,
                    precision="float32", initial_lr=.002, final_lr=2e-6, checks=CHECKS,
                    parameter_count=sum(w.numel() for w in model.parameters()))
    if checkpoint.exists():
        final = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert all(final[k] == v for k, v in settings.items())
        model.load_state_dict(final["state"])
        return model, {k: v for k, v in final.items() if k != "state"}
    best = float("inf")
    best_step, start_step, elapsed = 0, 0, 0.
    best_state = None
    history = []
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        assert saved["settings"] == settings
        model.load_state_dict(saved["current"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        best, best_step, best_state = saved["best"], saved["best_step"], saved["best_state"]
        start_step, elapsed, history = saved["step"], saved["elapsed"], saved["history"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(start_step + 1, n_steps + 1):
        ix = indices[step - 1]
        h = horizons[(step - 1) % len(horizons)]
        optimizer.zero_grad(set_to_none=True)
        loss = relative(predict(model, family, p, ix, h), p["Y"][h][ix]).square().mean()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % (n_steps // CHECKS) == 0:
            with torch.no_grad():
                model.eval()
                errors = [torch.cat([relative(predict(model, family, p, v, hv), p["Y"][hv][v])
                                     for v in validation.split(8)]).cpu().numpy() for hv in horizons]
                score = float(np.mean([np.median(err) for err in errors]))
                model.train()
            if not np.isfinite(score) or not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite fit at {prefix} step {step}")
            history.append([step, score])
            if score < best:
                best, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step % (n_steps // 10) == 0:
                seconds = elapsed + time.perf_counter() - start
                print(f"{prefix} {step}/{n_steps} val={score:.6g} elapsed={seconds:.1f}s", flush=True)
                atomic_save(dict(settings=settings, step=step, elapsed=seconds, history=history,
                                 current=model.state_dict(), optimizer=optimizer.state_dict(),
                                 scheduler=schedule.state_dict(), best=best, best_step=best_step,
                                 best_state=best_state), resume)
    torch.cuda.synchronize()
    info = dict(**settings, selected_step=best_step, validation_median=best, validation_history=history,
                fit_seconds=elapsed + time.perf_counter() - start,
                device=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
    atomic_save(dict(**info, state=best_state), checkpoint)
    if resume.exists():
        resume.unlink()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state"])
    return model, info


def row_norm(a):
    return np.linalg.norm(a.reshape(len(a), -1), axis=1)


@torch.no_grad()
def score_model(model, family, p, scale, truth, test, info, root):
    model.eval()
    rows = []
    dt = p["dt"]
    width = round(.5 / dt)
    for h in HORIZONS:
        for start in range(0, len(test), 2):
            ids = test[start:start + 2]
            ix = torch.as_tensor(ids, dtype=torch.long)
            out = predict(model, family, p, ix, h)
            norm = scale[ix].view((-1,) + (1,) * (out.ndim - 1))
            est = (out * norm).cpu().numpy().astype(np.float64)
            ref = truth[h][start:start + len(ids)].astype(np.float64)
            assert est.shape == ref.shape and est.shape[1] == round(h / dt)
            error = est - ref
            columns = [row_norm(error) / row_norm(ref)]
            for sl in (slice(None, width), slice(-width, None)):
                columns.extend([np.sqrt(dt) * row_norm(error[:, sl]), np.sqrt(dt) * row_norm(ref[:, sl])])
            assert np.isfinite(columns).all()
            for row in zip(ids.tolist(), *columns):
                rows.append((info["family"], info["arm"], info["seed"], h, *row))
    prefix = f"{family}_{info['arm']}_seed{info['seed']}"
    dest = root / (prefix + ".csv")
    temp = dest.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "arm", "seed", "T", "test_id", "relative_control_error",
                         "initial_error", "initial_truth", "terminal_error", "terminal_truth"))
        writer.writerows(rows)
    temp.replace(dest)
    (root / (prefix + ".json")).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--arms", nargs="+", choices=COMMON + MOVING_EXTRA)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.out or output_directory("multiseed")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    for family in args.families:
        allowed = COMMON + MOVING_EXTRA if family == "2D-semi-move" else COMMON
        arms = args.arms or allowed
        if not set(arms).issubset(allowed):
            raise ValueError(f"Unsupported arms for {family}: {arms}")
        p, scale, truth, split = load_family(family, args.raw_data / DATASETS[family])
        for arm in arms:
            for seed in args.seeds:
                print(f"Starting {family} {arm} seed {seed}", flush=True)
                model, info = fit(family, arm, seed, p, split, root)
                score_model(model, family, p, scale, truth, split[2], info, root)
                del model
        del p, scale, truth
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

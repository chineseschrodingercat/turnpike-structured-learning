"""Replay all newly fitted structured checkpoints and compare coefficient and FE norms."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.multiseed.run import DATASETS, HORIZONS, SEEDS, load_family, make_model, predict
from experiments.paths import RAW_DATA


def coefficient_norm(a):
    flat = a.reshape(len(a), -1)
    return np.sqrt(np.einsum("ij,ij->i", flat, flat))


def mass_squared(a):
    flat = a.reshape(len(a), -1)
    if a.ndim == 2:
        return np.einsum("ij,ij->i", flat, flat)
    # Consistent P1 mass on the uniform edge. The common h/3 factor cancels.
    return 2 * np.sum(a*a, axis=(1, 2)) + np.sum(a[:, :, 1:] * a[:, :, :-1], axis=(1, 2))


@torch.no_grad()
def replay(family, raw, fits):
    path = raw / DATASETS[family]
    with path.open("rb") as f:
        raw_hash = hashlib.file_digest(f, "sha256").hexdigest()
    p, scale, truth, split = load_family(family, path)
    ids = split[2]
    largest_difference = 0.
    rows = []
    for seed in SEEDS:
        stem = f"{family}_structured_seed{seed}"
        saved = torch.load(fits/(stem + ".pt"), map_location="cpu", weights_only=False)
        model = make_model(family, "structured", p)
        model.load_state_dict(saved["state"])
        model.eval()
        with (fits/(stem + ".csv")).open(encoding="utf-8", newline="") as f:
            logged = {(int(row["T"]), int(row["test_id"])): row for row in csv.DictReader(f)}
        for h in HORIZONS:
            for j in range(0, len(ids), 2):
                ix = torch.as_tensor(ids[j:j+2], dtype=torch.long)
                output = predict(model, family, p, ix, h)
                norm = scale[ix].reshape((-1,) + (1,) * (output.ndim - 1))
                a = (output * norm).cpu().numpy().astype("float64")
                b = truth[h][j:j+len(ix)].astype("float64")
                d = a - b
                whole = coefficient_norm(d) / coefficient_norm(b)
                width = round(.5/p["dt"])
                absolute = [np.sqrt(p["dt"]) * coefficient_norm(x[:, sl]) for sl in (slice(None,width),slice(-width,None)) for x in (d,b)]
                fe = np.sqrt(mass_squared(d) / mass_squared(b))
                if h == 8:
                    mid = slice(round(2/p["dt"]),round(6/p["dt"]))
                    interior = coefficient_norm(d[:,mid]) / coefficient_norm(b[:,mid])
                for k, idx in enumerate(ids[j:j+len(ix)]):
                    ref = logged[h, int(idx)]
                    values = [float(ref[field]) for field in ("relative_control_error","initial_error","initial_truth","terminal_error","terminal_truth")]
                    calculated = np.array([v[k] for v in (whole, *absolute)])
                    np.testing.assert_allclose(calculated, values, rtol=2e-6, atol=1e-8)
                    largest_difference = max(largest_difference, abs(calculated[0]-values[0]))
                    if h == 8:
                        rows.append((family, seed, int(idx), float(whole[k]), float(fe[k]), float(interior[k])))
    return rows, dict(dataset_sha256=raw_hash, split_sizes=list(map(len,split)), replay_values=3*4*len(ids)*5,
                      maximum_absolute_relative_error_difference=float(largest_difference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    rows = []
    audit = {}
    for family in DATASETS:
        checked, record = replay(family, args.raw_data, args.fits)
        rows.extend(checked)
        metrics = np.asarray([r[-3:] for r in checked]) * 100
        audit[family] = dict(**record, median_pct=np.median(metrics,axis=0).tolist(), max_pct=metrics.max(axis=0).tolist())
    with (args.out/"fe_norms.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(("family","seed","test_id","coefficient_relative_error","fe_relative_error","interior_relative_error"))
        writer.writerows(rows)
    (args.out/"checkpoint_replay.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8",newline="\n")


if __name__ == "__main__":
    main()

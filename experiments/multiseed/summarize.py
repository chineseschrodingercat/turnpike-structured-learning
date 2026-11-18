"""Rebuild the semilinear tables and window bands from the public case errors."""
import argparse
import csv
import gzip
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


FAMILIES = ("2D-semi", "2D-semi-strip", "2D-semi-move")
HORIZONS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
BASE = ("structured", "cnn_T1", "cnn_T12")
EXTRA = ("global", "residual_T1", "residual_T12")
PUBLIC = Path(__file__).resolve().parents[2] / "results/semilinear_multiseed"
FIELDS = ("relative_control_error", "initial_error", "initial_truth", "terminal_error", "terminal_truth")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def assemble(fit_dir, output):
    configs = []
    destination = output / "case_errors.csv.gz"
    with destination.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("family", "arm", "seed", "T", "test_id", *FIELDS))
            for family in FAMILIES:
                arms = BASE + EXTRA if family == "2D-semi-move" else BASE
                for arm in arms:
                    for seed in SEEDS:
                        stem = f"{family}_{arm}_seed{seed}"
                        config = json.loads((fit_dir / (stem + ".json")).read_text(encoding="utf-8"))
                        assert (config["family"], config["arm"], config["seed"]) == (family, arm, seed)
                        configs.append(config)
                        with (fit_dir / (stem + ".csv")).open(encoding="utf-8", newline="") as source:
                            reader = csv.DictReader(source)
                            assert tuple(reader.fieldnames) == ("family", "arm", "seed", "T", "test_id", *FIELDS)
                            writer.writerows([row[key] for key in reader.fieldnames] for row in reader)
    write_json(output / "fit_settings.json", configs)


def load_cases(path):
    cases = defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            group = (row["family"], row["arm"], int(row["seed"]), int(row["T"]))
            idx = int(row["test_id"])
            assert idx not in cases[group]
            values = [float(row[f]) for f in FIELDS]
            assert np.isfinite(values).all() and min(values) >= 0 and values[2] > 0 and values[4] > 0
            cases[group][idx] = values
    return cases


def summarize_arm(array, configs):
    # array axes: seed, horizon, physical parameter, measurement
    whole = 100 * array[:, :, :, 0]
    whole_summary = {}
    windows = {}
    for j, h in enumerate(HORIZONS):
        w = whole[:, j]
        whole_summary[h] = {"pooled_median_pct": float(np.median(w)), "pooled_max_pct": float(w.max()),
                            "seed_median_pct": np.median(w, axis=1).tolist(), "seed_max_pct": w.max(axis=1).tolist()}
    for name, pos in (("initial", 1), ("terminal", 3)):
        error, truth = array[:, :, :, pos], array[:, :, :, pos + 1]
        # Keep the same physical parameters at every horizon. Each paired denominator
        # must pass the article's short-window filter for every one of the three fits.
        eligible = (error[:, 0] > 1e-6 * truth[:, 0]).all(axis=0)
        assert eligible.any()
        ratios = error[:, :, eligible] / error[:, 0:1, eligible]
        per_parameter_ratio = np.median(ratios, axis=0)
        rel = np.median(100 * error / truth, axis=0)
        windows[name] = {
            "ratio_eligible_n": int(eligible.sum()),
            "quantiles": {h: np.quantile(per_parameter_ratio[j], [.1, .5, .9]).tolist()
                          for j, h in enumerate(HORIZONS)},
            "window_median_pct": {h: float(np.median(rel[j])) for j, h in enumerate(HORIZONS)},
        }
    return {"n_test_parameters": array.shape[2], "n_seeds": array.shape[0],
            "training_horizons": configs[0]["train_horizons"], "updates": configs[0]["updates"],
            "batch": configs[0]["batch"], "parameter_count": configs[0]["parameter_count"],
            "fit_seconds": [c["fit_seconds"] for c in configs], "whole": whole_summary, "windows": windows}


def summarize(source):
    cases = load_cases(source / "case_errors.csv.gz")
    configs = json.loads((source / "fit_settings.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(configs) == 36 and len(cases) == 36 * len(HORIZONS)
    settings = {(c["family"], c["arm"], c["seed"]): c for c in configs}
    out = {}
    for family in FAMILIES:
        arms = BASE + EXTRA if family == "2D-semi-move" else BASE
        ids = sorted(cases[family, "structured", 0, 1])
        assert len(ids) == 100 and ids == sorted(manifest[family]["test_idx"])
        for seed in SEEDS:
            paired = [settings[family, arm, seed] for arm in arms]
            for field in ("updates", "batch", "parameter_draw_sha256", "precision", "initial_lr", "final_lr", "checks"):
                assert len({c[field] for c in paired}) == 1, (family, seed, field)
        results = {}
        paired_errors = {}
        for arm in arms:
            data = np.empty((len(SEEDS), len(HORIZONS), len(ids), len(FIELDS)))
            for i, seed in enumerate(SEEDS):
                for j, h in enumerate(HORIZONS):
                    row = cases[family, arm, seed, h]
                    assert sorted(row) == ids
                    data[i, j] = [row[idx] for idx in ids]
            results[arm] = summarize_arm(data, [settings[family, arm, seed] for seed in SEEDS])
            paired_errors[arm] = np.median(data[:, -1, :, 0], axis=0)
        out[family] = {"arms": results, "test_ids": ids,
                       "n_parameters_structured_lower_T8": {
                           arm: int((paired_errors["structured"] < paired_errors[arm]).sum())
                           for arm in arms if arm != "structured"}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PUBLIC, help="Published CSV and fit settings")
    parser.add_argument("--fits", type=Path, help="Assemble the source records from this completed fit directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fits:
        assemble(args.fits, args.out)
    elif args.out.resolve() != args.source.resolve():
        for name in ("case_errors.csv.gz", "fit_settings.json"):
            shutil.copyfile(args.source / name, args.out / name)
    if args.out.resolve() != args.source.resolve():
        shutil.copyfile(args.source / "dataset_manifest.json", args.out / "dataset_manifest.json")
    write_json(args.out / "summary.json", summarize(args.out))


if __name__ == "__main__":
    main()

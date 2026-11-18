# Turnpike-Structured Learning for Horizon Extrapolation in PDE-Constrained Optimal Control

This repository contains the computational source for manuscript V1.2. The method learns initial and terminal corrections to a computed steady or periodic control reference, then evaluates them at new horizons.

The public release for the three-seed comparisons is tagged **v1.2-multiseed**.
The original full-batch timing fits and solvers are also preserved in **v1.2-code**.

## Quick start

Use Python 3.13. The recorded V1.2 environments used Python 3.13.9. From the repository root, run

```text
python -m pip install -r requirements.txt
python quick_start.py
```

The quick start generates a 12-parameter linear Robin-control dataset, fits a small TurnpikeNet example using four training epochs at T = 1, and evaluates it through T = 8. It runs on CPU and saves the summary and held-out predictions in `artifacts/quick_start`. These quantities are from the small example. The manuscript uses the datasets, splits, fit budgets and reporting protocols specified in its supplement.

## Source and inputs

| Location | Contents |
|---|---|
| `experiments/multiseed/` | Matched training of the three semilinear 2D families, endpoint-window evaluation, finite-element norm replay, plotting and aggregation. |
| `results/semilinear_multiseed/` | Case errors, fitting configurations, dataset hashes, split IDs, finite-element errors and paired query timings. |
| `exp/robin/` | PDE discretizations, optimal-control solvers, exact endpoint and stitching baselines, synthetic label generators, structured and generic learners, evaluations and figure scripts. Use `--help` for command-line entry points. The optional external cross-check in `validate_robin.py` uses the upstream `ctrl1d` package. |
| `experiments/rebuild/` | V1.2 neural fits for the main semilinear 2D and generic comparison rows, and the paired nonlinear cost calculation and verification. |
| `experiments/matched/` | V1.2 constrained and strip fits, fresh-cohort reconstruction, and linear paired calculations. |
| `mpc/` | Recorded MPC verification code and the exact server-run source and protocol. |
| `SOURCE_PROVENANCE.json` | Origins and hashes of the original computational source files. |

### Reproduce the semilinear accuracy tables and figures

The published case errors cover four horizons, three seeds and 100 distinct test parameters per family. The whole-control tables pool the parameter–seed evaluations. The fixed-window curves first take the median of the three paired ratios at each physical parameter and then summarize the parameters.

```text
python -m experiments.multiseed.summarize --out artifacts/table_readback
python -m experiments.multiseed.plot --results results/semilinear_multiseed --out artifacts/figures
```

Both commands use the published records and run on CPU. The plot command uses Matplotlib with Arial. The fresh-query records use the full-batch checkpoints listed in the paper's timing experiment.

To refit all 36 networks on a CUDA device, set `TURNPIKE_RAW_DATA` to the location of the original synthetic labels and run the following commands. The dataset manifest specifies the exact input files, checksums and split IDs. The complete label arrays and fitted weights are available from the authors on request.

```text
python -m experiments.multiseed.run
python -m experiments.multiseed.summarize --fits artifacts/multiseed --out artifacts/refit_tables
python -m experiments.multiseed.verify_norms --fits artifacts/multiseed --out artifacts/refit_norms
```

### Full-batch fits and paired timing experiments

The fitting drivers expect the original synthetic-label files in the author-supplied experiment archive. Set `TURNPIKE_RAW_DATA` to its `Raw Data` directory containing `01 - Initial delivery` and `02 - Supplement`. Set `TURNPIKE_OUTPUT` to an output base directory if desired. The default output base is `artifacts`. The full-size V1.2 fitting drivers require a CUDA-capable PyTorch installation.

Run the commands from the repository root. For the semilinear 2D models and paired online timing, use the commands in the following order.

```text
python -m experiments.rebuild.rebuild_structured_full
python -m experiments.rebuild.rebuild_moving_full
python -m experiments.rebuild.nonlinear_paired_cost
python -m experiments.rebuild.verify_paired_arrays
```

Use `python -m experiments.rebuild.rebuild_generic` for the V1.2 FNO and DeepONet fits. Use `python -m experiments.matched.rebuild_cnns` followed by `python -m experiments.matched.rebuild_box_matched700` for the fixed 700-parameter box comparison. For the linear paired-cost calculations, run `python -m experiments.matched.reconstruct` to produce the reference checkpoint and kernel state, then `python -m experiments.matched.paired_baseline` and `python -m experiments.matched.paired_cnn_extension` after the CNN fits. Keep the output directory associated with the code snapshot and configuration used to produce it. Timing depends on the numerical environment and hardware.

### MPC replay and audit

The separate complete V1.2 MPC archive contains the exact scenario files, weights, trajectory references, manifest and returned records. Run its `server_package/run_mpc.py --stage full --suite core --workers 1 --dry-run` to check the fixed inputs and enumerate the full suite before execution. Use the `timing` stage with one worker for sequential timing. The verifier in this repository recomputes the paper's statistics from an existing return.

```text
python mpc/verify_mpc_return.py --delivery "/path/to/server_results" --package "/path/to/server_package" --out artifacts/mpc_verification
```

The repository includes the MPC run configuration and compact per-case results for the semilinear study. The complete label arrays, model checkpoints and detailed solver outputs are available from the authors on request.

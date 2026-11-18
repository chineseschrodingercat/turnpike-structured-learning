# Paired semilinear comparison

The fixed primary outcome is the relative coefficient-norm control error at T = 8. The same 100 held-out physical parameters are evaluated for each seed and model in each family. The prespecified aggregate is the median (maximum) of all 300 parameter–seed evaluations. Per-seed medians are reported to show training variability. The physical test parameters are shared across seeds.

| Family | Train / validation / test | Adam updates per fit | Parameters per update |
|---|---|---:|---:|
| 2D-semi | 450 / 100 / 100 | 6,000 | 32 |
| 2D-semi-strip | 550 / 100 / 100 | 3,000 | 32 |
| 2D-semi-move | 550 / 100 / 100 | 20,000 | 32 |

Each family has new fits of TurnpikeNet at T = 1 and the unstructured temporal CNN at T = 1 and T = {1, 2}. The moving family additionally has the global harmonic model at T = 1 and the reference-residual CNN at T = 1 and T = {1, 2}. All arms use seeds 0, 1 and 2. A seed determines the network initialization and a separate fixed sequence of training-parameter draws, shared by every arm at that seed. Each update samples 32 distinct training parameters. Dual-horizon arms alternate T = 1 and T = 2, one complete trajectory per draw. This fixes optimizer steps and parameter appearances. Runtime and temporal grid-point counts differ with the architecture and training horizons.

All fits use float32 network tensors. CUDA allows TF32 for convolutions and uses the IEEE float32 setting for matrix products. The optimizer uses Adam with an initial learning rate of 0.002 and cosine decay to 0.000002. Forty equally spaced validation checks are scheduled before training. The selected checkpoint minimizes the median relative error on the 100 validation parameters, averaged over the training horizons when both are used. The four held-out horizons are scored after checkpoint selection. Original physical parameters, labels, split IDs, and train-set input normalization are retained. Moving periodic reference construction uses double precision before conversion to the neural tensor type.

The endpoint comparison uses a physical window of 0.5 at each end. Compute each horizon-to-one ratio for the same seed and parameter. When computing ratio quantiles, keep a physical parameter if its T = 1 window-relative error exceeds 10⁻⁶ for every seed. Take the median across the three seeds for each retained physical parameter and then compute the 10th, 50th and 90th percentiles across parameters. Window-relative errors themselves use all 100 parameters. The example profiles and paired online cost experiments use separately identified full-batch checkpoints. The finite-element norm comparison replays the matched structured fits.

The original single-seed results and earlier historical multi-seed predictions are retained for provenance. No checkpoint, horizon, or training budget in the paired comparison is selected on the basis of its T = 8 result.

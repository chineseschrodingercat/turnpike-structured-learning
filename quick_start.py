"""Run a small 1D label-generation, fitting and horizon-transfer example."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main() -> None:
    output = ROOT / "artifacts" / "quick_start"
    solver = ROOT / "exp" / "robin"
    dataset = output / "data.npz"
    summary = output / "results.json"
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = "1"

    subprocess.run(
        [sys.executable, str(solver / "gen_turnpike_data.py"), "--out", str(dataset),
         "--n", "12", "--workers", "1"],
        cwd=ROOT, env=env, check=True,
    )
    subprocess.run(
        [sys.executable, str(solver / "baselines_multihorizon.py"), "--kind", "const",
         "--data", str(dataset), "--out", str(summary), "--models", "turnpike",
         "--train-T", "1", "--seeds", "0", "--epochs", '{"turnpike": 4}',
         "--threads", "1", "--device", "cpu", "--save-pred", str(output / "predictions")],
        cwd=ROOT, env=env, check=True,
    )
    run = json.loads(summary.read_text(encoding="utf-8"))["models"]["turnpike"]["runs"][0]
    assert set(run["test"]) == {"1", "2", "4", "8"}
    assert all(math.isfinite(scores["median"]) for scores in run["test"].values())
    print(f"Quick start passed. Results: {summary}")

if __name__ == "__main__":
    main()

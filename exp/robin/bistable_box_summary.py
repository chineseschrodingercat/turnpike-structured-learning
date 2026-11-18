#!/usr/bin/env python3
"""Summary table of the bistable / box pilot runs: J values, |u - u_ref|/|u_ref|, rank@1e-3, strip tracking errors."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_bistable_box_gt import load, rel_track, target_of  # noqa: E402


def rank(Z):
    sv = np.linalg.svd(Z.astype(float), compute_uv=False)
    return int(np.sum(sv > 1e-3 * sv[0])), float(sv[1] / sv[0])


def main():
    d = pathlib.Path(sys.argv[1])
    rows = []
    for p in sorted(d.glob("*.npz")):
        z = load(p)
        cf, C, Cr, inf, m = z["cf"], z["C"], z["C_ref"], z["info"], z["meta"]
        tf = target_of(z)
        r, r_ref = rank(C), rank(Cr)
        rows.append({"case": p.stem, "alpha": cf.alpha, **m, "J": inf["J"],
                     "J_ref_on_plant": inf.get("J_lq_on_plant", inf.get("J_unc")), "J_linear": inf.get("J_lq_linear", inf.get("J_unc")),
                     "J_clip": inf.get("J_clip_unc"), "diff_vs_ref": float(np.linalg.norm(C - Cr) / np.linalg.norm(Cr)),
                     "rank": r[0], "s2s1": r[1], "rank_ref": r_ref[0], "s2s1_ref": r_ref[1],
                     "track": float(rel_track(cf, z["strip"], tf)), "track_ref": float(rel_track(cf, z["strip_ref"], tf)),
                     "max_u": float(np.abs(C).max()), "max_u_ref": float(np.abs(Cr).max()),
                     "newton_iters": inf.get("newton_iters"), "neg_curv": inf.get("neg_curv"), "seconds": inf.get("total_seconds")})
    (d / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    for r in rows:
        print(f"{r['case']:28s} a={r['alpha']:.0e} J={r['J']:.5g} Jref@plant={r['J_ref_on_plant']:.5g} Jlin={r['J_linear']:.5g} "
              f"diff={100 * r['diff_vs_ref']:.0f}% rank={r['rank']}/{r['rank_ref']} s2s1={r['s2s1']:.2f}/{r['s2s1_ref']:.2f} "
              f"track={100 * r['track']:.0f}%/{100 * r['track_ref']:.0f}% maxu={r['max_u']:.1f}/{r['max_u_ref']:.1f} "
              f"it={r['newton_iters']} neg={r['neg_curv']} t={r['seconds'] or 0:.0f}s")


if __name__ == "__main__":
    main()

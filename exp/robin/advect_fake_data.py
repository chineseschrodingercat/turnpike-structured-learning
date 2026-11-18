#!/usr/bin/env python3
"""Tiny random dataset with the layout of gen_advect_data.py --merge, for smoke-testing advect_surrogate.py."""

import json
import sys

import numpy as np

sys.path.insert(0, "exp/robin")
from gen_advect_data import FIX, MU_NAMES, sample_mu  # noqa: E402

rng = np.random.default_rng(0)
n_tr, n_va, n_te, m, nh = 12, 4, 4, 99, 20
n = n_tr + n_va + n_te
uhat = np.zeros((n, 80, m), np.complex64)
uhat[:, :nh] = (rng.standard_normal((n, nh, m)) + 1j * rng.standard_normal((n, nh, m))) / np.arange(1, nh + 1)[None, :, None]
np.savez(sys.argv[1], X=sample_mu(rng, n), mu_names=np.array(MU_NAMES), uhat=uhat, n_harm=np.full(n, nh - 1),
         U_T2=rng.standard_normal((n, 640, m)).astype(np.float32), U_T4_test=rng.standard_normal((n_te, 1280, m)).astype(np.float32),
         J_T2=np.ones(n), J_T4_test=np.ones(n_te), iters_T2=np.ones(n, int), hb_seconds=np.ones(n), pcg_seconds_T2=np.ones(n),
         pcg_seconds_T4_test=np.ones(n_te), train_idx=np.arange(n_tr), val_idx=np.arange(n_tr, n_tr + n_va),
         test_idx=np.arange(n_tr + n_va, n), dt=1.0 / 320, m=m, fixed=json.dumps(FIX))

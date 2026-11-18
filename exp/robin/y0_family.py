#!/usr/bin/env python3
"""Initial states for the y0-dependent experiments (1D, free dofs x_i = i h, i = 0..nx-1, y(1) = 0).

Sampled family (T_ref = 1 units): y0 = s * y_bar(mu) + sum_{k=1}^{K} c_k phi_k(x),
    phi_k(x) = cos((k - 1/2) pi x)   (vanishes at x = 1),
    s ~ U(0, 1.5),  c_k ~ N(0, (amp / k)^2),  K = 12, amp = 0.4.
s * y_bar covers 'starting from 0' (s = 0), 'at the turnpike' (s = 1) and overshoot; the cosine part is a smooth
random disturbance. Network input: M-orthogonal projection of y0 - y_bar onto the first K_in cosines.
"""

from __future__ import annotations

import numpy as np

K_SAMPLE, AMP, K_IN = 12, 0.4, 16


def basis(x: np.ndarray, K: int) -> np.ndarray:
    return np.cos((np.arange(1, K + 1)[None, :] - 0.5) * np.pi * x[:, None])


def sample(rng: np.random.Generator, x: np.ndarray, ybar: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    s = rng.uniform(0.0, 1.5)
    c = rng.normal(0.0, AMP / np.arange(1, K_SAMPLE + 1))
    return s * ybar + basis(x, K_SAMPLE) @ c, s, c


class Projector:
    """z = argmin ||Phi z - d||_M (M = consistent mass matrix)."""

    def __init__(self, x: np.ndarray, M, K: int = K_IN):
        self.Phi = basis(x, K)
        MP = M @ self.Phi
        self.G = np.linalg.cholesky(self.Phi.T @ MP)
        self.MP = MP

    def __call__(self, d: np.ndarray) -> np.ndarray:
        rhs = d @ self.MP  # works for (n,) or (B, n)
        return np.linalg.solve(self.G.T, np.linalg.solve(self.G, rhs.T)).T

    def back(self, z: np.ndarray) -> np.ndarray:
        return z @ self.Phi.T

"""Polynomial selection by leave-one-out prediction on the archived sweep."""

from __future__ import annotations

import numpy as np


def fit_candidates(x, y, degrees=(1, 2, 3, 4)) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(np.unique(x)) != len(x) or not np.isfinite(y).all():
        raise ValueError('Expected finite paired observations at distinct frequencies')
    candidates = {}
    denominator = float(np.sum((y - y.mean()) ** 2))
    for degree in degrees:
        if degree >= len(x) - 1:
            raise ValueError('Insufficient observations for leave-one-out fit')
        coefficients = np.polyfit(x, y, degree)
        residual = y - np.polyval(coefficients, x)
        loo_residual = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            fit = np.polyfit(x[keep], y[keep], degree)
            loo_residual.append(y[i] - np.polyval(fit, x[i]))
        loo_residual = np.asarray(loo_residual)
        candidates[str(degree)] = {
            'coefficients': coefficients.tolist(),
            'r2': 1 - float(np.sum(residual ** 2)) / denominator,
            'loo_r2': 1 - float(np.sum(loo_residual ** 2)) / denominator,
            'loo_rmse': float(np.sqrt(np.mean(loo_residual ** 2))),
        }
    selected = min(candidates, key=lambda k: candidates[k]['loo_rmse'])
    return {'selected_degree': int(selected), 'candidates': candidates}


def bracketed_root(fit: dict, target: float, bracket) -> float:
    coefficients = np.array(fit['candidates'][str(fit['selected_degree'])]['coefficients'])
    coefficients[-1] -= target
    left, right = map(float, bracket)
    candidates = [float(r.real) for r in np.roots(coefficients)
                  if abs(r.imag) < 1e-9 and left <= r.real <= right]
    if len(candidates) != 1:
        raise ValueError(f'Expected one root in {bracket}, found {candidates}')
    return candidates[0]

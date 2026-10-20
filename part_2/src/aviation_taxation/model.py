from __future__ import annotations
from dataclasses import dataclass
import numpy as np
NSLOTS = 3

@dataclass(frozen=True)
class FractionalParams:
    T: float = 1.0
    d: float = 0.4
    N: int = 5
    a: float = 1.0
    lambda_bar: float = 5.0
    c: float = 0.0
    tau: float = 0.0
    ticket_tax: float = 0.0
    kappa: float = 1.0

    @property
    def M(self) -> float:
        return self.T / self.d

    @property
    def entry_phi(self) -> float:
        return 3.0 - self.M

def validate(params: FractionalParams) -> None:
    if not 2.0 < params.M <= 3.0:
        raise ValueError(f'shared requires M=T/d in (2,3], got {params.M:.6f}')
    if not 0.0 <= params.entry_phi < 1.0:
        raise ValueError(f'entry_phi must be in [0,1), got {params.entry_phi:.6f}')

def active_mask(phi, params: FractionalParams):
    """Return active mask `[True, True, active2]` for scalar or array phi."""
    validate(params)
    phi_arr = np.asarray(phi)
    active2 = phi_arr >= params.entry_phi
    ones = np.ones_like(active2, dtype=bool)
    return np.stack([ones, ones, active2], axis=-1)

def departure_offsets(phi, params: FractionalParams):
    """Departure offsets from current time for rank slots 0, 1, 2."""
    phi_arr = np.asarray(phi, dtype=float)
    return np.stack([(1.0 - phi_arr) * params.d, (2.0 - phi_arr) * params.d, (3.0 - phi_arr) * params.d], axis=-1)

def _branch_integral(x0, x1, t_r, p_r, a, kappa=1.0):
    """Integral of exp(-a (p_r + kappa*|tp - t_r|)) over [x0, x1].

    The schedule-delay disutility scale `kappa` enters the closed form as a
    denominator factor (a -> a*kappa) and scales the exponent arguments; the
    price prefactor exp(-a*p_r) is unaffected (price is not displaced)."""
    if x1 <= x0:
        return 0.0
    ak = a * kappa
    scale = np.exp(-a * p_r)
    left0, left1 = (x0, min(x1, t_r))
    right0, right1 = (max(x0, t_r), x1)
    value = 0.0
    if left1 > left0:
        value += scale / ak * (np.exp(-ak * (t_r - left1)) - np.exp(-ak * (t_r - left0)))
    if right1 > right0:
        value += scale / ak * (np.exp(-ak * (right0 - t_r)) - np.exp(-ak * (right1 - t_r)))
    return float(value)

def demand_M3(p0, p1, p2, phi, n0, n1, n2, params: FractionalParams):
    """Closed-form purchase probabilities (D0, D1, D2).

    Uses the lower envelope of the V-shaped effective prices
    `q_j(tp) = p_j + |tp - t_j|`. The winning firm changes only at pairwise
    crossings, so partitioning [0,T] at those crossings and classifying each
    sub-interval by its midpoint winner integrates each firm's demand exactly.
    Only active, stocked firms participate; the rest post no effective price.
    """
    validate(params)
    t = departure_offsets(float(phi), params)
    mask = active_mask(float(phi), params)
    p = [float(p0), float(p1), float(p2)]
    n = [int(n0), int(n1), int(n2)]
    part = [j for j in range(NSLOTS) if bool(mask[j]) and n[j] > 0]
    D = [0.0, 0.0, 0.0]
    if not part:
        return (0.0, 0.0, 0.0)
    T, a, kappa = (params.T, params.a, float(params.kappa))
    breakpoints = {0.0, T}
    for ii in range(len(part)):
        for jj in range(ii + 1, len(part)):
            i, j = (part[ii], part[jj])
            tc = (t[i] + t[j]) / 2.0 + (p[j] - p[i]) / (2.0 * kappa)
            breakpoints.add(min(max(float(tc), 0.0), T))
    bps = sorted(breakpoints)
    for k in range(len(bps) - 1):
        x0, x1 = (bps[k], bps[k + 1])
        if x1 <= x0:
            continue
        mid = 0.5 * (x0 + x1)
        w = min(part, key=lambda j: p[j] + kappa * abs(mid - float(t[j])))
        D[w] += _branch_integral(x0, x1, float(t[w]), p[w], a, kappa)
    return (D[0] / T, D[1] / T, D[2] / T)

def brute_force_demand(p0, p1, p2, phi, n0, n1, n2, params: FractionalParams, n_grid: int=200000):
    """Numerical quadrature reference for tests."""
    t = departure_offsets(float(phi), params)
    mask = active_mask(float(phi), params)
    p = [float(p0), float(p1), float(p2)]
    n = [int(n0), int(n1), int(n2)]
    part = [j for j in range(NSLOTS) if bool(mask[j]) and n[j] > 0]
    if not part:
        return (0.0, 0.0, 0.0)
    kappa = float(params.kappa)
    xs = np.linspace(0.0, params.T, n_grid)
    q = np.full((NSLOTS, n_grid), np.inf)
    for j in part:
        q[j] = p[j] + kappa * np.abs(xs - float(t[j]))
    winner = np.argmin(q, axis=0)
    D = [0.0, 0.0, 0.0]
    for j in part:
        u = np.exp(-params.a * q[j])
        D[j] = float(np.trapezoid(np.where(winner == j, u, 0.0), xs) / params.T)
    return (D[0], D[1], D[2])

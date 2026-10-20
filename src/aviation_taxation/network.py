from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from aviation_taxation.model import FractionalParams, validate
NSLOTS = 3

class FractionalDEQN(nn.Module):
    """Shared rank-indexed network with a padded three-slot state."""

    def __init__(self, hidden: int=128, activation: str='relu'):
        super().__init__()
        self.activation = activation
        act = {'relu': nn.ReLU, 'silu': nn.SiLU}[activation]
        self.body = nn.Sequential(nn.Linear(7, hidden), act(), nn.Linear(hidden, hidden), act())
        self.price_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.softplus = nn.Softplus()

    def forward(self, phi, rank, n, active2, M, N):
        if rank.ndim == 0:
            rank = rank.expand_as(phi)
        if active2.ndim == 0:
            active2 = active2.expand_as(phi)
        if not torch.is_tensor(M):
            M = torch.full_like(phi, float(M))
        x = torch.cat([phi.unsqueeze(-1), rank.float().unsqueeze(-1), n / float(N), active2.float().unsqueeze(-1), M.unsqueeze(-1)], dim=-1)
        h = self.body(x)
        return (self.softplus(self.price_head(h)).squeeze(-1), self.value_head(h).squeeze(-1))

def _device(net: nn.Module) -> torch.device:
    return next(net.parameters()).device

def active2_torch(phi, params: FractionalParams):
    validate(params)
    return phi >= float(params.entry_phi)

def active_rank_mask(phi, n, params: FractionalParams):
    a2 = active2_torch(phi, params)
    ones = torch.ones_like(a2, dtype=torch.bool)
    return torch.stack([ones, ones, a2], dim=-1)

def rank_values(net: FractionalDEQN, phi, n, params: FractionalParams):
    B = phi.shape[0]
    a2 = active2_torch(phi, params).to(phi.dtype)
    ps, Vs = ([], [])
    for r in range(NSLOTS):
        rr = torch.full((B,), float(r), dtype=phi.dtype, device=phi.device)
        p_r, V_r = net(phi, rr, n, a2, params.M, params.N)
        ps.append(p_r)
        Vs.append(V_r)
    return (torch.stack(ps, dim=-1), torch.stack(Vs, dim=-1))

def _branch_integral_torch(x0, x1, tr, pr, a, kappa=1.0):
    """Integral of exp(-a (pr + kappa*|tp - tr|)) over [x0, x1]; 0 for empty intervals.

    `kappa` (schedule-delay disutility scale) enters as a denominator factor
    (a -> a*kappa) and scales the exponent arguments; the price prefactor is
    unaffected. Mirrors `model._branch_integral`."""
    ak = float(a) * float(kappa)
    scale = torch.exp(-float(a) * pr)
    left0 = x0
    left1 = torch.minimum(x1, tr)
    has_left = (left1 > left0).to(pr.dtype)
    left = scale / ak * (torch.exp(-ak * (tr - left1)) - torch.exp(-ak * (tr - left0))) * has_left
    right0 = torch.maximum(x0, tr)
    right1 = x1
    has_right = (right1 > right0).to(pr.dtype)
    right = scale / ak * (torch.exp(-ak * (right0 - tr)) - torch.exp(-ak * (right1 - tr))) * has_right
    return left + right

def _winning_intervals_torch(p, phi, n, params: FractionalParams):
    """Shared geometry for demand and displacement: vertices, participation, and
    the ordered winning intervals [0,R0] (rank 0), [R0,L2] (rank 1), [L2,T] (rank
    2) from the lower envelope of q_j(tp) = p_j + kappa*|tp - t_j|."""
    d, T, kappa = (float(params.d), float(params.T), float(params.kappa))
    t0 = (1.0 - phi) * d
    t1 = (2.0 - phi) * d
    t2 = (3.0 - phi) * d
    active2 = active2_torch(phi, params)
    part0 = n[:, 0] > 0.5
    part1 = n[:, 1] > 0.5
    part2 = active2 & (n[:, 2] > 0.5)
    BIG = 1000.0

    def sentinel(col, part):
        return col + torch.where(part, torch.zeros_like(col), torch.full_like(col, BIG))
    pe0 = sentinel(p[:, 0], part0)
    pe1 = sentinel(p[:, 1], part1)
    pe2 = sentinel(p[:, 2], part2)

    def boundary(raw, ti, tj):
        hi = torch.full_like(raw, T)
        zero = torch.zeros_like(raw)
        x = torch.where(raw >= tj, hi, torch.where(raw <= ti, zero, raw))
        return torch.clamp(x, min=0.0, max=T)
    x01 = boundary((t0 + t1) / 2.0 + (pe1 - pe0) / (2.0 * kappa), t0, t1)
    x12 = boundary((t1 + t2) / 2.0 + (pe2 - pe1) / (2.0 * kappa), t1, t2)
    x02 = boundary((t0 + t2) / 2.0 + (pe2 - pe0) / (2.0 * kappa), t0, t2)
    R0 = torch.minimum(x01, x02)
    L2 = torch.maximum(x12, x02)
    lo = torch.zeros_like(phi)
    hiT = torch.full_like(phi, T)
    return (t0, t1, t2, R0, L2, lo, hiT, part0, part1, part2)

def demand_torch(p, phi, n, params: FractionalParams):
    """Differentiable 3-firm purchase probabilities for active, stocked ranks.

    Vertices t0<t1<t2; the lower envelope gives ordered winning intervals split
    at the pairwise crossings b01,b12, with the dominated-middle case (b01>=b12)
    splitting ranks 0 and 2 at b02 instead. Non-participants (inactive or
    stocked out) are pushed out with a large sentinel price so a single formula
    covers the 2-firm subphase, 3-firm subphase, and any stockout combination.
    """
    a, kappa, T = (params.a, float(params.kappa), float(params.T))
    t0, t1, t2, R0, L2, lo, hiT, part0, part1, part2 = _winning_intervals_torch(p, phi, n, params)
    D0 = _branch_integral_torch(lo, R0, t0, p[:, 0], a, kappa) / T
    D1 = _branch_integral_torch(R0, L2, t1, p[:, 1], a, kappa) / T
    D2 = _branch_integral_torch(L2, hiT, t2, p[:, 2], a, kappa) / T
    z = torch.zeros_like(D0)
    D0 = torch.where(part0, D0, z)
    D1 = torch.where(part1, D1, z)
    D2 = torch.where(part2, D2, z)
    return torch.stack([D0, D1, D2], dim=-1)

def fischer_burmeister(x, y, eps=1e-08):
    return x + y - torch.sqrt(x * x + y * y + eps)

def _post_boundary_state(n, params: FractionalParams):
    """At the period boundary: (n0,n1,n2) -> (n1, n2, fresh).

    fresh = N for the synchronized M=3 limit (entry coincides with departure),
    else 0 (the new rank-2 slot is padded until the next entry).
    """
    if params.entry_phi == 0.0:
        fresh = torch.full_like(n[:, 2:3], float(params.N))
    else:
        fresh = torch.zeros_like(n[:, 2:3])
    return torch.cat([n[:, 1:2], n[:, 2:3], fresh], dim=-1)

def _apply_entry_if_crossed(n, phi, phi_next, params: FractionalParams):
    if params.entry_phi == 0.0:
        return n
    crosses = (phi < float(params.entry_phi)) & (phi_next >= float(params.entry_phi))
    n_entry = n.clone()
    n_entry[:, 2] = float(params.N)
    return torch.where(crosses.unsqueeze(-1), n_entry, n)

def successor_value(net, phi, r: int, n, params: FractionalParams, dphi: float):
    """One-step continuation value with entry and phase-boundary routing.

    Interior: advance phi by dphi (injecting rank-2 entry if crossed). At the
    period boundary (phi_next >= 1) rank 0 departs (terminal value 0, subsidy
    form) and each surviving rank r>=1 advances to rank r-1 carrying its
    inventory, valued at the post-boundary state.
    """
    phi_next_raw = phi + float(dphi)
    interior = phi_next_raw < 1.0
    phi_next = torch.clamp(phi_next_raw, max=1.0)
    n_next = _apply_entry_if_crossed(n, phi, phi_next, params)
    a2_next = active2_torch(phi_next, params).to(phi.dtype)
    rr = torch.full_like(phi, float(r))
    _, V_interior = net(phi_next, rr, n_next, a2_next, params.M, params.N)
    if r == 0:
        V_boundary = torch.zeros_like(phi)
    else:
        n_post = _post_boundary_state(n, params)
        phi0 = torch.zeros_like(phi)
        r_adv = torch.full_like(phi, float(r - 1))
        a2_0 = active2_torch(phi0, params).to(phi.dtype)
        _, V_boundary = net(phi0, r_adv, n_post, a2_0, params.M, params.N)
    return torch.where(interior, V_interior, V_boundary)

def sample_states(B: int, params: FractionalParams, eps: float=0.001, device=None):
    validate(params)
    if device is None:
        device = torch.device('cpu')
    phi = eps + torch.rand(B, device=device) * (1.0 - 2.0 * eps)
    n = torch.randint(0, params.N + 1, (B, NSLOTS), device=device).float()
    inactive = ~active2_torch(phi, params)
    n[inactive, 2] = 0.0
    return (phi, n)

def _sample_phi_band(count: int, center: float, half_width: float, lo: float, hi: float, device) -> torch.Tensor:
    left = max(lo, center - half_width)
    right = min(hi, center + half_width)
    if count <= 0:
        return torch.empty(0, device=device)
    if right <= left:
        return torch.full((count,), float(np.clip(center, lo, hi)), device=device)
    return left + torch.rand(count, device=device) * (right - left)

def sample_interior(B: int, params: FractionalParams, dphi: float, device=None, two_firm_fraction: float | None=None, stockout_fraction: float=0.3, hard_region_fraction: float=0.3, hard_region_width: float=0.06):
    """Sample interior phase states with extra mass near regime-change boundaries.

    The hard phase regions move with M through `entry_phi`. We therefore
    oversample around:
    - the start/end of the interior phase box (`phi ~= dphi`, `phi ~= 1-dphi`);
    - the moving active-set switch at `phi ~= entry_phi` when M is fractional;
    - stockout-neighbour states where a rival disappears from the demand system.
    """
    if device is None:
        device = torch.device('cpu')
    upper = 1.0 - float(dphi)
    if upper <= float(dphi):
        raise ValueError(f'dphi={dphi} leaves no interior HJB region')
    phi = float(dphi) + torch.rand(B, device=device) * (upper - float(dphi))
    if two_firm_fraction is not None and params.entry_phi > 0.0:
        n_two = max(0, min(B, int(round(float(two_firm_fraction) * B))))
        two_hi = min(float(params.entry_phi) - 1e-06, upper)
        if n_two > 0 and two_hi > float(dphi):
            phi[:n_two] = float(dphi) + torch.rand(n_two, device=device) * (two_hi - float(dphi))
        n_three = B - n_two
        three_lo = max(float(params.entry_phi) + 1e-06, float(dphi))
        if n_three > 0 and upper > three_lo:
            phi[n_two:] = three_lo + torch.rand(n_three, device=device) * (upper - three_lo)
    if hard_region_fraction and hard_region_fraction > 0.0:
        n_hard = max(0, min(B, int(round(float(hard_region_fraction) * B))))
        centers = [float(dphi), upper]
        if float(dphi) < float(params.entry_phi) < upper:
            centers.append(float(params.entry_phi))
        if n_hard > 0 and centers:
            counts = [n_hard // len(centers)] * len(centers)
            for j in range(n_hard % len(centers)):
                counts[j] += 1
            cursor = 0
            for count, center in zip(counts, centers):
                if count <= 0:
                    continue
                phi[cursor:cursor + count] = _sample_phi_band(count, center, float(hard_region_width), float(dphi), upper, device)
                cursor += count
    n = torch.randint(0, params.N + 1, (B, NSLOTS), device=device).float()
    if stockout_fraction and stockout_fraction > 0.0:
        k = max(0, min(B, int(round(float(stockout_fraction) * B))))
        if k > 0:
            ranks = torch.randint(0, NSLOTS, (k,), device=device)
            n[torch.arange(k, device=device), ranks] = 0.0
    inactive = ~active2_torch(phi, params)
    n[inactive, 2] = 0.0
    return (phi, n)

def boundary_residuals(net, B: int, params: FractionalParams):
    """Stockout, inactive-slot, departure, periodicity, and entry residuals."""
    device = _device(net)
    phi, n_base = sample_states(B, params, device=device)
    active = active_rank_mask(phi, n_base, params)
    a2 = active2_torch(phi, params).to(phi.dtype)
    stockout = []
    for r in range(NSLOTS):
        n_zero = n_base.clone()
        n_zero[:, r] = 0.0
        rr = torch.full((B,), float(r), device=device)
        _, V_zero = net(phi, rr, n_zero, a2, params.M, params.N)
        stockout.append(torch.where(active[:, r], V_zero, torch.zeros_like(V_zero)))
    r2 = torch.full((B,), 2.0, device=device)
    inactive_phi = torch.rand(B, device=device) * max(float(params.entry_phi), 1e-08)
    inactive_n = torch.randint(0, params.N + 1, (B, NSLOTS), device=device).float()
    inactive_n[:, 2] = 0.0
    a2_inactive = active2_torch(inactive_phi, params).to(inactive_phi.dtype)
    p_inactive, V_inactive = net(inactive_phi, r2, inactive_n, a2_inactive, params.M, params.N)
    inactive_slot = torch.stack([p_inactive, V_inactive], dim=-1)
    if params.entry_phi == 0.0:
        inactive_slot = torch.zeros_like(inactive_slot)
    n_boundary = torch.randint(0, params.N + 1, (B, NSLOTS), device=device).float()
    phi_end = torch.ones(B, device=device)
    a2_end = active2_torch(phi_end, params).to(phi_end.dtype)
    r0 = torch.zeros(B, device=device)
    _, V_depart0 = net(phi_end, r0, n_boundary, a2_end, params.M, params.N)
    departure0 = V_depart0
    n_post = _post_boundary_state(n_boundary, params)
    phi0 = torch.zeros(B, device=device)
    a2_0 = active2_torch(phi0, params).to(phi0.dtype)
    periodic = []
    for r in (1, 2):
        rr = torch.full((B,), float(r), device=device)
        _, V_end = net(phi_end, rr, n_boundary, a2_end, params.M, params.N)
        r_adv = torch.full((B,), float(r - 1), device=device)
        _, V_start = net(phi0, r_adv, n_post, a2_0, params.M, params.N)
        periodic.append(V_end - V_start)
    entry = []
    if params.entry_phi == 0.0:
        entry = [torch.zeros(B, device=device)]
    else:
        n01 = torch.randint(0, params.N + 1, (B, 2), device=device).float()
        n_before = torch.cat([n01, torch.zeros((B, 1), device=device)], dim=-1)
        n_after = n_before.clone()
        n_after[:, 2] = float(params.N)
        phi_entry = torch.full((B,), float(params.entry_phi), device=device)
        for r in (0, 1):
            rr = torch.full((B,), float(r), device=device)
            _, V_before = net(phi_entry, rr, n_before, torch.zeros_like(phi_entry), params.M, params.N)
            _, V_after = net(phi_entry, rr, n_after, torch.ones_like(phi_entry), params.M, params.N)
            entry.append(V_before - V_after)
    return {'stockout': stockout, 'inactive_slot': inactive_slot, 'departure0': departure0, 'periodic_value': periodic, 'entry_value': entry}

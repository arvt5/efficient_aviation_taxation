from __future__ import annotations
import numpy as np
import torch
from aviation_taxation.network import demand_torch
from aviation_taxation.policy_tables import _continuation_and_red, state_grid
from aviation_taxation.model import FractionalParams
from aviation_taxation.evaluation import _shift_down

def continuous_best_response_full(P: np.ndarray, V: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, *, device: torch.device, cross_weight: float=1.0, price_min: float=0.02, price_max: float=6.0, coarse_points: int=81, local_steps: int=16, threshold_candidates: bool=False, dtype: torch.dtype=torch.float32) -> tuple[np.ndarray, np.ndarray]:
    """Full-grid conditional BR prices and one-step payoff gains, with the
    cross-price/cross-inventory terms active when ``cross_weight != 0``."""
    Np = params.N + 1
    S = Np ** 3
    cw = float(cross_weight)
    states = state_grid(params.N, device=device, dtype=dtype)
    pgrid = torch.linspace(price_min, price_max, coarse_points, device=device, dtype=dtype)
    G = coarse_points
    invphi = 0.5 * (np.sqrt(5.0) - 1.0)
    best = P.copy()
    gain = np.zeros_like(P)
    lam = float(params.lambda_bar)
    margin_shift = float(params.tau - params.c - params.ticket_tax)
    with torch.no_grad():
        for i, phi in enumerate(phi_grid[:-1]):
            dt = float(params.d) * float(phi_grid[i + 1] - phi_grid[i])
            rivals = torch.as_tensor(P[i].reshape(S, 3), device=device, dtype=dtype)
            n = states
            ph = torch.full((S,), float(phi), device=device, dtype=dtype)
            for rank in range(3):
                if rank == 2 and phi < params.entry_phi:
                    continue
                cont_np, red_np = _continuation_and_red(V, params, phi_grid, i, rank)
                cont = torch.as_tensor(cont_np.reshape(S), device=device, dtype=dtype)
                red = torch.as_tensor(red_np.reshape(S), device=device, dtype=dtype)
                vshift = {}
                if cw != 0.0:
                    for j in range(3):
                        if j == rank:
                            continue
                        vshift[j] = torch.as_tensor(_shift_down(np.ascontiguousarray(cont_np), j, Np).reshape(S), device=device, dtype=dtype)

                def rhs_at(prices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    if prices.ndim == 1:
                        trial = rivals.clone()
                        trial[:, rank] = prices
                        demand = demand_torch(trial, ph, n, params)
                        sale = (lam * dt * demand[:, rank]).clamp(max=1.0)
                        num = cont + sale * (prices + margin_shift + red)
                        den = 1.0 + sale
                        if cw != 0.0:
                            for j, vsj in vshift.items():
                                sj = (lam * dt * demand[:, j]).clamp(max=1.0)
                                num = num + cw * sj * vsj
                                den = den + cw * sj
                        return (num / den, sale)
                    width = prices.shape[1]
                    trial = rivals[:, None, :].expand(-1, width, -1).clone()
                    trial[:, :, rank] = prices
                    demand = demand_torch(trial.reshape(S * width, 3), ph.repeat_interleave(width), n.repeat_interleave(width, dim=0), params).reshape(S, width, 3)
                    sale = (lam * dt * demand[:, :, rank]).clamp(max=1.0)
                    num = cont[..., None] + sale * (prices + margin_shift + red[..., None])
                    den = 1.0 + sale
                    if cw != 0.0:
                        for j, vsj in vshift.items():
                            sj = (lam * dt * demand[:, :, j]).clamp(max=1.0)
                            num = num + cw * sj * vsj[..., None]
                            den = den + cw * sj
                    return (num / den, sale)
                coarse_prices = pgrid[None, :].expand(S, -1)
                coarse_rhs, _ = rhs_at(coarse_prices)
                k = torch.argmax(coarse_rhs, dim=1)
                lo = pgrid[torch.clamp(k - 1, min=0)]
                hi = pgrid[torch.clamp(k + 1, max=G - 1)]
                x1 = hi - invphi * (hi - lo)
                x2 = lo + invphi * (hi - lo)
                f1, _ = rhs_at(x1)
                f2, _ = rhs_at(x2)
                for _ in range(local_steps):
                    right = f2 > f1
                    lo_new = torch.where(right, x1, lo)
                    hi_new = torch.where(right, hi, x2)
                    x1_new = hi_new - invphi * (hi_new - lo_new)
                    x2_new = lo_new + invphi * (hi_new - lo_new)
                    f1_new, _ = rhs_at(x1_new)
                    f2_new, _ = rhs_at(x2_new)
                    lo, hi, x1, x2, f1, f2 = (lo_new, hi_new, x1_new, x2_new, f1_new, f2_new)
                pbr = torch.where(f2 > f1, x2, x1)
                br_rhs, _ = rhs_at(pbr)
                current = rivals[:, rank]
                if threshold_candidates:
                    times = torch.tensor([(1.0 - float(phi)) * params.d, (2.0 - float(phi)) * params.d, (3.0 - float(phi)) * params.d], device=device, dtype=dtype)
                    candidates = []
                    candidate_masks = []
                    for rival_rank in range(3):
                        if rival_rank == rank:
                            continue
                        distance = torch.abs(times[rival_rank] - times[rank])
                        rival_active = states[:, rival_rank] > 0.5
                        if rival_rank == 2 and phi < params.entry_phi:
                            rival_active = torch.zeros_like(rival_active)
                        candidates.extend([rivals[:, rival_rank] - float(params.kappa) * distance, rivals[:, rival_rank] + float(params.kappa) * distance])
                        candidate_masks.extend([rival_active, rival_active])
                    gap = float(params.kappa) * (times[2] - times[0])
                    if rank == 0:
                        dominated = 2.0 * rivals[:, 1] - rivals[:, 2] - gap
                        mask = (states[:, 1] > 0.5) & (states[:, 2] > 0.5)
                        if phi < params.entry_phi:
                            mask = torch.zeros_like(mask)
                    elif rank == 1:
                        dominated = 0.5 * (rivals[:, 0] + rivals[:, 2] + gap)
                        mask = (states[:, 0] > 0.5) & (states[:, 2] > 0.5)
                        if phi < params.entry_phi:
                            mask = torch.zeros_like(mask)
                    else:
                        dominated = 2.0 * rivals[:, 1] - rivals[:, 0] - gap
                        mask = (states[:, 0] > 0.5) & (states[:, 1] > 0.5)
                    candidates.append(dominated)
                    candidate_masks.append(mask)
                    kink_prices = torch.stack(candidates, dim=1).clamp(min=float(price_min), max=float(price_max))
                    masks = torch.stack(candidate_masks, dim=1)
                    kink_prices = torch.where(masks, kink_prices, current[:, None])
                    kink_rhs, _ = rhs_at(kink_prices)
                    kink_idx = torch.argmax(kink_rhs, dim=1)
                    kink_price = kink_prices.gather(1, kink_idx[:, None]).squeeze(1)
                    kink_value = kink_rhs.gather(1, kink_idx[:, None]).squeeze(1)
                    take_kink = kink_value > br_rhs
                    pbr = torch.where(take_kink, kink_price, pbr)
                    br_rhs = torch.where(take_kink, kink_value, br_rhs)
                cur_rhs, _ = rhs_at(current)
                stocked = states[:, rank] > 0.5
                pbr = torch.where(stocked, pbr, current)
                br_gain = torch.where(stocked, torch.clamp(br_rhs - cur_rhs, min=0.0), torch.zeros_like(br_rhs))
                best[i, ..., rank] = pbr.cpu().numpy().reshape(Np, Np, Np)
                gain[i, ..., rank] = br_gain.cpu().numpy().reshape(Np, Np, Np)
    return (best, gain)

from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from aviation_taxation.network import demand_torch
from aviation_taxation.forward import _inject_entry, _sale_probs, phase_shift
from aviation_taxation.policy_tables import state_grid, write_json
from aviation_taxation.model import FractionalParams
FOCAL_RANKS = (2, 1, 0)

def _params_from_result(path: Path) -> FractionalParams:
    meta = json.loads(path.read_text())
    return FractionalParams(**meta['params'])

def _focal_entry_index(phi_grid: np.ndarray, params: FractionalParams) -> int:
    """First transition index at which the focal rank-2 flight is active."""
    if params.entry_phi == 0.0:
        return 0
    active = phi_grid[:-1] >= float(params.entry_phi)
    if not active.any():
        raise ValueError('phi_grid has no transition after rank-2 entry')
    return int(np.argmax(active))

def _life_start_index(life: int, phi_grid: np.ndarray, params: FractionalParams) -> int:
    return _focal_entry_index(phi_grid, params) if life == 0 else 0

def _entry_boundary_value(after_entry_value: np.ndarray, params: FractionalParams) -> np.ndarray:
    """Map a pre-entry padded state to the continuation after rank-2 entry."""
    N = params.N
    return np.broadcast_to(after_entry_value[:, :, N][:, :, None], (N + 1, N + 1, N + 1)).copy()

def _initial_focal_distribution(mu: np.ndarray, params: FractionalParams, phi_grid: np.ndarray) -> np.ndarray:
    """Focal flight enters as rank 2 with full inventory.

    Rivals are distributed as in the saved stationary distribution at the focal
    entry time, conditional on the newly-entered rank-2 slot being full.  For
    M=3 this is phase start; for fractional M it is the mid-phase entry index.
    """
    N = params.N
    init = np.zeros((N + 1, N + 1, N + 1), dtype=np.float64)
    entry_idx = _focal_entry_index(phi_grid, params)
    occ = mu[entry_idx]
    if params.entry_phi > 0.0 and float(occ[:, :, N].sum()) <= 0.0:
        occ = _inject_entry(occ)
    slice_full = occ[:, :, N]
    mass = float(slice_full.sum())
    if mass <= 0.0:
        raise ValueError('stationary distribution has no mass at focal entry n2=N')
    init[:, :, N] = slice_full / mass
    return init

def _landing_value(next_value: np.ndarray, rank: int) -> np.ndarray:
    """Value after a sale by ``rank`` from each source state."""
    Np = next_value.shape[0]
    out = next_value.copy()
    hi = [slice(None)] * 3
    lo = [slice(None)] * 3
    hi[rank] = slice(1, Np)
    lo[rank] = slice(0, Np - 1)
    out[tuple(hi)] = next_value[tuple(lo)]
    return out

def _expected_next(next_value: np.ndarray, sale_prob: np.ndarray) -> np.ndarray:
    stay = 1.0 - sale_prob.sum(axis=-1)
    out = stay * next_value
    for rank in range(3):
        out = out + sale_prob[..., rank] * _landing_value(next_value, rank)
    return out

def _transition_forward(mu: np.ndarray, sale_prob: np.ndarray) -> np.ndarray:
    stay = 1.0 - sale_prob.sum(axis=-1)
    sales = mu[..., None] * sale_prob
    nxt = mu * stay
    nxt[:-1, :, :] += sales[1:, :, :, 0]
    nxt[:, :-1, :] += sales[:, 1:, :, 1]
    nxt[:, :, :-1] += sales[:, :, 1:, 2]
    return nxt

def _phase_boundary_value(next_phase_start: np.ndarray, params: FractionalParams) -> np.ndarray:
    """Map old terminal state to next phase-start continuation value."""
    N = params.N
    fresh = N if params.entry_phi == 0.0 else 0
    mapped = next_phase_start[:, :, fresh]
    return np.broadcast_to(mapped[None, :, :], (N + 1, N + 1, N + 1)).copy()

def _terminal_unsold_value(params: FractionalParams) -> np.ndarray:
    Np = params.N + 1
    return np.broadcast_to(np.arange(Np, dtype=np.float64).reshape(Np, 1, 1), (Np, Np, Np)).copy()

def focal_forward_baseline(P: np.ndarray, D: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, initial_mu: np.ndarray) -> tuple[np.ndarray, dict]:
    """Transient focal lifecycle distribution and baseline focal accounting."""
    nsteps = len(phi_grid) - 1
    entry_idx = _focal_entry_index(phi_grid, params)
    alpha = np.zeros((3, nsteps, params.N + 1, params.N + 1, params.N + 1), dtype=np.float64)
    mu = initial_mu.copy()
    revenue = 0.0
    sales = 0.0
    for life, rank in enumerate(FOCAL_RANKS):
        start_k = _life_start_index(life, phi_grid, params)
        for k in range(start_k, nsteps):
            if params.entry_phi > 0.0 and life > 0 and (k == entry_idx):
                mu = _inject_entry(mu)
            alpha[life, k] = mu
            dt = float(params.d) * float(phi_grid[k + 1] - phi_grid[k])
            sp = _sale_probs(D[k], float(params.lambda_bar), dt)
            q_focal = mu * sp[..., rank]
            sales += float(q_focal.sum())
            revenue += float((q_focal * P[k, ..., rank]).sum())
            mu = _transition_forward(mu, sp)
        if life < 2:
            mu = phase_shift(mu, params)
    unsold = float((mu * _terminal_unsold_value(params)).sum())
    profit = revenue - (float(params.c) + float(params.ticket_tax)) * sales - float(params.tau) * unsold
    moments = {'gross_revenue': revenue, 'sales': sales, 'probability_weighted_price': revenue / sales if sales > 0 else float('nan'), 'unsold': unsold, 'ticket_tax_payment': float(params.ticket_tax) * sales, 'vacancy_tax_payment': float(params.tau) * unsold, 'firm_operating_profit_before_fixed': profit}
    return (alpha, moments)

def focal_accounting_continuations(P: np.ndarray, D: np.ndarray, params: FractionalParams, phi_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Expected future focal sales/revenue/unsold from every lifecycle state."""
    nsteps = len(phi_grid) - 1
    entry_idx = _focal_entry_index(phi_grid, params)
    shape = (3, nsteps, params.N + 1, params.N + 1, params.N + 1)
    cont_sales = np.zeros(shape, dtype=np.float64)
    cont_revenue = np.zeros(shape, dtype=np.float64)
    cont_unsold = np.zeros(shape, dtype=np.float64)
    terminal_unsold = _terminal_unsold_value(params)
    for life in reversed(range(3)):
        rank = FOCAL_RANKS[life]
        start_k = _life_start_index(life, phi_grid, params)
        next_sales = np.zeros_like(terminal_unsold)
        next_revenue = np.zeros_like(terminal_unsold)
        next_unsold = terminal_unsold.copy() if life == 2 else _phase_boundary_value(cont_unsold[life + 1, 0], params)
        if life < 2:
            next_sales = _phase_boundary_value(cont_sales[life + 1, 0], params)
            next_revenue = _phase_boundary_value(cont_revenue[life + 1, 0], params)
        for k in reversed(range(start_k, nsteps)):
            dt = float(params.d) * float(phi_grid[k + 1] - phi_grid[k])
            sp = _sale_probs(D[k], float(params.lambda_bar), dt)
            immediate_sales = sp[..., rank]
            immediate_revenue = immediate_sales * P[k, ..., rank]
            cont_sales[life, k] = immediate_sales + _expected_next(next_sales, sp)
            cont_revenue[life, k] = immediate_revenue + _expected_next(next_revenue, sp)
            cont_unsold[life, k] = _expected_next(next_unsold, sp)
            next_sales = cont_sales[life, k]
            next_revenue = cont_revenue[life, k]
            next_unsold = cont_unsold[life, k]
            if params.entry_phi > 0.0 and life > 0 and (k == entry_idx):
                next_sales = _entry_boundary_value(next_sales, params)
                next_revenue = _entry_boundary_value(next_revenue, params)
                next_unsold = _entry_boundary_value(next_unsold, params)
    return {'sales': cont_sales, 'revenue': cont_revenue, 'unsold': cont_unsold}

def _candidate_demand_tables(P: np.ndarray, Pbr: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, *, life: int, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    """Demand at every state when that state's focal price is replaced by p_BR."""
    rank = FOCAL_RANKS[life]
    Np = params.N + 1
    S = Np ** 3
    states = state_grid(params.N, device=device, dtype=dtype)
    out = np.zeros((len(phi_grid) - 1, Np, Np, Np, 3), dtype=np.float64)
    with torch.no_grad():
        for k, phi in enumerate(phi_grid[:-1]):
            trial = P[k].reshape(S, 3).copy()
            trial[:, rank] = Pbr[k, ..., rank].reshape(S)
            prices = torch.as_tensor(trial, device=device, dtype=dtype)
            ph = torch.full((S,), float(phi), device=device, dtype=dtype)
            d = demand_torch(prices, ph, states, params)
            out[k] = d.cpu().numpy().reshape(Np, Np, Np, 3)
    return out

def _weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict:
    positive = weights > 0
    if not positive.any():
        return {'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'p99': 0.0, 'p999': 0.0, 'max': 0.0}
    v = values[positive]
    w = weights[positive]
    order = np.argsort(v)
    vv = v[order]
    ww = w[order]
    cdf = np.cumsum(ww) / ww.sum()

    def q(prob: float) -> float:
        return float(vv[min(np.searchsorted(cdf, prob, side='left'), len(vv) - 1)])
    return {'mean': float((v * w).sum() / w.sum()), 'p50': q(0.5), 'p90': q(0.9), 'p99': q(0.99), 'p999': q(0.999), 'max': float(v.max())}

def evaluate_focal_one_shot_gains(P: np.ndarray, D: np.ndarray, Pbr: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, initial_mu: np.ndarray, *, device_name: str='cpu', dtype: torch.dtype=torch.float64, top_k: int=200) -> tuple[dict, list[dict], np.ndarray]:
    alpha, baseline = focal_forward_baseline(P, D, params, phi_grid, initial_mu)
    cont = focal_accounting_continuations(P, D, params, phi_grid)
    nsteps = len(phi_grid) - 1
    Np = params.N + 1
    device = torch.device(device_name)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    grids = np.indices((Np, Np, Np), sparse=False)
    gain = np.zeros((3, nsteps, Np, Np, Np), dtype=np.float64)
    delta_sales = np.zeros_like(gain)
    delta_revenue = np.zeros_like(gain)
    delta_unsold = np.zeros_like(gain)
    top_rows: list[dict] = []
    for life, rank in enumerate(FOCAL_RANKS):
        Ddev = _candidate_demand_tables(P, Pbr, params, phi_grid, life=life, device=device, dtype=dtype)
        for k in range(nsteps):
            dt = float(params.d) * float(phi_grid[k + 1] - phi_grid[k])
            sp_dev = _sale_probs(Ddev[k], float(params.lambda_bar), dt)
            next_sales, next_revenue, next_unsold = _next_continuation_arrays(cont, params, phi_grid, life=life, k=k, nsteps=nsteps)
            dev_sales_total = sp_dev[..., rank] + _expected_next(next_sales, sp_dev)
            dev_revenue_total = sp_dev[..., rank] * Pbr[k, ..., rank] + _expected_next(next_revenue, sp_dev)
            dev_unsold_total = _expected_next(next_unsold, sp_dev)
            ds = alpha[life, k] * (dev_sales_total - cont['sales'][life, k])
            dr = alpha[life, k] * (dev_revenue_total - cont['revenue'][life, k])
            du = alpha[life, k] * (dev_unsold_total - cont['unsold'][life, k])
            dp = dr - (float(params.c) + float(params.ticket_tax)) * ds - float(params.tau) * du
            active = grids[rank] > 0
            gain[life, k] = np.where(active, dp, 0.0)
            delta_sales[life, k] = np.where(active, ds, 0.0)
            delta_revenue[life, k] = np.where(active, dr, 0.0)
            delta_unsold[life, k] = np.where(active, du, 0.0)
            if top_k > 0:
                flat = np.argpartition(gain[life, k].reshape(-1), -min(top_k, gain[life, k].size))[-min(top_k, gain[life, k].size):]
                for idx in flat:
                    n0, n1, n2 = np.unravel_index(int(idx), (Np, Np, Np))
                    if not active[n0, n1, n2]:
                        continue
                    g = float(gain[life, k, n0, n1, n2])
                    if g <= 0.0:
                        continue
                    top_rows.append({'profit_gain_raw': g, 'profit_gain_scaled': g * float(nsteps), 'life_phase': int(life), 'focal_rank': int(rank), 'phi_index': int(k), 'phi': float(phi_grid[k]), 'n0': int(n0), 'n1': int(n1), 'n2': int(n2), 'reach_probability': float(alpha[life, k, n0, n1, n2]), 'current_price': float(P[k, n0, n1, n2, rank]), 'br_price': float(Pbr[k, n0, n1, n2, rank]), 'abs_price_deviation': float(abs(Pbr[k, n0, n1, n2, rank] - P[k, n0, n1, n2, rank])), 'delta_sales': float(delta_sales[life, k, n0, n1, n2]), 'delta_revenue': float(delta_revenue[life, k, n0, n1, n2]), 'delta_unsold': float(delta_unsold[life, k, n0, n1, n2])})
                top_rows.sort(key=lambda row: row['profit_gain_raw'], reverse=True)
                del top_rows[top_k:]
    positive_gain = np.maximum(gain, 0.0)
    weights = alpha
    baseline_profit = baseline['firm_operating_profit_before_fixed']
    scale = int(nsteps)
    scaled_positive_gain = scale * positive_gain
    summary = {'params': asdict(params), 'baseline_focal': baseline, 'initial_distribution_mass': float(initial_mu.sum()), 'grid_scaling': {'nsteps': scale, 'dphi': float(phi_grid[1] - phi_grid[0]), 'rule': 'scaled_profit_gain = raw_profit_gain * nsteps', 'interpretation': 'raw_profit_gain is the finite-cell unconditional full-horizon gain; scaled_profit_gain divides by dphi and is the primary grid-invariant one-shot metric'}, 'candidate_count': int(np.broadcast_to(grids[2] > 0, (nsteps, Np, Np, Np)).sum() + np.broadcast_to(grids[1] > 0, (nsteps, Np, Np, Np)).sum() + np.broadcast_to(grids[0] > 0, (nsteps, Np, Np, Np)).sum()), 'positive_gain_count': int((positive_gain > 0.0).sum()), 'max_profit_gain_raw': float(positive_gain.max()), 'max_profit_gain_scaled': float(scaled_positive_gain.max()), 'max_profit_gain_raw_relative_to_baseline': float(positive_gain.max() / baseline_profit) if baseline_profit != 0 else None, 'max_profit_gain_scaled_relative_to_baseline': float(scaled_positive_gain.max() / baseline_profit) if baseline_profit != 0 else None, 'profit_gain_raw_weighted_by_reach_probability': _weighted_summary(positive_gain, weights), 'profit_gain_scaled_weighted_by_reach_probability': _weighted_summary(scaled_positive_gain, weights)}
    for life, rank in enumerate(FOCAL_RANKS):
        summary[f'rank{rank}'] = {'max_profit_gain_raw': float(positive_gain[life].max()), 'max_profit_gain_scaled': float(scaled_positive_gain[life].max()), 'raw_weighted_by_reach_probability': _weighted_summary(positive_gain[life], weights[life]), 'scaled_weighted_by_reach_probability': _weighted_summary(scaled_positive_gain[life], weights[life])}
    return (summary, top_rows, alpha)

def _scalar_landing(arr: np.ndarray, n0: int, n1: int, n2: int, rank: int) -> float:
    if rank == 0 and n0 > 0:
        return float(arr[n0 - 1, n1, n2])
    if rank == 1 and n1 > 0:
        return float(arr[n0, n1 - 1, n2])
    if rank == 2 and n2 > 0:
        return float(arr[n0, n1, n2 - 1])
    return float(arr[n0, n1, n2])

def _next_continuation_arrays(cont: dict[str, np.ndarray], params: FractionalParams, phi_grid: np.ndarray, *, life: int, k: int, nsteps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Np = params.N + 1
    entry_idx = _focal_entry_index(phi_grid, params)
    if k == nsteps - 1:
        if life == 2:
            next_sales = np.zeros((Np, Np, Np), dtype=np.float64)
            next_revenue = np.zeros_like(next_sales)
            next_unsold = _terminal_unsold_value(params)
        else:
            next_sales = _phase_boundary_value(cont['sales'][life + 1, 0], params)
            next_revenue = _phase_boundary_value(cont['revenue'][life + 1, 0], params)
            next_unsold = _phase_boundary_value(cont['unsold'][life + 1, 0], params)
    else:
        next_sales = cont['sales'][life, k + 1]
        next_revenue = cont['revenue'][life, k + 1]
        next_unsold = cont['unsold'][life, k + 1]
        if params.entry_phi > 0.0 and life > 0 and (k + 1 == entry_idx):
            next_sales = _entry_boundary_value(next_sales, params)
            next_revenue = _entry_boundary_value(next_revenue, params)
            next_unsold = _entry_boundary_value(next_unsold, params)
    return (next_sales, next_revenue, next_unsold)

def _forward_accounting_gain_for_prices(P: np.ndarray, cont: dict[str, np.ndarray], alpha: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, *, life: int, k: int, n0: int, n1: int, n2: int, prices: np.ndarray) -> np.ndarray:
    rank = FOCAL_RANKS[life]
    nsteps = len(phi_grid) - 1
    next_sales, next_revenue, next_unsold = _next_continuation_arrays(cont, params, phi_grid, life=life, k=k, nsteps=nsteps)
    base_profit = cont['revenue'][life, k, n0, n1, n2] - (float(params.c) + float(params.ticket_tax)) * cont['sales'][life, k, n0, n1, n2] - float(params.tau) * cont['unsold'][life, k, n0, n1, n2]
    trial = np.broadcast_to(P[k, n0, n1, n2], (len(prices), 3)).copy()
    trial[:, rank] = prices
    state = torch.tensor([[n0, n1, n2]], dtype=torch.float64).expand(len(prices), -1)
    price_t = torch.tensor(trial, dtype=torch.float64)
    phi_t = torch.full((len(prices),), float(phi_grid[k]), dtype=torch.float64)
    with torch.no_grad():
        demand = demand_torch(price_t, phi_t, state, params).numpy()
    dt = float(params.d) * float(phi_grid[k + 1] - phi_grid[k])
    sp = _sale_probs(demand, float(params.lambda_bar), dt)
    stay = 1.0 - sp.sum(axis=1)
    exp_sales = stay * float(next_sales[n0, n1, n2])
    exp_revenue = stay * float(next_revenue[n0, n1, n2])
    exp_unsold = stay * float(next_unsold[n0, n1, n2])
    for j in range(3):
        exp_sales += sp[:, j] * _scalar_landing(next_sales, n0, n1, n2, j)
        exp_revenue += sp[:, j] * _scalar_landing(next_revenue, n0, n1, n2, j)
        exp_unsold += sp[:, j] * _scalar_landing(next_unsold, n0, n1, n2, j)
    total_sales = sp[:, rank] + exp_sales
    total_revenue = sp[:, rank] * prices + exp_revenue
    total_profit = total_revenue - (float(params.c) + float(params.ticket_tax)) * total_sales - float(params.tau) * exp_unsold
    return float(alpha[life, k, n0, n1, n2]) * (total_profit - base_profit)

def optimize_forward_accounting_price(P: np.ndarray, cont: dict[str, np.ndarray], alpha: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, row: dict, *, price_min: float, price_max: float, coarse_points: int, local_steps: int) -> dict:
    life = int(row['life_phase'])
    k = int(row['phi_index'])
    n0 = int(row['n0'])
    n1 = int(row['n1'])
    n2 = int(row['n2'])
    grid = np.linspace(float(price_min), float(price_max), int(coarse_points), dtype=np.float64)
    gains = _forward_accounting_gain_for_prices(P, cont, alpha, params, phi_grid, life=life, k=k, n0=n0, n1=n1, n2=n2, prices=grid)
    best_idx = int(np.argmax(gains))
    lo = grid[max(best_idx - 1, 0)]
    hi = grid[min(best_idx + 1, len(grid) - 1)]
    invphi = 0.5 * (np.sqrt(5.0) - 1.0)
    x1 = hi - invphi * (hi - lo)
    x2 = lo + invphi * (hi - lo)
    f1 = float(_forward_accounting_gain_for_prices(P, cont, alpha, params, phi_grid, life=life, k=k, n0=n0, n1=n1, n2=n2, prices=np.array([x1]))[0])
    f2 = float(_forward_accounting_gain_for_prices(P, cont, alpha, params, phi_grid, life=life, k=k, n0=n0, n1=n1, n2=n2, prices=np.array([x2]))[0])
    for _ in range(int(local_steps)):
        if f2 > f1:
            lo = x1
        else:
            hi = x2
        x1 = hi - invphi * (hi - lo)
        x2 = lo + invphi * (hi - lo)
        f1 = float(_forward_accounting_gain_for_prices(P, cont, alpha, params, phi_grid, life=life, k=k, n0=n0, n1=n1, n2=n2, prices=np.array([x1]))[0])
        f2 = float(_forward_accounting_gain_for_prices(P, cont, alpha, params, phi_grid, life=life, k=k, n0=n0, n1=n1, n2=n2, prices=np.array([x2]))[0])
    candidate_prices = np.array([x1, x2, grid[best_idx], float(row['br_price']), float(row['current_price'])], dtype=np.float64)
    candidate_gains = _forward_accounting_gain_for_prices(P, cont, alpha, params, phi_grid, life=life, k=k, n0=n0, n1=n1, n2=n2, prices=candidate_prices)
    best = int(np.argmax(candidate_gains))
    p_opt = float(candidate_prices[best])
    gain_opt = float(candidate_gains[best])
    return {'direct_opt_price': float(p_opt), 'direct_profit_gain_raw': float(gain_opt), 'direct_profit_gain_scaled': float(gain_opt * (len(phi_grid) - 1)), 'direct_gain_improvement_raw': float(gain_opt - row['profit_gain_raw']), 'direct_gain_improvement_scaled': float((gain_opt - row['profit_gain_raw']) * (len(phi_grid) - 1)), 'direct_abs_price_vs_saved_br': float(abs(p_opt - row['br_price'])), 'direct_abs_price_vs_current': float(abs(p_opt - row['current_price']))}

def brute_force_deviation_profit(P: np.ndarray, D: np.ndarray, Pbr: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, initial_mu: np.ndarray, *, life: int, k_dev: int, n0: int, n1: int, n2: int) -> dict:
    """Literal forward-pass check for one focal one-price replacement."""
    nsteps = len(phi_grid) - 1
    entry_idx = _focal_entry_index(phi_grid, params)
    rank_dev = FOCAL_RANKS[life]
    mu = initial_mu.copy()
    revenue = 0.0
    sales = 0.0
    for life_i, rank in enumerate(FOCAL_RANKS):
        start_k = _life_start_index(life_i, phi_grid, params)
        for k in range(start_k, nsteps):
            if params.entry_phi > 0.0 and life_i > 0 and (k == entry_idx):
                mu = _inject_entry(mu)
            Dk = D[k]
            P_focal = P[k, ..., rank]
            if life_i == life and k == k_dev:
                Dk = Dk.copy()
                trial = P[k, n0, n1, n2].copy()
                trial[rank_dev] = Pbr[k, n0, n1, n2, rank_dev]
                state = torch.tensor([[n0, n1, n2]], dtype=torch.float64)
                price = torch.tensor(trial[None, :], dtype=torch.float64)
                phi = torch.tensor([float(phi_grid[k])], dtype=torch.float64)
                Dk[n0, n1, n2] = demand_torch(price, phi, state, params).numpy()[0]
                P_focal = P_focal.copy()
                P_focal[n0, n1, n2] = Pbr[k, n0, n1, n2, rank_dev]
            dt = float(params.d) * float(phi_grid[k + 1] - phi_grid[k])
            sp = _sale_probs(Dk, float(params.lambda_bar), dt)
            q_focal = mu * sp[..., rank]
            sales += float(q_focal.sum())
            revenue += float((q_focal * P_focal).sum())
            mu = _transition_forward(mu, sp)
        if life_i < 2:
            mu = phase_shift(mu, params)
    unsold = float((mu * _terminal_unsold_value(params)).sum())
    profit = revenue - (float(params.c) + float(params.ticket_tax)) * sales - float(params.tau) * unsold
    return {'gross_revenue': revenue, 'sales': sales, 'probability_weighted_price': revenue / sales if sales > 0 else float('nan'), 'unsold': unsold, 'firm_operating_profit_before_fixed': profit}

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--result-json', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--verify-top', type=int, default=5)
    ap.add_argument('--optimize-top', type=int, default=0, help='directly maximize the forward-accounting objective for the top saved-BR candidates')
    ap.add_argument('--optimize-price-min', type=float, default=0.02)
    ap.add_argument('--optimize-price-max', type=float, default=6.0)
    ap.add_argument('--optimize-coarse-points', type=int, default=81)
    ap.add_argument('--optimize-local-steps', type=int, default=24)
    ap.add_argument('--save-arrays', action='store_true')
    args = ap.parse_args()
    z = np.load(args.policy, allow_pickle=True)
    params = _params_from_result(args.result_json)
    phi_grid = z['phi_grid']
    P = z['prices']
    D = z['demand']
    br_key = 'best_response_prices' if 'best_response_prices' in z.files else 'best_response_price'
    Pbr = z[br_key]
    initial_mu = _initial_focal_distribution(z['occupancy'], params, phi_grid)
    summary, top_rows, alpha = evaluate_focal_one_shot_gains(P, D, Pbr, params, phi_grid, initial_mu, device_name=args.device, top_k=args.top_k)
    if args.optimize_top > 0:
        cont = focal_accounting_continuations(P, D, params, phi_grid)
        opt_count = min(int(args.optimize_top), len(top_rows))
        optimized = []
        for row in top_rows[:opt_count]:
            direct = optimize_forward_accounting_price(P, cont, alpha, params, phi_grid, row, price_min=args.optimize_price_min, price_max=args.optimize_price_max, coarse_points=args.optimize_coarse_points, local_steps=args.optimize_local_steps)
            row.update(direct)
            optimized.append(row)
        if optimized:
            best_direct = max(optimized, key=lambda r: r['direct_profit_gain_raw'])
            best_improvement = max(optimized, key=lambda r: r['direct_gain_improvement_raw'])
            summary['direct_forward_optimization_top'] = {'candidate_count': opt_count, 'price_min': float(args.optimize_price_min), 'price_max': float(args.optimize_price_max), 'coarse_points': int(args.optimize_coarse_points), 'local_steps': int(args.optimize_local_steps), 'max_direct_profit_gain_raw': float(best_direct['direct_profit_gain_raw']), 'max_direct_profit_gain_scaled': float(best_direct['direct_profit_gain_scaled']), 'max_direct_gain_raw_relative_to_baseline': float(best_direct['direct_profit_gain_raw']) / summary['baseline_focal']['firm_operating_profit_before_fixed'], 'max_direct_gain_scaled_relative_to_baseline': float(best_direct['direct_profit_gain_scaled']) / summary['baseline_focal']['firm_operating_profit_before_fixed'], 'max_direct_gain_relative_to_baseline': float(best_direct['direct_profit_gain_scaled']) / summary['baseline_focal']['firm_operating_profit_before_fixed'], 'max_improvement_over_saved_br_raw': float(best_improvement['direct_gain_improvement_raw']), 'max_improvement_over_saved_br_scaled': float(best_improvement['direct_gain_improvement_scaled']), 'max_abs_price_vs_saved_br': float(max((r['direct_abs_price_vs_saved_br'] for r in optimized)))}
    if args.verify_top > 0:
        baseline = summary['baseline_focal']
        verifications = []
        for row in top_rows[:args.verify_top]:
            brute = brute_force_deviation_profit(P, D, Pbr, params, phi_grid, initial_mu, life=int(row['life_phase']), k_dev=int(row['phi_index']), n0=int(row['n0']), n1=int(row['n1']), n2=int(row['n2']))
            brute_gain = brute['firm_operating_profit_before_fixed'] - baseline['firm_operating_profit_before_fixed']
            row['brute_force_profit_gain_raw'] = brute_gain
            row['brute_force_profit_gain_scaled'] = brute_gain * summary['grid_scaling']['nsteps']
            row['brute_force_abs_error_raw'] = abs(brute_gain - row['profit_gain_raw'])
            verifications.append({**{k: row[k] for k in ('life_phase', 'focal_rank', 'phi_index', 'n0', 'n1', 'n2')}, 'fast_gain_raw': row['profit_gain_raw'], 'fast_gain_scaled': row['profit_gain_scaled'], 'brute_force_gain_raw': brute_gain, 'brute_force_gain_scaled': brute_gain * summary['grid_scaling']['nsteps'], 'abs_error_raw': abs(brute_gain - row['profit_gain_raw'])})
        summary['brute_force_verification'] = verifications
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / 'summary.json', summary)
    _write_csv(args.out_dir / 'top_deviations.csv', top_rows)
    if args.save_arrays:
        np.savez_compressed(args.out_dir / 'focal_forward_arrays.npz', alpha=alpha)
    print(json.dumps({'baseline_profit': summary['baseline_focal']['firm_operating_profit_before_fixed'], 'baseline_sales': summary['baseline_focal']['sales'], 'baseline_Ep': summary['baseline_focal']['probability_weighted_price'], 'max_profit_gain_raw': summary['max_profit_gain_raw'], 'max_profit_gain_scaled': summary['max_profit_gain_scaled'], 'relative_max_gain_raw': summary['max_profit_gain_raw_relative_to_baseline'], 'relative_max_gain_scaled': summary['max_profit_gain_scaled_relative_to_baseline'], 'positive_gain_count': summary['positive_gain_count'], 'direct_forward_optimization_top': summary.get('direct_forward_optimization_top'), 'out_dir': str(args.out_dir)}, indent=2))
if __name__ == '__main__':
    main()

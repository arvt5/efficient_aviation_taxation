from __future__ import annotations
import numpy as np
import torch
from aviation_taxation.network import demand_torch, rank_values
from aviation_taxation.model import FractionalParams

def price_demand_tables(net, params: FractionalParams, phi_grid: np.ndarray):
    """Return P[K,Np,Np,Np,3] prices and D[K,Np,Np,Np,3] demand from the net."""
    N = params.N
    Np = N + 1
    K = len(phi_grid)
    states = np.array([(a, b, c) for a in range(Np) for b in range(Np) for c in range(Np)], dtype=np.float64)
    S = len(states)
    dtype = next(net.parameters()).dtype
    states_t = torch.tensor(states, dtype=dtype)
    P = np.zeros((K, Np, Np, Np, 3))
    D = np.zeros((K, Np, Np, Np, 3))
    with torch.no_grad():
        for i, phi in enumerate(phi_grid):
            phi_t = torch.full((S,), float(phi), dtype=dtype)
            p_i, _ = rank_values(net, phi_t, states_t, params)
            d_i = demand_torch(p_i, phi_t, states_t, params)
            p_np = p_i.numpy().reshape(Np, Np, Np, 3)
            d_np = d_i.numpy().reshape(Np, Np, Np, 3)
            P[i] = p_np
            D[i] = d_np
    return (P, D)

def phase_shift(mu_term: np.ndarray, params: FractionalParams) -> np.ndarray:
    """Apply the period-boundary map to the terminal occupancy."""
    N = params.N
    Np = N + 1
    out = np.zeros((Np, Np, Np))
    fresh = N if params.entry_phi == 0.0 else 0
    for n0 in range(Np):
        for n1 in range(Np):
            for n2 in range(Np):
                m = mu_term[n0, n1, n2]
                if m > 0.0:
                    out[n1, n2, fresh] += m
    return out

def _entry_index(phi_grid: np.ndarray, params: FractionalParams) -> int:
    """First phase index at which rank 2 is active (the mid-phase entry point).

    Returns -1 for the synchronized M=3 limit (entry_phi=0): no mid-phase entry,
    the fresh firm enters at the period boundary via `phase_shift`.
    """
    if params.entry_phi == 0.0:
        return -1
    active = phi_grid >= params.entry_phi
    return int(np.argmax(active)) if active.any() else -1

def _inject_entry(mu_i: np.ndarray) -> np.ndarray:
    """Fractional M: a fresh rank-2 firm enters at full N. All surviving mass
    (which has rank 2 padded at 0 before entry) is moved to n2 = N."""
    out = np.zeros_like(mu_i)
    out[:, :, -1] = mu_i.sum(axis=2)
    return out

def _sale_probs(D_i, lam, dt):
    """Per-rank sale probabilities, rescaled so the total never exceeds 1
    (guards mass conservation at large dt / lambda; matches Stage 4 occupancy)."""
    sp = np.minimum(lam * dt * D_i, 1.0)
    ssum = sp.sum(axis=-1)
    over = ssum > 1.0
    if np.ndim(ssum) == 0:
        if over:
            sp = sp / ssum
    else:
        scale = np.where(over, 1.0 / np.maximum(ssum, 1e-300), 1.0)
        sp = sp * scale[..., None]
    return sp

def propagate_one_phase(P, D, params: FractionalParams, phi_grid: np.ndarray, initial_mu: np.ndarray):
    """Vectorized occupancy propagation across one phase; return (mu, q).

    Equivalent to `_propagate_one_phase_loop`: demand is 0 where a firm is
    stocked out, so a sale is a pure shift-down along that rank's inventory axis;
    sale probabilities are rescaled when their sum exceeds 1; and for fractional
    M a fresh rank-2 firm is injected at full N at the entry phase index.
    """
    N = params.N
    Np = N + 1
    d = float(params.d)
    lam = float(params.lambda_bar)
    K = len(phi_grid)
    nsteps = K - 1
    entry_idx = _entry_index(phi_grid, params)
    mu = np.zeros((K, Np, Np, Np))
    q = np.zeros((nsteps, Np, Np, Np, 3))
    mu[0] = initial_mu
    for i in range(nsteps):
        if i == entry_idx:
            mu[i] = _inject_entry(mu[i])
        dt = d * float(phi_grid[i + 1] - phi_grid[i])
        sp = _sale_probs(D[i], lam, dt)
        stay = 1.0 - sp.sum(axis=-1)
        mu_i = mu[i]
        sales = mu_i[..., None] * sp
        q[i] = sales
        nxt = mu_i * stay
        nxt[:-1, :, :] += sales[1:, :, :, 0]
        nxt[:, :-1, :] += sales[:, 1:, :, 1]
        nxt[:, :, :-1] += sales[:, :, 1:, 2]
        mu[i + 1] = nxt
    if nsteps == entry_idx:
        mu[nsteps] = _inject_entry(mu[nsteps])
    return (mu, q)

def stationary_forward(net, params: FractionalParams, phi_grid: np.ndarray, burn_in: int=20):
    """Iterate the phase cycle to the stationary occupancy; return (mu, q, P, D)."""
    N = params.N
    Np = N + 1
    P, D = price_demand_tables(net, params, phi_grid)
    mu0 = np.zeros((Np, Np, Np))
    if params.entry_phi == 0.0:
        mu0[N, N, N] = 1.0
    else:
        mu0[N, N, 0] = 1.0
    mu = q = None
    for _ in range(max(burn_in, 0)):
        mu, q = propagate_one_phase(P, D, params, phi_grid, mu0)
        mu0 = phase_shift(mu[-1], params)
    mu, q = propagate_one_phase(P, D, params, phi_grid, mu0)
    return (mu, q, P, D)

def forward_moments(mu, q, P, params: FractionalParams) -> dict:
    """Calibration moments from a stationary forward result."""
    N = params.N
    Np = N + 1
    terminal = mu[-1]
    n0_grid = np.arange(Np).reshape(Np, 1, 1)
    n1_grid = np.arange(Np).reshape(1, Np, 1)
    nsteps = q.shape[0]
    sale_mass = float(q.sum())
    pw_price = float((q * P[:nsteps]).sum() / sale_mass) if sale_mass > 0 else float('nan')
    relcost = float(params.c / pw_price) if pw_price > 0 else float('nan')
    per_rank = {}
    for r in range(3):
        mr = float(q[..., r].sum())
        per_rank[f'pw_price_rank{r}'] = float((q[..., r] * P[:nsteps, ..., r]).sum() / mr) if mr > 0 else float('nan')
        per_rank[f'sale_mass_rank{r}'] = mr
    return {'vacancy_departing': float((terminal * n0_grid).sum() / N), 'vacancy_survivor_rank1_pre_shift': float((terminal * n1_grid).sum() / N), 'probability_weighted_price': pw_price, 'relcost': relcost, 'sale_mass_total': sale_mass, 'mu_mass_start': float(mu[0].sum()), 'mu_mass_terminal': float(terminal.sum()), **per_rank}

def welfare_moments(mu, q, P, params: FractionalParams, disp_per_buyer=None) -> dict:
    """Ex-post welfare / profit accounting over the stationary occupancy.

    Uses the REALIZED sale mass `q` from the occupancy law (so the accounts match
    the simulated process exactly, even if the `_sale_probs` mass-conservation
    guard binds), rather than re-deriving sales from D. Per (phase, state, rank):
        revenue = q * p,  producer_surplus = q * (p - c),  consumer_surplus = q / a.

    The vacancy tax falls on seats unsold AT DEPARTURE, i.e. terminal rank-0
    inventory = N * vacancy_departing, taken directly from the occupancy law (not
    the residual N - sales, which equals it only in exact stationarity). The gap
    between the two is reported as a stationarity diagnostic.

    Welfare = producer_surplus + consumer_surplus is the COMPLETE social surplus
    and already accounts for the schedule-delay disutility: a buyer with value
    Z~Exp(a) buys iff Z >= q_j = p_j + kappa*|tp - t_j|, so per transaction the
    social value E[Z|buy] - c - kappa*disp = (q_j + 1/a) - c - kappa*disp =
    p_j - c + 1/a -- the displacement cancels (it acts only through the extensive
    margin, lowering the purchase probability e^{-a q_j} and hence sales). The
    displacement therefore must NOT be subtracted again.

    If `disp_per_buyer` (Gd_r/D_r per phase/state/rank, see
    `displacement_per_buyer_table`) is supplied, the average realized schedule
    delay `displacement_cost = sum_q q * disp_per_buyer` is reported as a
    DESCRIPTIVE diagnostic only (it does not enter welfare). Tax is a transfer:
    welfare excludes it. Calendar rates divide by d.
    """
    N = params.N
    a = float(params.a)
    c = float(params.c)
    d = float(params.d)
    tau = float(params.tau)
    ticket_tax = float(params.ticket_tax)
    nsteps = q.shape[0]
    sales_total = float(q.sum())
    revenue = float((q * P[:nsteps]).sum())
    producer_surplus = float((q * (P[:nsteps] - c)).sum())
    consumer_surplus = sales_total / a
    welfare_before_fixed = producer_surplus + consumer_surplus
    terminal = mu[-1]
    n0_grid = np.arange(N + 1).reshape(N + 1, 1, 1)
    vacancy = float((terminal * n0_grid).sum() / N)
    terminal_unsold = N * vacancy
    vacancy_tax_payment = tau * terminal_unsold
    ticket_tax_payment = ticket_tax * sales_total
    tax_payment = vacancy_tax_payment + ticket_tax_payment
    firm_operating_profit_before_fixed = revenue - c * sales_total - tax_payment
    out = {'gross_revenue': revenue, 'sales': sales_total, 'sold_fraction': sales_total / N, 'unsold': terminal_unsold, 'sales_identity_gap': abs(N - sales_total - terminal_unsold), 'vacancy_departing': vacancy, 'producer_surplus_before_fixed': producer_surplus, 'consumer_surplus': consumer_surplus, 'welfare_before_fixed': welfare_before_fixed, 'tax_payment': tax_payment, 'vacancy_tax_payment': vacancy_tax_payment, 'ticket_tax_payment': ticket_tax_payment, 'firm_operating_profit_before_fixed': firm_operating_profit_before_fixed, 'calendar_revenue': revenue / d, 'calendar_welfare_rate': welfare_before_fixed / d, 'sales_by_rank': [float(q[..., r].sum()) for r in range(3)]}
    if disp_per_buyer is not None:
        out['displacement_cost_diagnostic'] = float((q * disp_per_buyer[:nsteps]).sum())
        out['mean_displacement_per_buyer'] = out['displacement_cost_diagnostic'] / sales_total if sales_total > 0 else 0.0
    return out

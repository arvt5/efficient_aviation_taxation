from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from aviation_taxation.network import FractionalDEQN, demand_torch, rank_values
from aviation_taxation.model import FractionalParams

def load_model(path: Path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    params = FractionalParams(**payload['params'])
    hidden = int(payload.get('hidden', 128))
    activation = payload.get('activation', 'silu')
    net = FractionalDEQN(hidden=hidden, activation=activation)
    net.load_state_dict(payload['model_state_dict'])
    net.eval()
    return (net, params)

def build_tables(net, params: FractionalParams, phi_grid: np.ndarray):
    """P, D, Vnet tables of shape [K, Np, Np, Np, 3]."""
    N = params.N
    Np = N + 1
    K = len(phi_grid)
    states = np.array([(x, y, z) for x in range(Np) for y in range(Np) for z in range(Np)], dtype=np.float64)
    S = len(states)
    dtype = next(net.parameters()).dtype
    states_t = torch.tensor(states, dtype=dtype)
    P = np.zeros((K, Np, Np, Np, 3))
    D = np.zeros((K, Np, Np, Np, 3))
    Vnet = np.zeros((K, Np, Np, Np, 3))
    with torch.no_grad():
        for i, phi in enumerate(phi_grid):
            phi_t = torch.full((S,), float(phi), dtype=dtype)
            p_i, v_i = rank_values(net, phi_t, states_t, params)
            d_i = demand_torch(p_i, phi_t, states_t, params)
            P[i] = p_i.numpy().reshape(Np, Np, Np, 3)
            D[i] = d_i.numpy().reshape(Np, Np, Np, 3)
            Vnet[i] = v_i.numpy().reshape(Np, Np, Np, 3)
    return (P, D, Vnet)

def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'n': 0, 'mean': None, 'median': None, 'p90': None, 'p99': None, 'max': None}
    return {'n': int(x.size), 'mean': float(x.mean()), 'median': float(np.median(x)), 'p90': float(np.percentile(x, 90)), 'p99': float(np.percentile(x, 99)), 'max': float(x.max())}

def weighted_summarize(x: np.ndarray, w: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    x = x[mask]
    w = w[mask]
    if x.size == 0 or w.sum() <= 0.0:
        return {'n': int(x.size), 'total_weight': 0.0, 'mean': None, 'median': None, 'p90': None, 'p99': None, 'max': None}
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws) / ws.sum()

    def q(p: float) -> float:
        return float(xs[min(int(np.searchsorted(cdf, p, side='left')), len(xs) - 1)])
    return {'n': int(x.size), 'total_weight': float(w.sum()), 'mean': float(np.sum(x * w) / w.sum()), 'median': q(0.5), 'p90': q(0.9), 'p99': q(0.99), 'max': float(xs[-1])}

def regime_name(rank: int) -> str:
    return {2: 'early_life_rank2', 1: 'middle_life_rank1', 0: 'late_life_rank0'}[int(rank)]

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_policy_eval_artifacts(out_dir: Path, phi_grid: np.ndarray, mu: np.ndarray, q: np.ndarray, P: np.ndarray, D: np.ndarray, Vnet: np.ndarray, V: np.ndarray, pe_hist: list[float], osd: dict) -> None:
    """Persist the backward policy-evaluation objects used by certification."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'policy_eval_arrays.npz', phi_grid=phi_grid, occupancy=mu, sale_mass=q, prices=P, demand=D, value_network=Vnet, value_policy_eval=V, best_response_price=osd['best_price'], one_shot_gain=osd['gain'], current_price=osd['current_price'])
    write_csv(out_dir / 'policy_eval_outer_history.csv', [{'outer': i, 'gap': float(gap)} for i, gap in enumerate(pe_hist)])

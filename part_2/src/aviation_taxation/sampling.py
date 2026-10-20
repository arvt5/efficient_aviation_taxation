from __future__ import annotations
import numpy as np
import torch
from aviation_taxation.model import FractionalParams

def _state_grid(N: int) -> np.ndarray:
    return np.array([(n0, n1, n2) for n0 in range(N + 1) for n1 in range(N + 1) for n2 in range(N + 1)], dtype=np.float32)

def _phase_region_weights(phi_grid: np.ndarray, params: FractionalParams, args) -> np.ndarray:
    phi = phi_grid[:-1]
    w = np.ones_like(phi, dtype=np.float64)
    w[phi <= args.phase_edge_width] *= args.align_phase_edge_weight
    w[phi >= 1.0 - args.phase_edge_width] *= args.align_phase_edge_weight
    if params.entry_phi > 0.0 and args.align_entry_weight != 1.0:
        near_entry = np.abs(phi - float(params.entry_phi)) <= args.entry_width
        w[near_entry] *= args.align_entry_weight
    return w

def _build_alignment_sampling(mu: np.ndarray, phi_grid: np.ndarray, params: FractionalParams, args) -> dict:
    """Build a probability vector over stocked-active (phase,state,rank) targets."""
    N = params.N
    Np = N + 1
    states = _state_grid(N).astype(np.int64)
    S = states.shape[0]
    K = len(phi_grid) - 1
    phase_w = _phase_region_weights(phi_grid, params, args)
    rows = []
    probs = []
    rank_grid = np.arange(3)
    for k in range(K):
        active2 = phi_grid[k] >= params.entry_phi
        for s, (n0, n1, n2) in enumerate(states):
            ns = (n0, n1, n2)
            occ = float(mu[k].reshape(S)[s])
            for r in rank_grid:
                if r == 2 and (not active2):
                    continue
                if ns[r] <= 0:
                    continue
                rows.append((k, n0, n1, n2, int(r)))
                base = 1.0
                if args.align_sampling in ('occupancy', 'mixed'):
                    base = occ
                    if args.align_sampling == 'mixed':
                        base = args.align_uniform_floor + base
                if min(ns) <= 1:
                    base *= args.align_low_inventory_weight
                base *= phase_w[k]
                probs.append(max(base, 0.0))
    rows_np = np.array(rows, dtype=np.int64)
    probs_np = np.array(probs, dtype=np.float64)
    if probs_np.sum() <= 0.0:
        probs_np[:] = 1.0
    probs_np = probs_np / probs_np.sum()
    return {'rows': rows_np, 'probs': probs_np.astype(np.float32)}

def _sample_alignment_batch(targets: dict, params: FractionalParams, batch_size: int, device: torch.device):
    sampling = targets['sampling']
    rows = sampling['rows']
    probs = torch.tensor(sampling['probs'], device=device)
    pick = torch.multinomial(probs, batch_size, replacement=True).cpu().numpy()
    batch_rows = rows[pick]
    k_np = batch_rows[:, 0]
    n_np = batch_rows[:, 1:4]
    r_np = batch_rows[:, 4]
    phi_grid = targets['phi_grid']
    Vtarget = targets['value_policy_eval']
    phi = torch.tensor(phi_grid[k_np], dtype=torch.float32, device=device)
    n = torch.tensor(n_np, dtype=torch.float32, device=device)
    r = torch.tensor(r_np, dtype=torch.float32, device=device)
    vt = torch.tensor(Vtarget[k_np, n_np[:, 0], n_np[:, 1], n_np[:, 2], r_np], dtype=torch.float32, device=device)
    n_red_np = n_np.copy()
    n_red_np[np.arange(len(r_np)), r_np] -= 1
    vt_red = torch.tensor(Vtarget[k_np, n_red_np[:, 0], n_red_np[:, 1], n_red_np[:, 2], r_np], dtype=torch.float32, device=device)
    n_red = torch.tensor(n_red_np, dtype=torch.float32, device=device)
    return (phi, n, n_red, r, vt, vt_red, torch.tensor(k_np, dtype=torch.long, device=device))

from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from aviation_taxation.network import demand_torch
from aviation_taxation.forward import phase_shift, propagate_one_phase
from aviation_taxation.model import FractionalParams

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))

def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def state_grid(N: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.cartesian_prod(torch.arange(N + 1, device=device, dtype=dtype), torch.arange(N + 1, device=device, dtype=dtype), torch.arange(N + 1, device=device, dtype=dtype))

def demand_tables_from_policy(P: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, *, device: torch.device, dtype: torch.dtype=torch.float32) -> np.ndarray:
    """Evaluate demand for an explicit policy table."""
    Np = params.N + 1
    states = state_grid(params.N, device=device, dtype=dtype)
    S = states.shape[0]
    D = np.zeros_like(P)
    with torch.no_grad():
        for i, phi in enumerate(phi_grid):
            p = torch.as_tensor(P[i].reshape(S, 3), device=device, dtype=dtype)
            ph = torch.full((S,), float(phi), device=device, dtype=dtype)
            D[i] = demand_torch(p, ph, states, params).cpu().numpy().reshape(Np, Np, Np, 3)
    return D

def stationary_forward_tables(P: np.ndarray, D: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, burn_in: int):
    """Stationary forward propagation for an explicit price/demand table."""
    N = params.N
    mu0 = np.zeros((N + 1, N + 1, N + 1))
    if params.entry_phi == 0.0:
        mu0[N, N, N] = 1.0
    else:
        mu0[N, N, 0] = 1.0
    for _ in range(max(int(burn_in), 0)):
        mu, q = propagate_one_phase(P, D, params, phi_grid, mu0)
        mu0 = phase_shift(mu[-1], params)
    return (*propagate_one_phase(P, D, params, phi_grid, mu0),)

def _continuation_and_red(V: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, phase_index: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    Np = params.N + 1
    frac = params.entry_phi > 0.0
    entry_idx = int(np.argmax(phi_grid >= params.entry_phi)) if frac else 0
    cross = entry_idx - 1
    if frac and phase_index == cross and (rank in (0, 1)):
        cont = np.broadcast_to(V[phase_index + 1, :, :, params.N, rank][:, :, None], (Np, Np, Np))
    else:
        cont = V[phase_index + 1, ..., rank]
    red = np.zeros((Np, Np, Np))
    hi = [slice(None)] * 3
    lo = [slice(None)] * 3
    hi[rank] = slice(1, Np)
    lo[rank] = slice(0, Np - 1)
    red[tuple(hi)] = V[phase_index, ..., rank][tuple(lo)]
    return (cont, red)

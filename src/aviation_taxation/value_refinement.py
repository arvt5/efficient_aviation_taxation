from __future__ import annotations
import numpy as np
import torch
from aviation_taxation.tables import build_tables
from aviation_taxation.network import active2_torch, sample_interior
from aviation_taxation.forward import stationary_forward
from aviation_taxation.model import FractionalParams
from aviation_taxation.sampling import _build_alignment_sampling, _sample_alignment_batch
from aviation_taxation.evaluation import policy_eval_full
from aviation_taxation.equilibrium import loss_fn as residual_loss_fn
RESID_WEIGHTS = {'foc': 1.0, 'eta_violation': 1.0, 'hjb': 3.0, 'stockout': 1.0, 'inactive_slot': 0.1, 'departure0': 1.0, 'periodic_value': 1.0, 'entry_value': 1.0}

def make_targets(net, params: FractionalParams, n_phi: int, burn_in: int, cross_weight: float, args) -> dict:
    """Freeze the policy and build full-rationality value/shadow targets."""
    phi_grid = np.linspace(0.0, 1.0, n_phi)
    P, D, Vnet = build_tables(net, params, phi_grid)
    Veval, pe_hist = policy_eval_full(P, D, params, phi_grid, cross_weight=cross_weight)
    mu, _, _, _ = stationary_forward(net, params, phi_grid, burn_in=burn_in)
    return {'phi_grid': phi_grid.astype(np.float32), 'value_policy_eval': Veval.astype(np.float32), 'occupancy': mu.astype(np.float64), 'policy_eval_history': pe_hist, 'sampling': _build_alignment_sampling(mu, phi_grid, params, args)}

def alignment_loss(net, targets: dict, params: FractionalParams, args) -> tuple[torch.Tensor, dict]:
    """Value + own-shadow + cross-shadow alignment against the full V_eval."""
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    phi, n, n_red_own, r, vt, vt_red_own, k = _sample_alignment_batch(targets, params, args.align_batch, device)
    phi = phi.to(dtype)
    n = n.to(dtype)
    n_red_own = n_red_own.to(dtype)
    r = r.to(dtype)
    vt = vt.to(dtype)
    vt_red_own = vt_red_own.to(dtype)
    a2 = active2_torch(phi, params).to(dtype)
    _, v = net(phi, r, n, a2, params.M, params.N)
    _, v_red_own = net(phi, r, n_red_own, a2, params.M, params.N)
    value = ((v - vt) ** 2).mean()
    shadow_own = ((v - v_red_own - (vt - vt_red_own)) ** 2).mean()
    Vtarget = targets['value_policy_eval']
    k_np = k.cpu().numpy()
    n_np = n.detach().cpu().numpy().astype(np.int64)
    r_np = r.detach().cpu().numpy().astype(np.int64)
    phi_np = phi.detach().cpu().numpy()
    cross_terms = []
    for j in range(3):
        sel = (r_np != j) & (n_np[:, j] > 0)
        if j == 2:
            sel = sel & (phi_np >= float(params.entry_phi))
        if not sel.any():
            continue
        n_red_j = n_np.copy()
        n_red_j[:, j] = np.maximum(n_red_j[:, j] - 1, 0)
        vt_red_j = torch.tensor(Vtarget[k_np, n_red_j[:, 0], n_red_j[:, 1], n_red_j[:, 2], r_np], dtype=dtype, device=device)
        n_red_j_t = torch.tensor(n_red_j, dtype=dtype, device=device)
        _, v_red_j = net(phi, r, n_red_j_t, a2, params.M, params.N)
        d = v - v_red_j - (vt - vt_red_j)
        sel_t = torch.tensor(sel, device=device)
        cross_terms.append(d[sel_t] ** 2)
    if cross_terms:
        shadow_cross = torch.cat(cross_terms).mean()
    else:
        shadow_cross = torch.zeros((), dtype=dtype, device=device)
    total = float(args.value_weight) * value + float(args.shadow_weight) * shadow_own + float(args.cross_shadow_weight) * shadow_cross
    return (total, {'value_align': value.detach(), 'shadow_own_align': shadow_own.detach(), 'shadow_cross_align': shadow_cross.detach()})

def _refine_namespace(value_batch, w_eval, shadow_weight, cross_shadow_weight, burn_in):
    """SimpleNamespace consumed by make_targets / alignment_loss during refine()."""
    from types import SimpleNamespace
    return SimpleNamespace(align_batch=value_batch, value_weight=w_eval, shadow_weight=shadow_weight, cross_shadow_weight=cross_shadow_weight, cross_weight=1.0, burn_in=burn_in, monitor_n_phi=51, monitor_n_price=41, align_sampling='mixed', align_uniform_floor=0.02, align_low_inventory_weight=2.0, phase_edge_width=0.06, align_phase_edge_weight=2.0, align_entry_weight=2.0, entry_width=0.06)

def refine(net, params: FractionalParams, n_outer=3, inner_iters=2000, w_eval=1.0, dphi=0.02, lr=0.0001, batch=512, value_batch=2048, n_phi=101, burn_in=20, device=None, shadow_weight=1.0, cross_shadow_weight=1.0, log=False):
    """Full-rationality value+own+cross-shadow supervision FPI (callable, GPU-aware).

    Drop-in for `stage5.refine.refine` (same core signature). Each outer round
    freezes the policy, solves the full-rationality `V_eval` on CPU, then trains
    `inner_iters` of full residual + value/own/cross-shadow alignment on `device`.
    Returns the net on `device`.
    """
    import torch as _torch
    dev = device if device is not None else next(net.parameters()).device
    aargs = _refine_namespace(value_batch, w_eval, shadow_weight, cross_shadow_weight, burn_in)
    weights = RESID_WEIGHTS
    two_firm = 0.4 if params.entry_phi > 0 else None
    opt = _torch.optim.Adam(net.parameters(), lr=lr)
    for outer in range(n_outer):
        net_cpu = net.to('cpu')
        targets = make_targets(net_cpu, params, n_phi, burn_in, 1.0, aargs)
        net = net_cpu.to(dev)
        for _ in range(inner_iters):
            opt.zero_grad()
            batch_s = sample_interior(batch, params, dphi, device=dev, two_firm_fraction=two_firm)
            l_res, _ = residual_loss_fn(net, batch_s, params, dphi, weights=weights, cross_weight=1.0)
            l_align, _ = alignment_loss(net, targets, params, aargs)
            (l_res + l_align).backward()
            opt.step()
        if log:
            print(f'  [refine outer {outer}] done', flush=True)
    return net

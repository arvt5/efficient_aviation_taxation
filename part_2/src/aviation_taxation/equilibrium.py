from __future__ import annotations
import numpy as np
import torch
from aviation_taxation.network import NSLOTS, active_rank_mask, boundary_residuals, demand_torch, fischer_burmeister, rank_values, sample_interior, successor_value, _device
from aviation_taxation.model import FractionalParams

def _reduced_slot_values(net, phi, n, params: FractionalParams):
    """Per-slot value tables after decrementing one inventory slot.

    Returns a list `Vred` of length NSLOTS where `Vred[s][:, r]` is the value at
    rank r evaluated at the state with slot s reduced by one (clamped at 0). The
    own-rank shadow uses `Vred[r][:, r]`; the cross shadow Delta_j V_r uses
    `Vred[j][:, r]`.
    """
    out = []
    for s in range(NSLOTS):
        n_red = n.clone()
        n_red[:, s] = torch.clamp(n_red[:, s] - 1.0, min=0.0)
        _, V_red_all = rank_values(net, phi, n_red, params)
        out.append(V_red_all)
    return out

def residuals(net, phi, n, params: FractionalParams, dphi: float, cross_weight: float=1.0):
    """Full-rationality scaled-FOC, eta-violation, and HJB residuals.

    Mirrors `aviation_taxation.network.residuals`; at `cross_weight == 0.0` the returned tensors
    are bit-for-bit identical to it.
    """
    lam_bar = float(params.lambda_bar)
    tau = float(params.tau)
    c = float(params.c)
    tt = float(params.ticket_tax)
    cw = float(cross_weight)
    p, V = rank_values(net, phi, n, params)
    p = p.clone().requires_grad_(True)
    D = demand_torch(p, phi, n, params)
    lam = lam_bar * D
    J = torch.zeros(D.shape[0], NSLOTS, NSLOTS, dtype=D.dtype, device=D.device)
    for k in range(NSLOTS):
        g = torch.autograd.grad(D[:, k].sum(), p, create_graph=True)[0]
        J[:, k, :] = g
    dlam = lam_bar * J
    active = active_rank_mask(phi, n, params)
    Vred = _reduced_slot_values(net, phi, n, params)
    out = {k: [] for k in ('price', 'eta_violation', 'old_fb_price', 'normalized_foc', 'raw_foc', 'hjb', 'margins')}
    for r in range(NSLOTS):
        stocked_active = active[:, r] & (n[:, r] > 0.5)
        V_r = V[:, r]
        V_red = Vred[r][:, r]
        dV = V_r - torch.where(stocked_active, V_red, torch.zeros_like(V_red))
        eta = p[:, r] + tau - c - tt - dV
        cross_foc = torch.zeros_like(eta)
        cross_hjb = torch.zeros_like(eta)
        if cw != 0.0:
            for j in range(NSLOTS):
                if j == r:
                    continue
                rival = active[:, j] & (n[:, j] > 0.5)
                dVj = torch.where(rival, V_r - Vred[j][:, r], torch.zeros_like(V_r))
                cross_foc = cross_foc + dlam[:, j, r] * dVj
                cross_hjb = cross_hjb + D[:, j] * dVj
        dlam_rr = dlam[:, r, r]
        foc = 1.0 + dlam_rr / (lam[:, r] + 1e-12) * eta
        raw_foc = lam[:, r] + dlam_rr * eta - cw * cross_foc
        scaled_raw_foc = raw_foc / (1.0 + lam[:, r])
        eta_violation = torch.relu(-eta)
        old_fb = fischer_burmeister(torch.clamp(eta, min=0.0), torch.abs(foc))
        z = torch.zeros_like(scaled_raw_foc)
        out['margins'].append(eta)
        out['price'].append(torch.where(stocked_active, scaled_raw_foc, z))
        out['eta_violation'].append(torch.where(stocked_active, eta_violation, z))
        out['old_fb_price'].append(torch.where(stocked_active, old_fb, z))
        out['normalized_foc'].append(torch.where(stocked_active, foc, z))
        out['raw_foc'].append(torch.where(stocked_active, raw_foc, z))
        V_next = successor_value(net, phi, r, n, params, dphi)
        dt = float(params.d) * float(dphi)
        hjb = (V_r - V_next) / (dt * lam_bar) - D[:, r] * eta + cw * cross_hjb
        out['hjb'].append(torch.where(stocked_active, hjb, torch.zeros_like(hjb)))
    out.update({'lambda': lam, 'demand': D, 'prices': p, 'values': V})
    return out

def loss_fn(net, batch, params: FractionalParams, dphi: float, weights=None, cross_weight: float=1.0):
    """Full-rationality training loss (shared boundary residuals, full interior)."""
    if weights is None:
        weights = {'foc': 1.0, 'eta_violation': 1.0, 'hjb': 1.0, 'stockout': 1.0, 'inactive_slot': 0.1, 'departure0': 1.0, 'periodic_value': 1.0, 'entry_value': 1.0}
    phi, n = batch
    res = residuals(net, phi, n, params, dphi, cross_weight=cross_weight)
    bnd = boundary_residuals(net, phi.shape[0], params)
    comp = {'foc': sum(((e ** 2).mean() for e in res['price'])), 'eta_violation': sum(((e ** 2).mean() for e in res['eta_violation'])), 'hjb': sum(((e ** 2).mean() for e in res['hjb'])), 'stockout': sum(((e ** 2).mean() for e in bnd['stockout'])), 'inactive_slot': (bnd['inactive_slot'] ** 2).mean(), 'departure0': (bnd['departure0'] ** 2).mean(), 'periodic_value': sum(((e ** 2).mean() for e in bnd['periodic_value'])), 'entry_value': sum(((e ** 2).mean() for e in bnd['entry_value']))}
    total = sum((float(weights[k]) * comp[k] for k in comp))
    return (total, comp)

def _cross_weight_at(it: int, n_iter: int, ramp_frac: float, final: float) -> float:
    """Linear cross-weight ramp 0 -> `final` over the first `ramp_frac` of training."""
    if ramp_frac <= 0.0:
        return final
    ramp_iters = max(1, int(ramp_frac * n_iter))
    return final * min(1.0, it / ramp_iters)

def train(net, params: FractionalParams, dphi: float=0.02, n_iter: int=20000, B: int=512, lr: float=0.0003, log_every: int=2000, weights=None, two_firm_fraction: float | None=None, cross_weight: float=1.0, cross_ramp_frac: float=0.25):
    """Warm-startable full-rationality training with a 0 -> cross_weight ramp.

    The ramp reuses a mature shared solution (cross_weight ~ 0 early) and eases
    into the tighter full-rationality fixed point.
    """
    device = _device(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for it in range(n_iter):
        cw = _cross_weight_at(it, n_iter, cross_ramp_frac, cross_weight)
        batch = sample_interior(B, params, dphi, device=device, two_firm_fraction=two_firm_fraction)
        opt.zero_grad()
        L, comp = loss_fn(net, batch, params, dphi, weights=weights, cross_weight=cw)
        L.backward()
        opt.step()
        if it % log_every == 0 or it == n_iter - 1:
            parts = ' '.join((f'{k}={v.item():.2e}' for k, v in comp.items()))
            print(f'  iter {it:6d} cw={cw:.3f} loss={L.item():.3e} {parts}')
    return net

def residual_stats(net, params: FractionalParams, dphi: float=0.02, B: int=20000, cross_weight: float=1.0):
    device = _device(net)
    phi, n = sample_interior(B, params, dphi, device=device)
    res = residuals(net, phi, n, params, dphi, cross_weight=cross_weight)
    bnd = boundary_residuals(net, B, params)

    def cat(xs):
        return torch.cat([e.detach().abs().reshape(-1) for e in xs]).cpu().numpy()
    arrays = {'foc': cat(res['price']), 'eta_violation': cat(res['eta_violation']), 'raw_foc': cat(res['raw_foc']), 'hjb': cat(res['hjb']), 'stockout': cat(bnd['stockout']), 'inactive_slot': bnd['inactive_slot'].detach().abs().cpu().numpy(), 'departure0': bnd['departure0'].detach().abs().cpu().numpy(), 'periodic_value': cat(bnd['periodic_value']), 'entry_value': cat(bnd['entry_value'])}
    out = {}
    for name, arr in arrays.items():
        out[name] = {'mean': float(arr.mean()), 'median': float(np.median(arr)), 'p90': float(np.percentile(arr, 90)), 'p999': float(np.percentile(arr, 99.9))}
    return out

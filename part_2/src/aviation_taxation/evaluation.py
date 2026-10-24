from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from aviation_taxation.tables import build_tables, load_model, regime_name, summarize, weighted_summarize, write_csv, write_policy_eval_artifacts
from aviation_taxation.network import demand_torch
from aviation_taxation.forward import forward_moments, stationary_forward
from aviation_taxation.model import FractionalParams
from aviation_taxation.equilibrium import residual_stats

def _shift_down(cube: np.ndarray, axis: int, Np: int) -> np.ndarray:
    """Value table after decrementing one rival's inventory along `axis`.

    `out[..., m, ...] = cube[..., m-1, ...]` for m >= 1, and 0 at m == 0 (where the
    rival is stocked out and cannot sell, so the term is killed by a zero hazard).
    """
    out = np.zeros_like(cube)
    src = [slice(None)] * cube.ndim
    dst = [slice(None)] * cube.ndim
    src[axis] = slice(0, Np - 1)
    dst[axis] = slice(1, Np)
    out[tuple(dst)] = cube[tuple(src)]
    return out

def policy_eval_full(P, D, params: FractionalParams, phi_grid: np.ndarray, max_iter: int=200, tol: float=1e-10, cross_weight: float=1.0):
    """Full-rationality fixed-policy value fixed point. `cross_weight == 0` matches
    `aviation_taxation.tables.policy_eval` bit-for-bit."""
    N = params.N
    Np = N + 1
    d = float(params.d)
    lam = float(params.lambda_bar)
    tau = float(params.tau)
    ticket_tax = float(params.ticket_tax)
    c = float(params.c)
    cw = float(cross_weight)
    K = len(phi_grid)
    frac = params.entry_phi > 0.0
    fresh = 0 if frac else N
    entry_idx = int(np.argmax(phi_grid >= params.entry_phi)) if frac else 0
    cross = entry_idx - 1
    V = np.zeros((K, Np, Np, Np, 3))
    Vstart = np.zeros((Np, Np, Np, 3))
    history = []
    for outer in range(max_iter):
        V[K - 1, :, :, :, 0] = 0.0
        V[K - 1, :, :, :, 1] = np.broadcast_to(Vstart[:, :, fresh, 0][None, :, :], (Np, Np, Np))
        V[K - 1, :, :, :, 2] = np.broadcast_to(Vstart[:, :, fresh, 1][None, :, :], (Np, Np, Np))
        for i in range(K - 2, -1, -1):
            dt = d * float(phi_grid[i + 1] - phi_grid[i])
            for r in range(3):
                if r == 2 and frac and (i < entry_idx):
                    continue
                sale = np.minimum(dt * lam * D[i, ..., r], 1.0)
                if frac and i == cross and (r in (0, 1)):
                    cont = np.broadcast_to(V[i + 1, :, :, N, r][:, :, None], (Np, Np, Np))
                else:
                    cont = V[i + 1, ..., r]
                cross_num = np.zeros((Np, Np, Np))
                cross_den = np.zeros((Np, Np, Np))
                if cw != 0.0:
                    for j in range(3):
                        if j == r:
                            continue
                        sj = np.minimum(dt * lam * D[i, ..., j], 1.0)
                        cross_num = cross_num + sj * _shift_down(np.ascontiguousarray(cont), j, Np)
                        cross_den = cross_den + sj
                pr = P[i, ..., r]
                Vi_r = np.zeros((Np, Np, Np))
                for nr in range(1, Np):
                    sl = [slice(None)] * 3
                    sl[r] = nr
                    sl_red = [slice(None)] * 3
                    sl_red[r] = nr - 1
                    s = sale[tuple(sl)]
                    Vi_r[tuple(sl)] = (cont[tuple(sl)] + s * (pr[tuple(sl)] + tau - c - ticket_tax + Vi_r[tuple(sl_red)]) + cw * cross_num[tuple(sl)]) / (1.0 + s + cw * cross_den[tuple(sl)])
                V[i, ..., r] = Vi_r
        gap = float(np.max(np.abs(V[0] - Vstart)))
        history.append(gap)
        Vstart = V[0].copy()
        if gap < tol:
            break
    return (V, history)

def one_shot_full(P, D, V, params: FractionalParams, phi_grid: np.ndarray, net, price_min: float=0.02, price_max: float=6.0, n_price: int=81, cross_weight: float=1.0):
    """Full-rationality one-shot best-response gains. `cross_weight == 0` matches
    `aviation_taxation.tables.one_shot` bit-for-bit."""
    N = params.N
    Np = N + 1
    d = float(params.d)
    lam = float(params.lambda_bar)
    tau = float(params.tau)
    ticket_tax = float(params.ticket_tax)
    c = float(params.c)
    cw = float(cross_weight)
    K = len(phi_grid)
    frac = params.entry_phi > 0.0
    entry_idx = int(np.argmax(phi_grid >= params.entry_phi)) if frac else 0
    cross = entry_idx - 1
    S = Np ** 3
    states = np.array([(x, y, z) for x in range(Np) for y in range(Np) for z in range(Np)], dtype=np.float64)
    states_t = torch.tensor(states, dtype=next(net.parameters()).dtype)
    pgrid = np.linspace(price_min, price_max, n_price)
    best_price = np.full((K, Np, Np, Np, 3), np.nan)
    gain = np.zeros((K, Np, Np, Np, 3))
    cur_price = P.copy()
    with torch.no_grad():
        for i in range(K - 1):
            dt = d * float(phi_grid[i + 1] - phi_grid[i])
            Pi = P[i].reshape(S, 3)
            for r in range(3):
                p_batch = np.repeat(Pi[:, None, :], n_price, axis=1)
                p_batch[:, :, r] = pgrid[None, :]
                pb = torch.tensor(p_batch.reshape(S * n_price, 3), dtype=states_t.dtype)
                nb = states_t.repeat_interleave(n_price, dim=0)
                phib = torch.full((S * n_price,), float(phi_grid[i]), dtype=states_t.dtype)
                Dfull = demand_torch(pb, phib, nb, params).numpy().reshape(Np, Np, Np, n_price, 3)
                Dr = Dfull[..., r]
                sale = np.minimum(dt * lam * Dr, 1.0)
                if frac and i == cross and (r in (0, 1)):
                    cont_base = np.broadcast_to(V[i + 1, :, :, N, r][:, :, None], (Np, Np, Np))
                else:
                    cont_base = V[i + 1, ..., r]
                cont = cont_base[..., None]
                red = np.zeros((Np, Np, Np))
                idx = [slice(None)] * 3
                idx_lo = [slice(None)] * 3
                idx[r] = slice(1, Np)
                idx_lo[r] = slice(0, Np - 1)
                red[tuple(idx)] = V[i, ..., r][tuple(idx_lo)]
                red = red[..., None]
                cross_num = np.zeros((Np, Np, Np, n_price))
                cross_den = np.zeros((Np, Np, Np, n_price))
                cross_num_cur = np.zeros((Np, Np, Np))
                cross_den_cur = np.zeros((Np, Np, Np))
                if cw != 0.0:
                    for j in range(3):
                        if j == r:
                            continue
                        Vshift_j = _shift_down(np.ascontiguousarray(cont_base), j, Np)
                        sale_j = np.minimum(dt * lam * Dfull[..., j], 1.0)
                        cross_num = cross_num + sale_j * Vshift_j[..., None]
                        cross_den = cross_den + sale_j
                        sale_j_cur = np.minimum(dt * lam * D[i, ..., j], 1.0)
                        cross_num_cur = cross_num_cur + sale_j_cur * Vshift_j
                        cross_den_cur = cross_den_cur + sale_j_cur
                rhs = (cont + sale * (pgrid[None, None, None, :] + tau - c - ticket_tax + red) + cw * cross_num) / (1.0 + sale + cw * cross_den)
                k_best = np.argmax(rhs, axis=-1)
                best_rhs = np.take_along_axis(rhs, k_best[..., None], axis=-1)[..., 0]
                best_price[i, ..., r] = pgrid[k_best]
                sale_cur = np.minimum(dt * lam * D[i, ..., r], 1.0)
                cur_cont = cont[..., 0]
                cur_rhs = (cur_cont + sale_cur * (P[i, ..., r] + tau - c - ticket_tax + red[..., 0]) + cw * cross_num_cur) / (1.0 + sale_cur + cw * cross_den_cur)
                gain[i, ..., r] = best_rhs - cur_rhs
    return {'best_price': best_price, 'gain': gain, 'current_price': cur_price}

def foc_at_eval_full(P, D, Vnet, V, params: FractionalParams, phi_grid: np.ndarray, net, mu, h: float=0.001, cross_weight: float=1.0) -> dict:
    """Full-rationality FOC-at-value diagnostic. `cross_weight == 0` matches
    `aviation_taxation.tables.foc_at_eval` bit-for-bit."""
    N = params.N
    Np = N + 1
    lam_bar = float(params.lambda_bar)
    tau = float(params.tau)
    c = float(params.c)
    ticket_tax = float(params.ticket_tax)
    cw = float(cross_weight)
    S = Np ** 3
    states = np.array([(x, y, z) for x in range(Np) for y in range(Np) for z in range(Np)], dtype=np.float64)
    states_t = torch.tensor(states, dtype=next(net.parameters()).dtype)
    dJ = np.zeros(D.shape + (3,))
    with torch.no_grad():
        for i, phi in enumerate(phi_grid):
            Pi = P[i].reshape(S, 3)
            phib = torch.full((S,), float(phi), dtype=states_t.dtype)
            for r in range(3):
                pp = Pi.copy()
                pp[:, r] += h
                pm = Pi.copy()
                pm[:, r] -= h
                Dp = demand_torch(torch.tensor(pp, dtype=states_t.dtype), phib, states_t, params).numpy()
                Dm = demand_torch(torch.tensor(pm, dtype=states_t.dtype), phib, states_t, params).numpy()
                deriv = ((Dp - Dm) / (2.0 * h)).reshape(Np, Np, Np, 3)
                for k in range(3):
                    dJ[i, ..., k, r] = deriv[..., k]

    def shadow_axis(Varr, r, j):
        dV = np.zeros_like(Varr[..., r])
        sl = [slice(None)] * 4
        slo = [slice(None)] * 4
        sl[1 + j] = slice(1, Np)
        slo[1 + j] = slice(0, Np - 1)
        dV[tuple(sl)] = Varr[..., r][tuple(sl)] - Varr[..., r][tuple(slo)]
        return dV
    lam = lam_bar * D
    w = mu[:-1]
    out = {}
    for r in range(3):
        eta_net = P[..., r] + tau - c - ticket_tax - shadow_axis(Vnet, r, r)
        eta_evl = P[..., r] + tau - c - ticket_tax - shadow_axis(V, r, r)
        cross_net = np.zeros_like(eta_net)
        cross_evl = np.zeros_like(eta_evl)
        if cw != 0.0:
            for j in range(3):
                if j == r:
                    continue
                dlam_jr = lam_bar * dJ[..., j, r]
                cross_net = cross_net + dlam_jr * shadow_axis(Vnet, r, j)
                cross_evl = cross_evl + dlam_jr * shadow_axis(V, r, j)
        dlam_rr = lam_bar * dJ[..., r, r]
        sc_net = (lam[..., r] + dlam_rr * eta_net - cw * cross_net) / (1.0 + lam[..., r])
        sc_evl = (lam[..., r] + dlam_rr * eta_evl - cw * cross_evl) / (1.0 + lam[..., r])
        shp = [1, 1, 1]
        shp[r] = Np
        stk = np.broadcast_to(np.arange(Np).reshape(shp) > 0, (Np, Np, Np))[None]
        ww = w * np.broadcast_to(stk, w.shape)
        s = ww.sum()
        fn = float((ww * np.abs(sc_net[:-1])).sum() / s)
        fe = float((ww * np.abs(sc_evl[:-1])).sum() / s)
        out[f'rank{r}'] = {'foc_at_Vnet_occ': fn, 'foc_at_Veval_occ': fe, 'ratio_eval_over_net': fe / fn if fn > 0 else None}
    alln = float(np.mean([out[f'rank{r}']['foc_at_Vnet_occ'] for r in range(3)]))
    alle = float(np.mean([out[f'rank{r}']['foc_at_Veval_occ'] for r in range(3)]))
    out['overall'] = {'foc_at_Vnet_occ': alln, 'foc_at_Veval_occ': alle, 'ratio_eval_over_net': alle / alln if alln > 0 else None}
    return out

def certify(net, params: FractionalParams, n_phi: int=101, burn_in: int=20, n_price: int=81, dphi_resid: float=0.02, top_k: int=200, artifact_dir: Path | None=None, cross_weight: float=1.0):
    """Full-rationality certification summary (mirrors `aviation_taxation.tables.certify`)."""
    N = params.N
    Np = N + 1
    phi_grid = np.linspace(0.0, 1.0, n_phi)
    res = residual_stats(net, params, dphi=dphi_resid, B=20000, cross_weight=cross_weight)
    mu, q, Pf, Df = stationary_forward(net, params, phi_grid, burn_in=burn_in)
    masses = mu.sum(axis=(1, 2, 3))
    mom = forward_moments(mu, q, Pf, params)
    P, D, Vnet = build_tables(net, params, phi_grid)
    V, pe_hist = policy_eval_full(P, D, params, phi_grid, cross_weight=cross_weight)
    osd = one_shot_full(P, D, V, params, phi_grid, net, n_price=n_price, cross_weight=cross_weight)
    foc_consistency = foc_at_eval_full(P, D, Vnet, V, params, phi_grid, net, mu, cross_weight=cross_weight)
    if artifact_dir is not None:
        write_policy_eval_artifacts(Path(artifact_dir), phi_grid, mu, q, P, D, Vnet, V, pe_hist, osd)
    g = osd['gain'][:-1]
    dP = np.abs(osd['best_price'][:-1] - osd['current_price'][:-1])
    nr_grid = [np.arange(Np).reshape(*[1] * k + [Np] + [1] * (2 - k)) for k in range(3)]
    stocked = np.stack([np.broadcast_to(nr_grid[r] > 0, (Np, Np, Np)) for r in range(3)], axis=-1)
    stocked = np.broadcast_to(stocked[None], (n_phi - 1, Np, Np, Np, 3))
    active = np.ones_like(stocked, dtype=bool)
    active[:, :, :, :, 2] = (phi_grid[:-1] >= params.entry_phi).reshape(n_phi - 1, 1, 1, 1)
    stocked = stocked & active
    w = mu[:-1, :, :, :, None] * np.ones((1, 1, 1, 1, 3))
    w = w * stocked
    gpos = np.where(stocked, np.maximum(g, 0.0), 0.0)
    dPm = np.where(stocked, dP, 0.0)
    wsum = w.sum()
    gain_flat = gpos[stocked]
    dP_flat = dPm[stocked]
    weight_flat = w[stocked]
    rank_flat = np.broadcast_to(np.arange(3).reshape(1, 1, 1, 1, 3), stocked.shape)[stocked]
    gain_by_rank = {}
    gain_weighted_by_rank = {}
    price_dev_by_rank = {}
    value_diff = np.abs(Vnet[:-1] - V[:-1])
    value_diff_flat = value_diff[stocked]
    value_diff_by_rank = {}
    value_diff_weighted_by_rank = {}
    for r in range(3):
        m = rank_flat == r
        gain_by_rank[f'rank{r}'] = summarize(gain_flat[m])
        gain_weighted_by_rank[f'rank{r}'] = weighted_summarize(gain_flat[m], weight_flat[m])
        price_dev_by_rank[f'rank{r}'] = {'unweighted': summarize(dP_flat[m]), 'occupancy_weighted': weighted_summarize(dP_flat[m], weight_flat[m])}
        value_diff_by_rank[f'rank{r}'] = summarize(value_diff_flat[m])
        value_diff_weighted_by_rank[f'rank{r}'] = weighted_summarize(value_diff_flat[m], weight_flat[m])
    gain_by_regime = {regime_name(r): gain_by_rank[f'rank{r}'] for r in range(3)}
    gain_weighted_by_regime = {regime_name(r): gain_weighted_by_rank[f'rank{r}'] for r in range(3)}
    price_dev_by_regime = {regime_name(r): price_dev_by_rank[f'rank{r}'] for r in range(3)}
    value_diff_by_regime = {regime_name(r): value_diff_by_rank[f'rank{r}'] for r in range(3)}
    value_diff_weighted_by_regime = {regime_name(r): value_diff_weighted_by_rank[f'rank{r}'] for r in range(3)}
    top_candidates = []
    for i in range(n_phi - 1):
        phi = float(phi_grid[i])
        for n0 in range(Np):
            for n1 in range(Np):
                for n2 in range(Np):
                    occ = float(mu[i, n0, n1, n2])
                    for r in range(3):
                        if not stocked[i, n0, n1, n2, r]:
                            continue
                        gain_val = float(gpos[i, n0, n1, n2, r])
                        current_price = float(osd['current_price'][i, n0, n1, n2, r])
                        best_price = float(osd['best_price'][i, n0, n1, n2, r])
                        weighted_gain = occ * gain_val
                        top_candidates.append((weighted_gain, gain_val, {'phi': phi, 'rank': r, 'regime': regime_name(r), 'n0': n0, 'n1': n1, 'n2': n2, 'state_probability': occ, 'current_price': current_price, 'best_price': best_price, 'abs_price_deviation': abs(best_price - current_price), 'gain': gain_val, 'weighted_gain': weighted_gain}))
    top_candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    top_rows = [row for _, _, row in top_candidates[:top_k]]
    summary = {'cross_weight': float(cross_weight), 'one_shot_gain': {'n': int(stocked.sum()), 'mean': float(gain_flat.mean()), 'median': float(np.median(gain_flat)), 'p90': float(np.percentile(gain_flat, 90)), 'p99': float(np.percentile(gain_flat, 99)), 'max': float(gain_flat.max())}, 'one_shot_gain_occupancy_weighted': weighted_summarize(gain_flat, weight_flat), 'one_shot_gain_by_rank': gain_by_rank, 'one_shot_gain_occupancy_weighted_by_rank': gain_weighted_by_rank, 'one_shot_gain_by_regime': gain_by_regime, 'one_shot_gain_occupancy_weighted_by_regime': gain_weighted_by_regime, 'price_dev_abs': {'unweighted_mean': float(dP_flat.mean()), 'occupancy_weighted_mean': float((w * dPm).sum() / wsum) if wsum > 0 else float('nan'), 'p99': float(np.percentile(dP_flat, 99)), 'max': float(dP_flat.max()), 'occ_mass_dev_gt_0p5': float((w * (dPm > 0.5)).sum() / wsum) if wsum > 0 else float('nan')}, 'price_dev_abs_by_rank': price_dev_by_rank, 'price_dev_abs_by_regime': price_dev_by_regime, 'top_one_shot_deviations': top_rows, 'value_head_abs_diff': summarize(value_diff_flat), 'value_head_abs_diff_occupancy_weighted': weighted_summarize(value_diff_flat, weight_flat), 'value_head_abs_diff_by_rank': value_diff_by_rank, 'value_head_abs_diff_occupancy_weighted_by_rank': value_diff_weighted_by_rank, 'value_head_abs_diff_by_regime': value_diff_by_regime, 'value_head_abs_diff_occupancy_weighted_by_regime': value_diff_weighted_by_regime, 'policy_eval': {'outer_iters': len(pe_hist), 'final_gap': pe_hist[-1] if pe_hist else None}, 'forward_moments': mom, 'mass_conservation': {'min': float(masses.min()), 'max': float(masses.max())}, 'foc_consistency': foc_consistency, 'residual_stats': res, 'params': asdict(params)}
    return summary

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--n-phi', type=int, default=101)
    ap.add_argument('--burn-in', type=int, default=20)
    ap.add_argument('--n-price', type=int, default=81)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--cross-weight', type=float, default=1.0)
    ap.add_argument('--artifact-dir', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    net, params = load_model(args.model)
    print(f'loaded {args.model}\nparams: M={params.M} lambda_bar={params.lambda_bar:.4f} c={params.c:.4f} ticket_tax={params.ticket_tax:.4f} N={params.N} cross_weight={args.cross_weight}')
    summary = certify(net, params, n_phi=args.n_phi, burn_in=args.burn_in, n_price=args.n_price, top_k=args.top_k, artifact_dir=args.artifact_dir, cross_weight=args.cross_weight)
    out = args.out or args.model.parent / 'certify_full_result.json'
    out.write_text(json.dumps(summary, indent=2))
    top_csv = out.with_name(out.stem + '_top_deviations.csv')
    write_csv(top_csv, summary['top_one_shot_deviations'])
    g = summary['one_shot_gain']
    gw = summary['one_shot_gain_occupancy_weighted']
    m = summary['forward_moments']
    rs = summary['residual_stats']
    fc = summary['foc_consistency']['overall']
    print('\n=== FULL-RATIONALITY CERTIFICATION SUMMARY ===')
    print(f"mass conservation: [{summary['mass_conservation']['min']:.8f}, {summary['mass_conservation']['max']:.8f}]")
    print(f"forward moments:   vacancy={m['vacancy_departing']:.4f} relcost={m['relcost']:.4f} E[p]={m['probability_weighted_price']:.4f}")
    print(f"residuals (mean):  foc={rs['foc']['mean']:.3e} eta_violation={rs['eta_violation']['mean']:.3e} hjb={rs['hjb']['mean']:.3e}")
    print(f"policy eval:       outer={summary['policy_eval']['outer_iters']} final_gap={summary['policy_eval']['final_gap']:.2e}")
    print(f"one-shot gain:     mean={g['mean']:.3e} p99={g['p99']:.3e} max={g['max']:.3e}")
    print(f"weighted gain:     mean={gw['mean']:.3e} p99={gw['p99']:.3e}")
    print(f"FOC residual:      at V_net(occ)={fc['foc_at_Vnet_occ']:.3e}  at V_eval(occ)={fc['foc_at_Veval_occ']:.3e}")
    print(f'wrote {out}')
    print(f'wrote {top_csv}')
if __name__ == '__main__':
    main()

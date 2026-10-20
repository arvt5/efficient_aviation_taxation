from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from aviation_taxation.policy_tables import append_csv, demand_tables_from_policy, stationary_forward_tables, write_json
from aviation_taxation.model import FractionalParams
from aviation_taxation.evaluation import policy_eval_full
from aviation_taxation.diagnostics import _initial_focal_distribution, evaluate_focal_one_shot_gains
from aviation_taxation.best_response import continuous_best_response_full

def _load_params(path: Path) -> FractionalParams:
    data = json.loads(path.read_text())
    return FractionalParams(**data['params'])

def _evaluate(P: np.ndarray, D: np.ndarray, params: FractionalParams, phi_grid: np.ndarray, *, burn_in: int, device: torch.device, coarse_points: int, local_steps: int) -> dict:
    V, pe_history = policy_eval_full(P, D, params, phi_grid, cross_weight=1.0)
    mu, q = stationary_forward_tables(P, D, params, phi_grid, burn_in)
    best, bellman_gain = continuous_best_response_full(P, V, params, phi_grid, device=device, cross_weight=1.0, coarse_points=coarse_points, local_steps=local_steps, threshold_candidates=True, dtype=torch.float64)
    initial_mu = _initial_focal_distribution(mu, params, phi_grid)
    focal, _, _ = evaluate_focal_one_shot_gains(P, D, best, params, phi_grid, initial_mu, device_name=str(device), dtype=torch.float64, top_k=0)
    metric = float(focal['max_profit_gain_scaled_relative_to_baseline'])
    return {'P': P, 'D': D, 'V': V, 'mu': mu, 'q': q, 'best': best, 'bellman_gain': bellman_gain, 'metric': metric, 'max_gain_scaled': float(focal['max_profit_gain_scaled']), 'max_gain_raw': float(focal['max_profit_gain_raw']), 'baseline_profit': float(focal['baseline_focal']['firm_operating_profit_before_fixed']), 'policy_eval_iters': len(pe_history), 'policy_eval_gap': float(pe_history[-1]) if pe_history else None}

def _save_policy(path: Path, state: dict, phi_grid: np.ndarray) -> None:
    np.savez_compressed(path, phi_grid=phi_grid, prices=state['P'], demand=state['D'], values=state['V'], occupancy=state['mu'], sale_mass=state['q'], best_response_prices=state['best'], one_shot_gain=state['bellman_gain'])

def _row(state: dict, *, outer: int, proposal: str, damping: float, accepted: bool, elapsed: float) -> dict:
    return {'outer': outer, 'proposal': proposal, 'damping': damping, 'accepted': int(accepted), 'max_focal_gain_relative': state['metric'], 'max_focal_gain_scaled': state['max_gain_scaled'], 'max_focal_gain_raw': state['max_gain_raw'], 'baseline_operating_profit': state['baseline_profit'], 'policy_eval_iters': state['policy_eval_iters'], 'policy_eval_gap': state['policy_eval_gap'], 'elapsed_seconds': elapsed}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--result-json', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--outer-iters', type=int, default=8)
    ap.add_argument('--burn-in', type=int, default=20)
    ap.add_argument('--dampings', type=float, nargs='+', default=[0.4, 0.2, 0.1, 0.05])
    ap.add_argument('--coarse-points', type=int, default=81)
    ap.add_argument('--local-steps', type=int, default=20)
    ap.add_argument('--min-relative-improvement', type=float, default=1e-05)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--save-accepted-checkpoints', action='store_true', help='Preserve the initial policy and every accepted outer policy for later direct certification.')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = _load_params(args.result_json)
    z = np.load(args.policy)
    phi_grid = np.asarray(z['phi_grid'], dtype=np.float64)
    P = np.asarray(z['prices'], dtype=np.float64).copy()
    device = torch.device(args.device)
    D = demand_tables_from_policy(P, params, phi_grid, device=device)
    start = time.perf_counter()
    settings = {'policy': str(args.policy), 'result_json': str(args.result_json), 'params': asdict(params), 'outer_iters': args.outer_iters, 'burn_in': args.burn_in, 'dampings': args.dampings, 'coarse_points': args.coarse_points, 'local_steps': args.local_steps, 'min_relative_improvement': args.min_relative_improvement, 'device': args.device, 'save_accepted_checkpoints': args.save_accepted_checkpoints}
    write_json(args.out_dir / 'settings.json', settings)
    write_json(args.out_dir / 'status.json', {'state': 'evaluating_initial'})
    current = _evaluate(P, D, params, phi_grid, burn_in=args.burn_in, device=device, coarse_points=args.coarse_points, local_steps=args.local_steps)
    initial_metric = current['metric']
    append_csv(args.out_dir / 'history.csv', _row(current, outer=0, proposal='initial', damping=0.0, accepted=True, elapsed=time.perf_counter() - start))
    _save_policy(args.out_dir / 'accepted_policy.npz', current, phi_grid)
    checkpoint_dir = args.out_dir / 'accepted_checkpoints'
    if args.save_accepted_checkpoints:
        checkpoint_dir.mkdir(exist_ok=True)
        _save_policy(checkpoint_dir / 'outer_00_policy.npz', current, phi_grid)
    print(f'initial max_full_horizon_gain={initial_metric:.9g}', flush=True)
    modes: list[tuple[str, int | None]] = [('all', None), ('rank0', 0), ('rank1', 1), ('rank2', 2)]
    accepted_steps = 0
    for outer in range(1, args.outer_iters + 1):
        best_candidate = None
        best_label = ''
        best_damping = 0.0
        for label, rank in modes:
            for damping in args.dampings:
                proposal = current['P'].copy()
                if rank is None:
                    proposal[:-1] = (1.0 - damping) * proposal[:-1] + damping * current['best'][:-1]
                else:
                    proposal[:-1, ..., rank] = (1.0 - damping) * proposal[:-1, ..., rank] + damping * current['best'][:-1, ..., rank]
                proposal_D = demand_tables_from_policy(proposal, params, phi_grid, device=device)
                candidate = _evaluate(proposal, proposal_D, params, phi_grid, burn_in=args.burn_in, device=device, coarse_points=args.coarse_points, local_steps=args.local_steps)
                append_csv(args.out_dir / 'history.csv', _row(candidate, outer=outer, proposal=label, damping=damping, accepted=False, elapsed=time.perf_counter() - start))
                print(f'outer={outer} proposal={label} damping={damping:g} max_full_horizon_gain={candidate['metric']:.9g}', flush=True)
                if best_candidate is None or candidate['metric'] < best_candidate['metric']:
                    best_candidate = candidate
                    best_label = label
                    best_damping = damping
        required = current['metric'] * (1.0 - args.min_relative_improvement)
        if best_candidate is None or best_candidate['metric'] >= required:
            print(f'stop: no accepted proposal; current={current['metric']:.9g} best={(best_candidate or current)['metric']:.9g}', flush=True)
            break
        current = best_candidate
        accepted_steps += 1
        accepted_row = _row(current, outer=outer, proposal=best_label, damping=best_damping, accepted=True, elapsed=time.perf_counter() - start)
        append_csv(args.out_dir / 'accepted_history.csv', accepted_row)
        _save_policy(args.out_dir / 'accepted_policy.npz', current, phi_grid)
        if args.save_accepted_checkpoints:
            _save_policy(checkpoint_dir / f'outer_{outer:02d}_policy.npz', current, phi_grid)
        write_json(args.out_dir / 'status.json', {'state': 'running', 'latest': accepted_row})
        print(f'accepted outer={outer} proposal={best_label} damping={best_damping:g} max_full_horizon_gain={current['metric']:.9g}', flush=True)
    result = {**settings, 'initial_max_focal_gain_relative': initial_metric, 'final_max_focal_gain_relative': current['metric'], 'relative_reduction': 1.0 - current['metric'] / initial_metric, 'accepted_steps': accepted_steps, 'elapsed_seconds': time.perf_counter() - start, 'policy_artifact': str(args.out_dir / 'accepted_policy.npz')}
    write_json(args.out_dir / 'result.json', result)
    write_json(args.out_dir / 'status.json', {'state': 'done', 'result': result})
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()

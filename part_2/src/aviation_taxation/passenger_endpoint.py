"""Solve/evaluate the passenger endpoint at fixed parameters, without calibration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from .accounting import endpoint_metrics
from .reproduce import ROOT, read_json


def validate_parameters(actual: dict, expected: dict):
    for key, value in expected.items():
        if key not in actual or not math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'Checkpoint parameter {key} does not match the passenger endpoint')


def summarize(moments: dict, params: dict, eu_revenue_beur: float = 50) -> dict:
    if not math.isfinite(eu_revenue_beur) or eu_revenue_beur <= 0:
        raise ValueError('EU revenue base must be finite and positive')
    config = read_json(ROOT / 'config/paper.json')
    baseline = read_json(ROOT / 'data/archived/status_quo.json')
    reference = read_json(ROOT / 'data/archived/passenger_neutral.json')
    fixed = baseline['fixed_cost']
    record = {**moments, 'M': params['T'] / params['d'],
              'd': params['d'], 'fixed_cost': fixed,
              'implied_tau_c_model_per_flight': moments['firm_operating_profit_before_fixed'] - fixed}
    metrics = endpoint_metrics(record, baseline, config, carbon=True)
    expected = endpoint_metrics(reference, baseline, config, carbon=True)
    conversion = config['status_quo_gross_revenue_busd'] / baseline['calendar_revenue']
    savings = fixed * (1 - record['M'] / baseline['M']) * conversion
    passenger_change = 100 * (metrics['passengers_bn'] / config['status_quo_passengers_bn'] - 1)
    scale = eu_revenue_beur / config['status_quo_gross_revenue_busd']
    return {
        'params': params,
        'accounting': record,
        'global_results': metrics,
        'difference_from_selected_archive': {key: value - expected[key] for key, value in metrics.items()},
        'policy_conclusion': {
            'passenger_change_percent': passenger_change,
            'welfare_gain_positive': metrics['welfare_gain_busd'] > 0,
            'net_fiscal_dividend_positive': metrics['fiscal_dividend_busd'] > 0,
            'flight_frequency_reduced': record['M'] < baseline['M'],
        },
        'eu_results': {
            'status_quo_revenue_beur': eu_revenue_beur,
            'fiscal_dividend_beur': metrics['fiscal_dividend_busd'] * scale,
            'welfare_gain_beur': metrics['welfare_gain_busd'] * scale,
            'welfare_including_wedge_beur': (metrics['welfare_gain_busd'] + config['fiscal_wedge'] * savings) * scale,
        },
        'method': 'Fixed passenger-endpoint parameters; stationary policy evaluation. Baseline and monetary scale are archived, not re-solved. Carbon charge is derived from zero profit, not fitted to the reported tax. No interpolation, recalibration, or VAT-equivalence calculation.',
        'equilibrium_quality': 'Economic accounting alone is not equilibrium certification; use the evaluation/diagnostics modules for deviation checks.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--train', action='store_true', help='Explicitly launch the expensive endpoint training run')
    source.add_argument('--checkpoint', type=Path, help='Evaluate an existing neural checkpoint instead of training')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, help='Training seed override; only valid with --train')
    parser.add_argument('--eu-revenue-beur', type=float, default=50)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/passenger_endpoint')
    args = parser.parse_args()
    if args.seed is not None and (not args.train or args.seed < 0):
        parser.error('--seed requires --train and a nonnegative value')
    if not math.isfinite(args.eu_revenue_beur) or args.eu_revenue_beur <= 0:
        parser.error('EU revenue base must be finite and positive')
    settings = read_json(ROOT / 'config/passenger_training.json')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if args.train:
        checkpoint = args.output / 'network.pt'
        if checkpoint.exists():
            parser.error(f'Refusing to overwrite {checkpoint}; use --checkpoint or a new output directory')
        command = [sys.executable, '-m', 'aviation_taxation.train',
                   '--config', str(ROOT / 'config/passenger_training.json'),
                   '--device', args.device, '--output', str(checkpoint)]
        if args.seed is not None:
            command.extend(['--seed', str(args.seed)])
        subprocess.run(command, check=True)

    from .tables import load_model
    from .forward import stationary_forward, welfare_moments

    net, params = load_model(checkpoint)
    validate_parameters(asdict(params), settings['params'])
    grid = np.linspace(0, 1, settings['n_phi'])
    mu, q, prices, _ = stationary_forward(net, params, grid, burn_in=settings['burn_in'])
    moments = welfare_moments(mu, q, prices, params)
    report = summarize(moments, asdict(params), args.eu_revenue_beur)
    report['checkpoint'] = str(checkpoint.resolve())
    report['evaluation'] = {'n_phi': settings['n_phi'], 'burn_in': settings['burn_in'],
                            'mass_min': float(mu.sum(axis=(1, 2, 3)).min()),
                            'mass_max': float(mu.sum(axis=(1, 2, 3)).max()),
                            'sales_identity_gap': moments['sales_identity_gap']}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'policy_conclusion': report['policy_conclusion'],
                      'global_results': report['global_results'],
                      'eu_results': report['eu_results']}, indent=2))
    print(f'Report: {args.output / "results.json"}')


if __name__ == '__main__':
    main()

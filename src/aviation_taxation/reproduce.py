"""Regenerate published accounting and audit it against displayed paper values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .accounting import endpoint_metrics
from .interpolation import bracketed_root, fit_candidates


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def reproduce(output: Path) -> dict:
    data = ROOT / 'data/archived'
    config = read_json(ROOT / 'config/paper.json')
    baseline = read_json(data / 'status_quo.json')
    rows, checks = [], []
    for name, expected in config['published_endpoints'].items():
        record = read_json(data / (name + '.json'))
        metrics = endpoint_metrics(record, baseline, config, carbon=name == 'passenger_neutral')
        rows.append({'endpoint': name, **metrics})
        for column, value in zip(config['published_endpoint_columns'], expected):
            decimals = 2 if column in ('M', 'passengers_bn') else 1
            actual = round(metrics[column], decimals)
            checks.append({'endpoint': name, 'metric': column, 'expected': value,
                           'actual': metrics[column], 'passed': actual == value})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'policy_endpoints.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {'status_quo': 'Status quo', 'vacancy_optimum': 'Vacancy-tax optimum',
              'free_entry': r'Free entry $M^\star$', 'passenger_neutral': 'Passenger-neutral'}
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             r'Endpoint & $M$ & LF [\%] & Pax [bn] & Net rev. & Welfare gain & Carbon tax & Carbon rev. & Fiscal dividend \\',
             r'\midrule']
    for row in rows:
        values = [f"{row[k]:.{2 if k in ('M', 'passengers_bn') else 1}f}"
                  for k in config['published_endpoint_columns']]
        lines.append(labels[row['endpoint']] + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'policy_endpoints.tex').write_text('\n'.join(lines) + '\n')

    diagnostics = []
    for name, expected in config['published_diagnostics'].items():
        archived = read_json(data / f'diagnostic_{name}.json')
        direct = archived['direct_forward_optimization_top']
        actual = [direct['max_direct_profit_gain_scaled'],
                  direct['max_direct_gain_scaled_relative_to_baseline']]
        for key, observed, wanted in zip(('gain', 'relative_gain'), actual, expected):
            checks.append({'endpoint': name, 'metric': key, 'actual': observed,
                           'expected': wanted, 'passed': abs(observed - wanted) <= 5e-9})
        diagnostics.append({'endpoint': name, 'gain': actual[0], 'relative_gain': actual[1],
                            'source': 'archived direct diagnostic; not recomputed'})
    with (output / 'equilibrium_quality.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Endpoint & One-shot gain & Relative one-shot gain \\', r'\midrule']
    for row in diagnostics:
        lines.append(labels[row['endpoint']] + f" & {row['gain']:.6e} & {row['relative_gain']:.6e}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'equilibrium_quality.tex').write_text('\n'.join(lines) + '\n')

    with (data / 'frequency_sweep.csv').open(newline='') as handle:
        sweep = sorted(csv.DictReader(handle), key=lambda row: float(row['M']))
    x = [float(row['M']) for row in sweep]
    fits = {metric: fit_candidates(x, [float(row[metric]) for row in sweep])
            for metric in ('implied_tau_c_model_per_flight', 'passenger_rate')}
    endpoints = read_json(data / 'interpolation_endpoints.json')
    roots = {}
    for name, metric, target in (
        ('free_entry', 'implied_tau_c_model_per_flight', 0.0),
        ('passenger_neutral', 'passenger_rate', baseline['sales'] * baseline['M'] / config['selling_horizon_model']),
    ):
        roots[name] = bracketed_root(fits[metric], target, endpoints[name]['observed_bracket_M'])
        checks.append({'endpoint': name, 'metric': 'interpolated_M', 'actual': roots[name],
                       'expected': endpoints[name]['M'],
                       'passed': abs(roots[name] - endpoints[name]['M']) < 1e-9})
    (output / 'interpolation.json').write_text(json.dumps({'fits': fits, 'roots': roots}, indent=2) + '\n')
    audit = {'paper_commit': config['paper_commit'], 'checks': checks,
             'passed': all(check['passed'] for check in checks),
             'diagnostics_recomputed': False,
             'policy_pairing_note': 'Free-entry accounting and diagnostic refer to different archived policy versions; see provenance/OPEN_ISSUES.md.'}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'passed': audit['passed'], 'checks': len(checks), 'output': str(output)}))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    audit = reproduce(args.output)
    if not audit['passed']:
        raise SystemExit('Some published results did not reproduce; inspect audit.json')


if __name__ == '__main__':
    main()

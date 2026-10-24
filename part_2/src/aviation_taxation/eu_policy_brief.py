"""Rebase global passenger-endpoint accounting to intra-EU aviation revenue."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .accounting import endpoint_metrics
from .reproduce import ROOT, read_json


def calculate(eu_revenue_beur: float = 50.0) -> dict:
    if not math.isfinite(eu_revenue_beur) or eu_revenue_beur <= 0:
        raise ValueError('The intra-EU status quo revenue base must be finite and positive')
    config = read_json(ROOT / 'config/paper.json')
    data = ROOT / 'data/archived'
    baseline = read_json(data / 'status_quo.json')
    global_base = config['status_quo_gross_revenue_busd']
    scale = eu_revenue_beur / global_base
    conversion = global_base / baseline['calendar_revenue']
    rows, policies = [], {}
    for policy, filename in (
        ('refined', 'passenger_neutral_refined.json'),
        ('unrefined', 'passenger_neutral.json'),
    ):
        record = read_json(data / filename)
        metrics = endpoint_metrics(record, baseline, config, carbon=True)
        vacancy_tax = record['params']['tau'] if 'params' in record else record['tau']
        reduction = 1 - record['M'] / baseline['M']
        savings = baseline['fixed_cost'] * reduction * conversion
        values = {
            'Fiscal dividend': metrics['fiscal_dividend_busd'],
            'Welfare gain': metrics['welfare_gain_busd'],
            'Fixed-cost savings': savings,
            'Welfare including wedge': metrics['welfare_gain_busd'] + config['fiscal_wedge'] * savings,
            'Additional consumption-tax revenue': .115 * savings,
            'Additional corporate-tax revenue': .26 * .145 * savings,
        }
        policies[policy] = {
            'M': record['M'],
            'vacancy_percent': metrics['vacancy_percent'],
            'frequency_reduction_percent': 100 * reduction,
            'passenger_change_percent': 100 * (metrics['passengers_bn'] / config['status_quo_passengers_bn'] - 1),
            'vacancy_tax_percent_of_status_quo_price': 100 * vacancy_tax / baseline['probability_weighted_price'],
        }
        for metric, value in values.items():
            if policy == 'refined':
                rows.append({'metric': metric, 'refined_global_busd': value,
                             'refined_eu_beur': value * scale})
            else:
                row = next(row for row in rows if row['metric'] == metric)
                row.update(unrefined_global_busd=value, unrefined_eu_beur=value * scale)
    return {
        'assumptions': {
            'eu_status_quo_revenue_beur': eu_revenue_beur,
            'global_status_quo_revenue_busd': global_base,
            'effective_scale': scale,
            'basis': 'Revenue rebasing for flights starting and ending within the EU; not GDP weighting or an exchange rate.',
            'base_provenance': 'EUR 50bn is inferred from the brief, not independently estimated here; its EUR 45bn VAT base is distinct.',
            'fiscal_wedge': config['fiscal_wedge'],
            'consumption_tax_rate': .115,
            'corporate_tax_rate': .26,
            'markup_rate': .145,
            'scope': 'Annual monetary accounting for the archived global vacancy-tax/carbon-charge package; no EU equilibrium is re-solved.',
            'exclusions': 'No derivation of the 3.7% VAT, oil benefits, or climate benefits; no claim that VAT implements the same allocation.',
        },
        'policies': policies,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eu-revenue-beur', type=float, default=50.0,
                        help='Status quo intra-EU gross aviation revenue in billion EUR (default: 50)')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/eu_policy_brief')
    args = parser.parse_args()
    try:
        result = calculate(args.eu_revenue_beur)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    columns = ['metric', 'refined_global_busd', 'unrefined_global_busd',
               'refined_eu_beur', 'unrefined_eu_beur']
    with (args.output / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result['rows'])
    print(f"Revenue rebasing factor: {result['assumptions']['effective_scale']:.3f}; not GDP weighting or an FX rate.")
    print('Metric | Global refined / unrefined [bn USD] | EU refined / unrefined [bn EUR]')
    for row in result['rows']:
        print(f"{row['metric']} | {row['refined_global_busd']:.3f} / {row['unrefined_global_busd']:.3f}"
              f" | {row['refined_eu_beur']:.3f} / {row['unrefined_eu_beur']:.3f}")
    print('Passenger changes: ' + ', '.join(
        f"{name} {policy['passenger_change_percent']:+.3f}%" for name, policy in result['policies'].items()))
    print(f'Outputs: {args.output}')


if __name__ == '__main__':
    main()

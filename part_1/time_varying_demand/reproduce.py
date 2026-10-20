"""Recompute welfare-gap closures and conditional posted prices from archives."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from part_1.figures import save

HERE = Path(__file__).resolve().parent


def posted_prices(out):
    prices = out['prices']
    probability = out['prob'][:, :prices.shape[1]]
    active = np.isfinite(prices)
    weights = np.sum(probability * active, axis=0)
    numerator = np.sum(probability * np.where(active, prices, 0), axis=0)
    return np.divide(numerator, weights, out=np.full_like(weights, np.nan), where=weights >= 1e-12)


def gap_closed(data):
    baseline = data['out_base_mu']['total_welfare']
    return 100 * ((data['out_opt']['total_welfare'] - baseline)
                  / (data['out_plan']['total_welfare'] - baseline))


def main():
    out = HERE / 'outputs'
    out.mkdir(exist_ok=True)
    results = []
    for model in ('exponential', 'ces'):
        data = loadmat(HERE / 'data' / f'{model}.mat', simplify_cells=True)
        results.append({'model': model, 'gap_closed_pct': gap_closed(data),
                        'buyback_share': float(data['xi_opt']),
                        'vacancy_tax_model': float(data['tau_opt'])})
        if model != 'ces':
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=.17, right=.96, top=.97, bottom=.23)
        rows = []
        for key, label, color, style in (
            ('out_base_mu', 'Baseline', '#1F3A5F', '-'),
            ('out_plan', 'Planner', '#687846', ':'),
            ('out_opt', 'Buyback + vacancy tax', '#B65E3C', '--'),
        ):
            p = posted_prices(data[key])
            time = np.linspace(0, 1, len(p))
            ax.plot(time, p, label=label, color=color, linestyle=style, linewidth=1.8)
            rows.extend(dict(policy=label, time=float(t), price=float(v)) for t, v in zip(time, p))
        ax.set(xlabel='Sales horizon', ylabel='Posted price', xlim=(0, 1),
               xticks=[0, .5, 1], xticklabels=['0', '$T/2$', '$T$'])
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False,
                  fontsize=8, ncol=3, columnspacing=1, handlelength=2)
        save(fig, out, 'ces_tv_three_price_paths_return_overlay')
        with (out / 'ces_price_paths.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['policy', 'time', 'price'])
            writer.writeheader()
            writer.writerows(rows)
    with (out / 'refund_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('Mandatory buybacks:', results)


if __name__ == '__main__':
    main()

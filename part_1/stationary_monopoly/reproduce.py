"""Exponential-demand figures from the paper's finite-sum load-factor mapping."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from part_1.figures import save

HERE = Path(__file__).resolve().parent


def load_factor(n, x):
    if n < 1 or int(n) != n or x < 0:
        raise ValueError('Capacity must be a positive integer and x nonnegative')
    if x == 0:
        return 0.0
    k = np.arange(n + 1)
    terms = k * np.log(x) - gammaln(k + 1)
    return float(x / n * np.exp(logsumexp(terms[:-1]) - logsumexp(terms)))


def optimal_load_factor(n, observed):
    if not 0 <= observed <= 1:
        raise ValueError('Load factor must lie in [0, 1]')
    if observed in (0, 1):
        return float(observed)
    upper = float(n)
    while load_factor(n, upper) < observed:
        upper *= 2
    x = brentq(lambda x: load_factor(n, x) - observed, 0, upper)
    return load_factor(n, np.e * x)


def main():
    out = HERE / 'outputs'
    capacity = np.arange(1, 201)
    optimal = np.array([optimal_load_factor(int(n), .8) for n in capacity])
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.plot(capacity, 100 * optimal, color='#1F3A5F')
    ax.set(xlabel='Number of seats $N$', ylabel='Optimal load factor [%]',
           xlim=(0, 200), ylim=(90, 100))
    save(fig, out, 'optimal_vacancy_vs_seats')
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    inset = ax.inset_axes([.15, .52, .46, .38])
    vacancy = np.linspace(0, 100, 201)
    curves = []
    for n, color in zip((20, 50, 100), ('#1F3A5F', '#B65E3C', '#687846')):
        vr = np.array([100 * (1 - optimal_load_factor(n, 1 - v / 100)) for v in vacancy])
        ax.plot(vacancy, vr, label=f'$N={n}$', color=color)
        inset.plot(vacancy, vr, color=color)
        curves.append(vr)
    inset.set(xlim=(0, 40), ylim=(0, 10), xticks=(0, 20, 40), yticks=(0, 5, 10))
    ax.set(xlabel='Observed vacancy rate [%]', ylabel='Optimal vacancy rate [%]',
           xlim=(0, 100), ylim=(0, 100))
    ax.legend(frameon=False, loc='lower right')
    save(fig, out, 'optimal_vs_observed_vacancy_2050100')
    np.savetxt(out / 'load_factor_vs_seats.csv', np.column_stack([capacity, optimal]),
               delimiter=',', header='seats,optimal_load_factor', comments='')
    np.savetxt(out / 'optimal_vs_observed_vacancy.csv', np.column_stack([vacancy, *curves]),
               delimiter=',', header='observed_vacancy_pct,N20,N50,N100', comments='')
    result = {'N50_optimal_vacancy_pct': 100 * (1 - optimal_load_factor(50, .8))}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Stationary exponential demand:', result)


if __name__ == '__main__':
    main()

"""Full-information ticket-tax replacement, at the paper's central calibration."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from part_1.figures import save

HERE = Path(__file__).resolve().parent


def load(model):
    with (HERE / 'data' / f'{model}.csv').open() as f:
        return list(csv.DictReader(f))


def metrics(rows):
    baseline = next(r for r in rows if r['policy_case'] == 'current_ticket_tax')
    optimum = next(r for r in rows if r['policy_case'] == 'optimal_vacancy_tax')
    price = float(baseline['av_price'])
    gain = float(optimum['total_welfare']) - float(baseline['total_welfare'])
    planner = float(optimum['planner_welfare'])
    return {
        'Status quo vacancy [%]': float(baseline['vacancy_rate']),
        'Optimal vacancy [%]': float(optimum['vacancy_rate']),
        'Gap closed [%]': 100 * gain / (planner - float(baseline['total_welfare'])),
        'Vacancy tax [USD]': 194 * float(optimum['tau']) / price,
        'Vacancy tax / status quo price [%]': 100 * float(optimum['tau']) / price,
        'Welfare gain [bn USD]': 1000 * gain / float(baseline['revenue']),
    }


def main():
    out = HERE / 'outputs'
    out.mkdir(exist_ok=True)
    fig, left = plt.subplots(figsize=(5.6, 5.6))
    fig.subplots_adjust(left=.15, right=.85, bottom=.25, top=.94)
    right = left.twinx()
    right.spines['right'].set_visible(True)
    summaries = {}
    for model, label, color in [('exponential', 'Exponential', '#B65E3C'),
                                ('logit', 'Logit', '#1F3A5F')]:
        rows = load(model)
        summaries[label] = metrics(rows)
        base = next(r for r in rows if r['policy_case'] == 'current_ticket_tax')
        sweep = sorted((r for r in rows if r['policy_case'] == 'vacancy_tax_sweep'),
                       key=lambda r: float(r['tau']))
        x = [100 * float(r['tau']) / float(base['av_price']) for r in sweep]
        vacancy = [float(r['vacancy_rate']) for r in sweep]
        welfare = [1000 * (float(r['total_welfare']) - float(base['total_welfare']))
                   / float(base['revenue']) for r in sweep]
        left.plot(x, vacancy, color=color, linestyle='--', linewidth=1.6)
        right.plot(x, welfare, color=color, linewidth=1.8, label=label)
        maximum = summaries[label]['Vacancy tax / status quo price [%]']
        left.axvline(maximum, color=color, linestyle=':', linewidth=.8)
        left.text(maximum, 21.6 if model == 'exponential' else 20.8,
                  f'{label}: {maximum:.0f}%', ha='center', fontsize=8)
    left.set(xlabel=r'Vacancy tax $\tau / P_{sq}$ [%]', ylabel='Vacancy rate [%]',
             xlim=(0, 120), ylim=(0, 22))
    right.set(ylabel='Welfare gain [bn USD]', ylim=(0, 82))
    right.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2, frameon=False)
    left.text(.5, -.29, 'Dashed: vacancy rate     Solid: welfare gain',
              ha='center', transform=left.transAxes, fontsize=8)
    save(fig, out, 'duopoly_ticket_tax_welfare_gain_vs_tau_over_p_sq_vr20_dr30')
    with (out / 'duopoly_results.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', *summaries])
        for metric in next(iter(summaries.values())):
            writer.writerow([metric, *[values[metric] for values in summaries.values()]])
    lines = [r'\begin{tabular}{lrr}', r'\hline', r'Metric & Exponential & Logit \\', r'\hline']
    for metric in next(iter(summaries.values())):
        label = metric.replace('%', r'\%')
        values = ' & '.join(f'{m[metric]:.1f}' for m in summaries.values())
        lines.append(f'{label} & {values} ' + r'\\')
    lines.extend([r'\hline', r'\end{tabular}'])
    (out / 'duopoly_results.tex').write_text('\n'.join(lines) + '\n')
    print('Duopoly:', summaries)


if __name__ == '__main__':
    main()

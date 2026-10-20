# Stationary Monopoly

From the repository root:

```bash
python -m part_1.stationary_monopoly.reproduce
```

This regenerates `optimal_vacancy_vs_seats.pdf` and
`optimal_vs_observed_vacancy_2050100.pdf`, plus the underlying CSVs. It implements
the finite-sum load-factor mapping in the paper directly, using logarithms to
avoid overflow. At observed vacancy 20%, capacity 50 implies optimal vacancy
1.5417%. The figures have been redrawn with Matplotlib; their original plotting
script has not been recovered.

## Constant-Elasticity Demand

The recovered MATLAB Bellman and planner solvers are in `matlab/`. In MATLAB,
change to that directory and run:

```matlab
results = recompute_ces(100, 20000);
```

The arguments explicitly specify seat capacity and time steps. Calibration
matches vacancy and relative marginal cost of 20%. The optimal vacancy tax is
`c / elasticity`; outputs include the planner benchmark and welfare gains with
both welfare and revenue denominators. This is a new calculation using the
recovered method, not a recovered published run.

**Verification still needed:** the original calibration behind the paper's
approximate CES vacancy of 3% and welfare gain of 3.2% has not been identified.
The available 10-seat MATLAB experiments and older 50-seat Python experiment
cannot be substituted without changing those results. Neither is shipped as
evidence for these claims. No paper numbers have been changed.

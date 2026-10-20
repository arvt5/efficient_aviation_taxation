# Fixed-Flight Duopoly

```bash
python -m part_1.duopoly.reproduce
```

Run from the repository root. Outputs are the square dual-axis vacancy/welfare
figure and the transposed results table (CSV and LaTeX). The figure uses the
archived sweep directly, without fitting or changing its optimum. Matplotlib
replaces the old Plotly renderer; the numerical series are unchanged.

Only the paper's full-information central calibration is distributed:
10 seats per flight, 500 periods, status quo vacancy 20%, relative marginal
cost 20%, diversion ratio 30%, and ticket-tax burden 6.5% of the ticket price.
The two small CSVs contain 31 sweep points and three reference policies each.
Other information regimes and calibration cells are excluded.

The welfare gain is `(W_opt - W_ticket) / revenue_ticket * 1000` billion USD.
The vacancy tax in USD is `tau / price_ticket * 194`. Gap closed compares the
same welfare gain with the difference between planner and ticket-tax welfare.
Raw archived column names are preserved; output tables use plain English.

## Numerical Recomputation

In MATLAB R2025a with Optimization Toolbox, change to `matlab/`:

```matlab
recompute_duopoly('exponential');
recompute_duopoly('logit');
```

These commands solve the central calibration, planner, vacancy-tax sweep, and
local welfare maximum. They explicitly select full information. The saved
calibration is used by default; passing `true` also recalibrates its three
parameters to the moments above. New results go to `outputs/`. Without
recalibration, the baseline and every sweep welfare value are checked against
the archives. No alternative information-regime experiment is run. Historical
branches inside the recovered numerical kernels are retained to avoid changing
their arithmetic during extraction.

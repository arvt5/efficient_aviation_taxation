# Time-Varying Demand and Buybacks

```bash
python -m part_1.time_varying_demand.reproduce
```

Run from the repository root. This writes the CES three-price-path figure,
its underlying conditional posted prices, and the refund results to `outputs/`.
The exponential and CES joint reforms close 79.0514% and 82.3437% of the welfare
gap to their respective planners, matching the paper's 79.1% and 82.3%.

`data/` contains the selected calibration and policy outputs from the original
MATLAB archives. Unrelated workspace variables were removed. The price average
uses state occupancy probabilities, conditional on positive inventory; sold-out
prices are not treated as zeros. The gap denominator compares the planner and
baseline **with the same return hazard**, not the no-return calibration.

## Numerical Recomputation

In MATLAB R2025a with Optimization Toolbox, change to `matlab/`:

```matlab
recompute_refunds('ces');
recompute_refunds('exponential');
```

This solves the Bellman equations using the saved calibration and policy
parameters, then compares all five welfare outcomes with their archives.
It does not fit parameters or optimize instruments. For a fresh calibration
and policy optimization, pass `true` as the second argument. Optimization
starts at the archived solution; differences can depend on numerical tolerances.
All newly computed results are written separately under `outputs/`.

Both archived demand models use 10 seats, 1,000 periods, and cumulative return
hazard `-log(0.8)`. The paper's CES figure caption says 500 periods, but the
archived price arrays have 1,000 columns. Reproduction uses the archived
resolution, not the stale script default. The no-return calibration targets vacancy 20%, relative marginal
cost 20%, and a late/early posted-price ratio of two. The same solver evaluates
time-varying demand without returns by setting the hazard to zero. The joint
reform additionally chooses the buyback share and vacancy tax.

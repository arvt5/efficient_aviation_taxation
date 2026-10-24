# Efficient Aviation Taxation

Code and numerical inputs organized in the paper's reading order.
Reference: revision `06505f302b69eb47558e9c27a021c214b0ee61bc` of
[the paper](https://github.com/LennartStern/Efficient-aviation-tax-policy-with-endogenous-vacancy-rates).

## Structure and Coverage

| Folder | Paper results | Verified reproduction |
| --- | --- | --- |
| `part_1/stationary_monopoly` | Exponential load-factor figures | Direct analytical calculation |
| `part_1/stationary_monopoly/matlab` | Constant-elasticity calibration | Solver recovered; published calibration not identified |
| `part_1/time_varying_demand` | Buyback welfare gains and CES price paths | Archived results and re-solved MATLAB policies |
| `part_1/duopoly` | Full-information exponential/logit table and tax-sweep figure | Archived results and re-solved MATLAB baselines |
| `part_2` | Free-entry tables, diagnostics, price paths and welfare figure | Archived inputs; numerical solver smoke-tested |

This is **not yet a complete from-scratch reproduction of every paper result**.
Regenerating outputs from archived inputs is distinct from solving the model
again. No generated outputs, environments, deployment files or cluster job logs
are tracked. Only central paper experiments are included.

## Reproduce Tables and Figures

Python 3.12 is the tested version. Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e part_2
make reproduce PYTHON="$PWD/.venv/bin/python"
make test PYTHON="$PWD/.venv/bin/python"
```

These commands check inputs and regenerate outputs without recalibration or
network training. Results go into ignored `outputs/` directories beside their
experiments. Part II price-path PDF/PNG export requires `rsvg-convert` from
librsvg; SVG export does not. Other PDFs use Matplotlib. Part I plots have a new
renderer, with unchanged numerical series or the paper's analytical mapping.

For an individual Part I experiment, run
`python -m part_1.<folder>.reproduce`. Part II commands are
`python -m aviation_taxation.reproduce`, `python -m aviation_taxation.welfare_plot`
and `python -m aviation_taxation.price_figures`.

Part I exports the four figure assets, their underlying series and result
tables. Part II exports endpoint and diagnostic tables, interpolation fits,
price-path and welfare figures, and `audit.json` with 42 published-value checks.

## Intra-EU Policy Brief

`python -m aviation_taxation.eu_policy_brief` reproduces the monetary comparison
between refined and unrefined passenger-endpoint policies. It writes CSV and
JSON under `part_2/outputs/eu_policy_brief/`, including assumptions and passenger
deviations. It is also included in `make reproduce`.

The default rebases global status quo aviation revenue of USD 1,000bn to an
intra-EU revenue base of EUR 50bn: numerically a factor of 0.05, not GDP weighting
or currency conversion. The EUR 50bn base is inferred from the brief; it is not
independently estimated here and is distinct from its EUR 45bn VAT base.
Override it with `--eu-revenue-beur`. The script reports fiscal dividend,
welfare, fixed-cost savings, the 0.28 wedge correction and cross-sector fiscal
spillovers. It does not derive the brief's 3.7% VAT, oil or climate benefits,
or establish equivalence between VAT and the model's carbon instrument.

## Solve the Models

Part I requires MATLAB R2025a with Optimization Toolbox. From the repo root:

```matlab
addpath('part_1/tests'); verify_matlab;
```

This checks CES tax/planner alignment, archived refund policies, and both
duopoly baselines without recalibration. Experiment entry points:

```matlab
addpath('part_1/time_varying_demand/matlab');
recompute_refunds('ces');
recompute_refunds('exponential');
addpath('part_1/duopoly/matlab');
recompute_duopoly('exponential');
recompute_duopoly('logit');
```

Refund commands evaluate saved calibration and policy parameters. Duopoly
commands recompute the planner, tax sweep and local welfare maximum at saved
calibration parameters. Passing `true` as the second argument also recalibrates.
New results never overwrite archived inputs. For stationary CES, add its
`matlab/` folder and call `recompute_ces(N,T)` with explicit capacity and time
resolution; this does not reproduce an identified published calibration.

Part II numerical routines are in `part_2/src/aviation_taxation/`. Install
PyTorch support with `python -m pip install -e 'part_2[solver]'`; command-line
modules provide `--help`. The supplied training configuration is
`part_2/config/passenger_training.json`. The original passenger job requested
one GPU, 32 CPU cores, 192 GB RAM and 24 hours. Full retraining has not been
performed during migration and is not part of `make reproduce`.

To reproduce the passenger endpoint from its given parameters (not recalibrate
to its outcomes), run `python -m aviation_taxation.passenger_endpoint --train
--device cuda`. This explicitly launches the expensive training run, evaluates
the resulting policy, derives the carbon charge from zero profit, and reports
passenger deviations, welfare, net fiscal dividend and the EU revenue rebasing.
For an existing neural policy, use `--checkpoint /path/to/network.pt` instead
of `--train`. Results go to `part_2/outputs/passenger_endpoint/results.json`.
The archived baseline is held fixed; the report compares outcomes with the
selected unrefined endpoint, without forcing them to match. This workflow does
not perform tabular policy refinement or new one-shot certification.
Use `--seed` with `--train` for independent replications at identical parameters.

## Accounting and Remaining Limits

- The stationary CES calibration behind approximately 3% vacancy and 3.2%
  welfare gain remains unidentified. Other runs are not substituted.
- Refund archives use 10 seats and 1,000 periods; the paper caption and legacy
  script default say 500. Reproduction uses the actual archived resolution.
  Welfare-gap closure compares baseline and planner with the same return hazard.
- Duopoly inputs use full information, 10 seats per firm, 500 periods, vacancy
  20%, relative marginal cost 20%, diversion 30% and ticket-tax burden 6.5%.
  Dollar conversion uses status quo revenue of USD 1,000 billion and mean price
  USD 194. Price paths condition occupancy weights on positive inventory.
- Part II consistently uses the initial certified passenger policy; its actual
  passenger shortfall is 0.67%. Frequency and interpolation inputs are unchanged.
  Free-entry accounting and its diagnostic still use different archived policies.
- Part II diagnostics are archived, not new certifications. Complete selected
  checkpoints and calibration/sweep orchestration are not yet distributed.
  Validation elasticity and the empirical schedule-delay appendix have not yet
  received a complete claim-level reproduction check.
- The equilibrium implementation covers `2 < M <= 3`; the welfare figure beyond
  `M=3` uses quadratic extrapolation, not an equilibrium with additional firms.

## Verification and Origins

The Python suite has 15 tests; two network-specific tests are skipped without
optional PyTorch. MATLAB verification is separate. Full recalibration and tax
sweeps have not all been re-run. Tested versions: NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.10.9 and optional PyTorch 2.12.0. Historical cluster guidance lists
Python 3.9.21, NumPy 2.0.2 and PyTorch 2.8.0+cu128; the complete run-specific
environment freeze has not been recovered.

Source hashes are in `part_1/sources.json` and
`part_2/provenance/sources.json`. Input checksums are in
`part_1/data_checksums.json` and `part_2/provenance/files.json`.
Original paths in metadata identify origins, not runtime dependencies.

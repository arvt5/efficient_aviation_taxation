# Efficient Aviation Taxation

Code and numerical inputs for the current paper, organized in its reading order.

| Folder | Paper content | Method |
| --- | --- | --- |
| [part_1/stationary_monopoly](part_1/stationary_monopoly) | Stationary exponential and constant-elasticity demand | Closed-form mapping; backward induction |
| [part_1/time_varying_demand](part_1/time_varying_demand) | Time-varying demand and mandatory buybacks | Backward induction |
| [part_1/duopoly](part_1/duopoly) | Exponential and logit demand with fixed flights | Full-information backward induction |
| [part_2](part_2) | Dynamic competition with endogenous frequency | Deep equilibrium network |

Reference: paper revision `06505f302b69eb47558e9c27a021c214b0ee61bc` in
[the paper repository](https://github.com/LennartStern/Efficient-aviation-tax-policy-with-endogenous-vacancy-rates).

## Quick reproduction

Python 3.12 is the tested version. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e part_2
make reproduce PYTHON="$PWD/.venv/bin/python"
make test PYTHON="$PWD/.venv/bin/python"
```

This regenerates figures and tables from the distributed numerical inputs,
without training networks or recalibrating models. Outputs stay beside each
experiment in its ignored `outputs/` directory. Part II's price-path PDF export
also needs `rsvg-convert` (librsvg). Other PDFs use Matplotlib directly.
Network-specific tests are skipped unless the optional `part_2[solver]`
dependency is installed; the archived-results tests always run.

Part I's original numerical solvers are MATLAB code (R2025a, Optimization
Toolbox). Each experiment has one entry point, described in its README.
Recomputation writes new outputs, never overwrites the archived inputs, and is
separate from the quick reproduction. Part II documents its optional GPU solver.
To check the recovered Part I solvers against archived policies in MATLAB:

```matlab
addpath('part_1/tests'); verify_matlab;
```

Only the paper's central experiments are included: no alternative information
regimes, unpublished calibration sweeps, or presentation assets. Part I source
origins and hashes are in `part_1/sources.json`; Part II retains its own provenance.
The stationary-monopoly README identifies a numerical claim whose original
calibration has not yet been recovered. This is not a complete release until that
claim is verified.

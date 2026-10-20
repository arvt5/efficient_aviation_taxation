# Efficient Aviation Taxation

Reproduction of the paper's free-entry aviation model with full-information
dynamic pricing and endogenous flight frequency. The reference paper revision is
`8b74b56e040001907dc5b179d783644ea925ffa1` from
[the paper repository](https://github.com/LennartStern/Efficient-aviation-tax-policy-with-endogenous-vacancy-rates).

This is a working migration checkpoint, not yet a complete publication release.
The reported endpoint accounting, archived diagnostics, interpolation roots, and
three figure assets can be regenerated. The numerical solver has been extracted
and smoke-tested. See [the remaining work](provenance/OPEN_ISSUES.md), including
two policy-version discrepancies in the paper.

## Reproduce the archived results

Use Python 3.12 for the tested dependency pins. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m aviation_taxation.reproduce
MPLBACKEND=Agg .venv/bin/python -m aviation_taxation.welfare_plot
.venv/bin/python -m aviation_taxation.price_figures
```

The last command always writes SVGs. PDF and PNG exports additionally require
the `rsvg-convert` executable from librsvg (for example, `brew install librsvg`
or the distribution's `librsvg2-bin` package).

Results are written to `outputs/`:

- `policy_endpoints.csv` and `.tex`: all four rows, with calendar accounting.
- `equilibrium_quality.csv`: the directly optimized archived diagnostic values.
- `interpolation.json`: recomputed fits, leave-one-out diagnostics, and roots.
- `audit.json`: 42 comparisons against published values and archived roots.
- `welfare_vs_frequency.pdf`, `.svg`, `.png`: four welfare concepts.
- `price_path_status_quo.*` and `price_path_passenger_neutral.*`.

After installation, `make reproduce PYTHON=.venv/bin/python` runs the input
integrity check and all three reproduction commands.

Diagnostics in the default reproduction are archived outputs, explicitly labeled
as such. They are not represented as new equilibrium certification. Likewise,
the price-path CSVs are derived from archived policy arrays using stationary
occupancy weights conditional on the focal flight having inventory.

## Numerical model

`src/aviation_taxation/` contains the economic geometry and demand (`model`),
network and sampling (`network`, `sampling`), full-information equilibrium
conditions (`equilibrium`), forward aggregation (`forward`), fixed-policy value
evaluation (`evaluation`), value supervision (`value_refinement`), conditional
best responses and refinement (`best_response`, `policy_refinement`), and direct
one-shot diagnostics (`diagnostics`). No old project checkout is imported.

The current model implementation covers the frequency band `2 < M <= 3`, with
two or three active flights. The welfare figure's extension beyond `M=3` is a
quadratic extrapolation, not an equilibrium calculation with more active firms.

## Optional expensive recomputation

Install the solver dependency and use an allocated compute node:

```bash
.venv/bin/python -m pip install -e '.[solver]'
.venv/bin/python -m aviation_taxation.train \
  --config config/passenger_training.json \
  --device cuda --output outputs/passenger_network.pt
.venv/bin/python -m aviation_taxation.evaluation \
  --model outputs/passenger_network.pt --n-phi 101 --burn-in 20 \
  --n-price 101 --top-k 1000 --cross-weight 1 \
  --artifact-dir outputs/passenger_initial \
  --out outputs/passenger_evaluation.json
.venv/bin/python -m aviation_taxation.policy_refinement \
  --policy outputs/passenger_initial/policy_eval_arrays.npz \
  --result-json outputs/passenger_evaluation.json \
  --out-dir outputs/passenger_refinement --outer-iters 12 \
  --burn-in 20 --dampings 0.4 0.2 0.1 0.05 0.025 \
  --coarse-points 101 --local-steps 28 --save-accepted-checkpoints
```

Then run `aviation_taxation.diagnostics` on each retained checkpoint and select
the policy with the smallest directly checked maximum gain using
`aviation_taxation.checkpoint_selection`. The historical passenger run selected
the first refined checkpoint rather than the last. Exact commands and complete
archival inputs for the remaining calibration and sweep experiments are still
being assembled; the above is a single-equilibrium recomputation path.

The original passenger job requested one GPU, 32 CPU cores, 192 GB memory, and a
24-hour limit. These are requested resources, not a measured minimum. A full
retraining has not been run as part of this migration. Different hardware and
PyTorch versions may produce different trained policies; compare economic
metrics and deviation gains, not network parameter bytes.

## Verification and provenance

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -v
```

The tests include independent demand quadrature, full-residual differentiation,
a small value evaluation, fiscal identities, subsidy sign preservation, and
published-number comparisons. Solver tests require PyTorch.

`config/paper.json` records the target table and assumptions. `data/archived/`
contains only inputs needed for the current outputs. Original filenames inside
metadata are provenance references, not runtime paths.
`provenance/sources.json` records source hashes and extraction details, and
`provenance/files.json` records the distributed inputs' hashes.

The first release is scoped to the free-entry model with `kappa=0.28`. The paper's
monopoly, refund, and exogenous-flight duopoly experiments are outside this
package. Other calibration variants, unpublished endpoints, and presentation
material are excluded.

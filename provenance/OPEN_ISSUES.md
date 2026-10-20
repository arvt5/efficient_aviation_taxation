# Release audit

## Confirmed output and policy distinctions

1. **Free-entry accounting and certification use different policy versions.**
   The endpoint table uses the directly trained M=2.61 equilibrium before later
   tabular refinement. Its load factor is 96.2987%, its net revenue USD 849.0
   billion, and its welfare gain USD 67.1 billion. The diagnostic table uses a
   later refined policy with relative direct one-shot gain 0.0039268800142336436.
   These published values reproduce, but they do not constitute one consistently
   evaluated policy. The package preserves the distinct sources and does not
   imply otherwise. Selecting a common policy would require a paper update.

2. **The passenger endpoint now consistently uses the pre-refinement policy.**
   The authors selected the initial certified neural policy for the endpoint
   accounting and diagnostic, matching the existing price PDF. Comparing the
   PDF vector path with that policy's weighted prices gives 247 of 247 matching
   vertices within 0.015 PDF points. The formerly selected refined policy matches only
   one vertex at that tolerance. The archived figure uses USD 400 per model-price
   unit and y limits USD 50--250. The reproduction retains that figure's policy
   and scale explicitly in `price_path_metadata.json`.

3. **Passenger neutrality is approximate in the direct solve.**
   Interpolation selected M=2.4686982215230104. Under the selected initial policy,
   passenger volume is 4.747872281 billion, 0.67% below the 4.78 billion baseline.
   The package preserves the computed value rather than substituting the target.
   The frequency and interpolation are unchanged. No additional equilibrium
   calculations were run to adopt this checkpoint.

4. **Frequency figure and endpoint table use distinct numerical constructions.**
   The figure fits quadratics to the selected sweep's 15 observations with
   M>=2.3; the carbon instrument uses its own selected polynomial. Endpoint
   accounting comes from endpoint runs. Extrapolated maxima are not solved
   equilibria beyond the model's frequency band.

5. **Deviation statistics are limited direct searches.**
   Reported diagnostics continuously optimize a ranked subset of candidate
   deviations, with time-grid scaling. They should not be interpreted as a
   new exhaustive proof against every Markov deviation. The fast reproduction
   labels them as archived diagnostics.

## Outstanding package work

- Locate and reproduce the fixed-frequency ticket-tax validation elasticity
  reported as -0.94. No identified experiment has yet been substituted for it.
- Trace the empirical schedule-delay calibration inputs and appendix table.
- Pin calibration and sweep training configurations, including effective random
  seeds, and provide the complete optional recomputation orchestration.
- Package selected full policy arrays and network checkpoints in a public data
  archive with checksums. The small price series and accounting inputs in this
  repository regenerate outputs, but do not replace the full policy archive
  needed to recalculate all diagnostics and stationary accounts.
- Audit the remaining prose claims and fitted welfare maxima against the paper.
- Compare the extracted solver with the original code on larger numerical cases;
  complete cluster retraining is a separate expensive validation exercise.
- Record the license and archival release identifier before publication.

## Migration choices

Numerical routines were extracted from a cluster source snapshot by following
the dependency closure of the required full-information entry points. Imports
were renamed and unrelated functions omitted. The original local geometry file
contains a later floating-point boundary fix absent from the cluster snapshot;
this extraction retains the cluster geometry used for archived results. The
paper's baseline T=3,d=1 is exactly represented. Broader numerical improvements
must be versioned and tested separately from reproduction of archived policies.

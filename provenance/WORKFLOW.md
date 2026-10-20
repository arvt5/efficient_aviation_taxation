# Agent workflow

This file records completed checkpoints and the next autonomous work units.
The frozen scope is the paper's full-information free-entry results only.

## Checkpoint 1: source audit and working extraction

Completed on 2026-10-20:

- Pulled and pinned the paper revision; cloned the empty target repository.
- Retrieved a bounded cluster source/metadata snapshot with original-file hashes.
- Located the final selected passenger policy and verified its fiscal identity.
- Extracted only numerical routines reachable from the required entry points.
- Regenerated all 32 endpoint-table cells and eight archived diagnostic values.
- Recomputed both interpolation roots, including leave-one-out fit selection.
- Regenerated the price-path and four-objective welfare figures.
- Verified the four implementing tax/subsidy values at the paper's precision.
- Ran independent demand, gradient, policy-evaluation, and accounting smoke tests.
- Compared the extracted equilibrium loss and all network gradients with the
  cluster source on an identical small-state batch: bit-identical outputs.
- Identified and documented accounting/diagnostic and price-figure policy-version
  discrepancies. No paper values or source policies were altered to hide them.

## Next work units

1. Complete the claim-level provenance manifest, particularly validation
   elasticity and the empirical schedule-delay appendix.
2. Build the selected-policy archive, omitting rejected iterations and unused
   calibration variants. Publish a retrieval manifest with public archive URLs.
3. Recalculate stationary accounting from the selected archived policies and
   decide, with the authors, how the paper's policy-version discrepancies should
   be resolved. Preserve the current-paper reproduction as a versioned target.
4. Extract calibration and sweep orchestration and exact experiment settings;
   add resource-aware optional batch jobs and checkpoint recovery instructions.
5. Verify from a fresh environment and produce the release audit, licenses,
   numerical tolerances, and archival identifiers.

Full equilibrium retraining requires a separately scheduled compute allocation.
No cluster training jobs were submitted during checkpoint 1.

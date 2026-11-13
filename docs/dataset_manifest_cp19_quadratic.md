# Problem manifest — Checkpoint 19 quadratic optimizer comparison

- **Name:** Two-dimensional squared distance to a known parameter vector.
- **Source:** Defined analytically in `experiments/cp19_optimizer_comparison.py`:
  `loss = 0.5 * sum((parameters - target) ** 2)` and its exact gradient is
  `parameters - target`.
- **External data:** None downloaded or copied; this is a synthetic objective,
  not an external dataset.
- **License/provenance:** Mathematical test problem created by this project;
  no third-party data or license applies.
- **Configuration:** `configs/cp19_optimizer_comparison.json`.
- **Randomness:** None; all algorithms receive the same starting vector,
  target, objective and 200-step budget.
- **Comparison note:** Learning rates are explicitly recorded per optimizer.
  Results show behavior for this one convex quadratic, not a general ranking.

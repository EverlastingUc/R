# Dataset manifest — Checkpoint 18 synthetic linear regression

- **Name:** Locally generated one-feature linear regression samples.
- **Source:** `experiments/cp18_gradient_descent.py` creates evenly spaced
  inputs and computes `target = slope * input + intercept` from the versioned
  configuration.
- **External data:** None downloaded or copied.
- **License/provenance:** Generated from a mathematical rule by this project;
  no third-party dataset license applies.
- **Configuration:** `configs/cp18_gradient_descent.json`.
- **Seed:** 18 for network initialization. The fixed grid itself is
  deterministic and does not use random sampling.
- **Splits:** One tiny training set is used to demonstrate optimization. There
  is no validation or test set because this checkpoint measures whether a
  gradient step lowers training loss, not generalization.
- **Known limitation:** This simple rule is only a correctness demonstration;
  it does not establish performance on real-world regression data.

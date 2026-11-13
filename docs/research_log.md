# Research log

## 2026-11-08 — Checkpoint 01: Artificial Intelligence Fundamentals

- **Done:** Set up the Entropy project structure and documented AI, machine
  learning, deep learning, neural networks, and the training vocabulary.
- **Evidence:** `docs/checkpoints/cp01.md` covers the Checkpoint 01 acceptance
  line.
- **Failures:** None encountered. The project had no prior research log.
- **Next:** Checkpoint 02, a NumPy artificial neuron with hand-calculated tests.

## 2026-11-08 — Checkpoint 02: Biological and Artificial Neurons

- **Done:** Implemented a validated artificial neuron with identity and stable
  sigmoid activations. Added 32 hand-calculated and input-validation cases.
- **Evidence:** `docs/checkpoints/cp02.md`; `32 passed` from the project venv.
  The test-first run failed before implementation because the module was absent,
  as expected.
- **Failures:** No remaining test failures. `pytest` was missing from the host
  Python, so a project-local `.venv` was created and pinned dependencies were
  installed there.
- **Next:** Checkpoint 03, document the feedforward network architecture.

## 2026-11-08 — Checkpoint 03: Neural Network Architectures

- **Done:** Specified Entropy's 2–4–1 feedforward architecture, tensor shapes,
  and forward calculation.
- **Evidence:** `docs/checkpoints/cp03.md` contains the diagram and architecture
  specification.
- **Failures:** None.
- **Next:** Checkpoint 04, run a reproducible training experiment that contrasts
  memorization with generalization.

## 2026-11-08 — Checkpoint 04: Learning Mechanisms

- **Done:** Ran a fixed-seed experiment comparing structured XOR learning with
  fitting independent random labels. Saved a plot and a metrics record with
  config hash, runtime, and machine details.
- **Evidence:** Structured XOR achieved 1.000 train and held-out accuracy;
  random labels achieved 1.000 train accuracy and 0.500 held-out accuracy.
  The experiment tests passed; the full suite passed with 37 tests.
- **Failures:** None. The random-label result is expected because its labels
  are independent of the inputs.
- **Next:** Checkpoint 05, analyze three foundational neural-network papers.

## 2026-11-08 — Checkpoint 05: AI Research Literacy

- **Done:** Wrote notes on McCulloch and Pitts (1943), Rosenblatt (1958), and
  Rumelhart, Hinton, and Williams (1986). Marked paper-supported facts as
  established and project interpretations separately.
- **Evidence:** `docs/checkpoints/cp05.md` and three notes in `docs/papers/`;
  publication details were checked against publisher and PubMed records.
- **Failures:** Full text for the 1986 Nature article was unavailable from the
  publisher during this review; its note is limited to the publisher abstract
  and citation record.
- **Next:** Phase 01 is complete. Stop and wait for owner confirmation before
  beginning Phase 02.

## 2026-11-08 — Checkpoint 06: Linear Algebra

- **Done:** Implemented dot product, transpose, matrix-vector multiplication,
  and matrix multiplication; added a runnable example and NumPy comparisons.
- **Evidence:** `docs/checkpoints/cp06.md`; focused tests: 8 passed; full suite:
  45 passed; `black .` and `ruff check .` passed.
- **Failures:** The first test-first run correctly failed because the module did
  not exist. The demo then exposed that the `src` package was not installed in
  the project venv; added package metadata and an editable install, after which
  the demo ran successfully.
- **Next:** Checkpoint 07, calculus and the chain rule.

## 2026-11-08 — Checkpoint 07: Calculus

- **Done:** Implemented sigmoid and MSE derivatives plus the chain-rule
  gradients for a sigmoid neuron; added a runnable demonstration.
- **Evidence:** `docs/checkpoints/cp07.md`; focused tests: 9 passed; full suite:
  54 passed; finite-difference comparison passed; Black and Ruff passed.
- **Failures:** The initial test-first run failed because the calculus module
  did not yet exist, as expected.
- **Next:** Checkpoint 08, probability, expectation, variance, and sampling.

## 2026-11-08 — Checkpoint 08: Probability and Statistics

- **Done:** Implemented normalized distributions, expectation, variance,
  entropy in bits, conditional probability, seeded sampling, and a sample
  frequency experiment.
- **Evidence:** `docs/checkpoints/cp08.md`; focused tests: 10 passed; full suite:
  64 passed; fixed-seed sample frequencies were `[0.2522, 0.7478]` for target
  probabilities `[0.25, 0.75]`; Black and Ruff passed.
- **Failures:** Initial test-first run failed because the probability module
  did not yet exist, as expected.
- **Next:** Checkpoint 09, numerical stability and floating-point behavior.

## 2026-11-08 — Checkpoint 09: Numerical Computing

- **Done:** Implemented stable softmax and log-sum-exp, plus a float32 versus
  float64 precision demonstration.
- **Evidence:** `docs/checkpoints/cp09.md`; focused tests: 8 passed; full suite:
  72 passed; the large-logit demonstration remained finite with the stable
  implementation; Black and Ruff passed.
- **Failures:** Ruff initially reported one unused import; it was removed and
  the lint check passed.
- **Next:** Checkpoint 10, gradient descent and convergence.

## 2026-11-08 — Checkpoint 10: Optimisation Mathematics

- **Done:** Implemented vector gradient descent and a learning-rate comparison
  on a quadratic with a known minimum.
- **Evidence:** `docs/checkpoints/cp10.md`; focused tests: 7 passed; full suite:
  79 passed. Rate 0.5 reached the minimum; rate 1.1 increased loss from 64 to
  94,065.380. Black and Ruff passed.
- **Failures:** The initial test-first run failed because the optimizer module
  did not yet exist, as expected.
- **Next:** Phase 02 is complete; begin Phase 03, Checkpoint 11.

## 2026-11-08 — Checkpoint 11: Artificial Neuron

- **Done:** Reused the Phase 01 artificial neuron as the Phase 03 building
  block; it supports weighted inputs, bias, identity, and sigmoid activation.
- **Evidence:** `docs/checkpoints/cp11.md`; 32 hand-calculated and validation
  cases in `tests/test_neuron.py`.
- **Failures:** None. The implementation and tests were completed and recorded
  in Checkpoint 02.
- **Next:** Checkpoint 12, activation functions.

## 2026-11-08 — Checkpoint 12: Activation Functions

- **Done:** Implemented sigmoid, tanh, ReLU, leaky ReLU, and stable softmax,
  plus elementwise derivatives and a softmax Jacobian.
- **Evidence:** `docs/checkpoints/cp12.md`; focused tests: 13 passed; full suite:
  92 passed; finite-difference derivative checks passed; Black and Ruff passed.
- **Failures:** The initial test-first run failed because the activation module
  did not yet exist, as expected.
- **Next:** Checkpoint 13, a configurable multi-neuron layer.

## 2026-11-08 — Checkpoint 13: Multiple Neurons

- **Done:** Implemented a configurable dense layer with seeded Xavier-uniform
  initialization, explicit parameter injection, batch forward propagation,
  and activation support.
- **Evidence:** `docs/checkpoints/cp13.md`; focused tests: 6 passed; full suite
  after Checkpoint 14: 102 passed; Black and Ruff passed.
- **Failures:** One test initially supplied weights without the required bias;
  corrected the test setup to check the intended invalid-weight-shape case.
- **Next:** Checkpoint 14, a multilayer feedforward network.

## 2026-11-08 — Checkpoint 14: Multilayer Network

- **Done:** Implemented a configurable feedforward network with at least two
  hidden layers and a per-layer forward cache.
- **Evidence:** `docs/checkpoints/cp14.md`; focused tests: 4 passed; full suite:
  102 passed; Black and Ruff passed.
- **Failures:** The test-first run failed because the module did not yet exist,
  as expected. A strict-zip length mismatch was corrected before the focused
  tests passed.
- **Next:** Checkpoint 15, independently verified forward propagation.

## 2026-11-08 — Checkpoint 15: Forward Propagation

- **Done:** Added fixed-weight, independently calculated checks for each layer
  and final outputs, including a batch case.
- **Evidence:** `docs/checkpoints/cp15.md`; focused tests: 3 passed; full suite:
  105 passed; Black and Ruff passed.
- **Failures:** None. The hand-derived second batch output was corrected before
  the test run after recalculating the bias contribution.
- **Next:** Phase 03 is complete; begin Phase 04, Checkpoint 16.

## 2026-11-08 — Checkpoint 16: Loss Functions

- **Done:** Implemented MSE, MSE gradients, binary cross-entropy, categorical
  cross-entropy, stable logits cross-entropy, and its softmax-logit gradient.
- **Evidence:** `docs/checkpoints/cp16.md`; focused tests: 8 passed; full suite:
  113 passed; large-logit loss remained finite; Black and Ruff passed.
- **Failures:** The test-first run failed because the loss module did not yet
  exist, as expected.
- **Next:** Checkpoint 17, multi-layer backpropagation and gradient checking.

## 2026-11-08 — Checkpoint 17: Backpropagation

- **Done:** Implemented dense-network MSE backpropagation, centered finite-
  difference gradient checking, and a GitHub Actions workflow that runs the
  full suite (including gradient checks) on pushes and pull requests.
- **Evidence:** `docs/checkpoints/cp17.md`; 23 parameters checked for each of
  six activations. Largest measured relative error was `1.549e-6`, below the
  `1e-5` tolerance. Full suite: 121 passed.
- **Failures:** The initial test-first collection failed because the new
  backpropagation module did not exist yet. Initial quality checks then caught
  an unused import and formatting differences; both were corrected.
- **Next:** Checkpoint 18, gradient-descent parameter updates.

## 2026-11-08 — Checkpoint 18: Gradient Descent

- **Done:** Added full-batch gradient descent, a seeded synthetic regression
  experiment and config, a provenance manifest, and a measured metrics report.
- **Evidence:** `docs/checkpoints/cp18.md` and
  `reports/cp18_training_metrics.json`; MSE fell from `1.081216` to
  `6.2836e-27` over 500 steps. Full suite: 123 passed; Black and Ruff passed.
- **Failures:** The first test-first run failed because the training module was
  absent. The first implementation test exposed a staged-update tuple bug;
  corrected the tuple to retain its layer, and the focused tests then passed.
- **Next:** Checkpoint 19, controlled optimizer comparison.

## 2026-11-08 — Checkpoint 19: Optimisers

- **Done:** Implemented SGD, momentum, RMSProp and Adam, with reproducible
  state updates and validation. Added a fixed quadratic comparison config and
  saved measured loss traces with the hardware and duration metadata.
- **Evidence:** `docs/checkpoints/cp19.md` and
  `reports/cp19_optimizer_comparison.json`; all four reduced the shared loss
  from `17.0`, with final values from `0.0` to `0.0030313`. Full suite: 132
  passed; Black and Ruff passed.
- **Failures:** The first test-first collection failed because the optimizer
  module was absent. An early RMSProp run at learning rate `0.01` did not meet
  the test's convergence threshold within 200 steps; the experiment setting
  was adjusted to `0.05` and recorded. A finite-state regression test then
  caught an infinite RMSProp accumulator; the optimizer now rejects it before
  updating parameters. Ruff import ordering and formatting findings were
  corrected.
- **Next:** Checkpoint 20, batching, validation and checkpoint saving.

## 2026-11-08 — Checkpoint 20: Training Pipeline

- **Done:** Implemented seeded mini-batch training, per-epoch validation,
  optimizer-state serialization, and safe save/restore of model, optimizer,
  epoch count and shuffle state. Added a reproducible synthetic run that saves
  a new numbered report/checkpoint each time.
- **Evidence:** `docs/checkpoints/cp20.md` and run 003 report/checkpoint;
  training MSE fell from `0.654636` to `0.000862`, validation MSE from
  `0.577936` to `0.048382`, and restored predictions matched exactly. Full
  suite: 139 passed; Black and Ruff passed.
- **Failures:** Test-first collection failed because `Trainer` was not yet
  implemented. The first experiment invocation passed the config's `name`
  field into Adam and failed; the constructor now receives only optimizer
  hyperparameters. Initial quality checks also found formatting and exception
  type issues, which were corrected. Three successful runs are preserved with
  numbered reports and checkpoints.
- **Next:** Phase 04 is complete. Before Checkpoint 21 uses data, present
  candidate sources with license/provenance and get owner approval as required
  by `PROMPTS.md`.

## 2026-11-08 — Phase 05 preparation (Checkpoint 21 not started)

- **Done:** Prepared two dataset plans in `docs/dataset_candidates_cp21.md`:
  project-generated synthetic classification/regression, or UCI Iris plus UCI
  Bike Sharing with CC BY 4.0 attribution and recorded provenance.
- **Evidence:** The candidate page uses official UCI repository pages for the
  real-data option. No Phase 05 data has been downloaded or used.
- **Next:** Wait for owner approval of one dataset plan, then begin CP21.

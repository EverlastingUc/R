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


## 2026-11-17 — Proton-I Stage 1: Gluons baseline

- **Done:** Added the Gluons package under `src/proton_i/gluons/`. It performs
  web search through DuckDuckGo's public HTML results page, selects a concise
  relevant snippet, returns one plain-text paragraph with a source URL, and
  rejects JavaScript/framework requests as out of scope. Added a CLI entry
  point, usage guide, tests, and README instructions. Existing Entropy network
  modules and checkpoints were not changed.
- **Evidence:** Targeted Gluons tests: 9 passed. Python compile check passed.
  The full repository suite has not been run in this environment.
- **Limitations:** Gluons 0.1 is a retrieval-based baseline, not a trained
  generative model. The live web request could not be verified because outbound
  internet access is unavailable in this development runtime. The DDG HTML
  markup is best-effort and may change or be rate-limited.
- **Next:** Run the Gluons command from a network-enabled local environment and
  verify live search. Valence Quarks has not been started.


## 2026-11-17 — Proton-I Stage 2: Valence mathematics scope preparation

- **Done:** Defined Valence as Proton-I's complete mathematics capability, from
  elementary arithmetic through advanced mathematical extensions. Specified
  the proposed deterministic solver architecture, parser and expression-tree
  safety requirements, correctness/verification rules, output contract, and
  14 incremental implementation milestones.
- **Scope boundary:** Mathematical computation, including basic arithmetic,
  belongs to Valence. Sea Quarks remains undefined and must not duplicate
  mathematics under the current stage boundary. Gluons remains website
  retrieval only.
- **Files:** `docs/proton_i/VALENCE_MATH_SPEC.md` and the updated
  `docs/proton_i/PROTON_I_SCOPE.md`.
- **Evidence:** Documentation and repository state only. No Valence solver
  code has been implemented or tested in this preparation step.
- **Next:** Begin VQ-01 with tests-first implementation of the safe expression
  parser and exact arithmetic, subject to owner approval of the implementation
  milestone.


## 2026-11-17 — Proton-I Valence product-direction update and local-chat core

- **Owner-approved scope update:** Valence is the primary general-purpose conversational
  assistant and task-orchestration layer, with a DesignArena-inspired creation
  workspace. It is not merely a mathematics engine. Mathematics is one of its
  capabilities; the previous mathematics specification remains as that subsystem's
  detailed roadmap. Gluons remains website retrieval only. Sea Quarks remains undefined.
- **Design reference researched:** DesignArena's current product describes prompt-driven
  creation across websites, games, images, PDFs, spreadsheets, slides, 3D assets,
  mobile apps, SVG, UI components and data visualisations, plus model comparison and
  leaderboards. These are target capability areas, not current Valence features.
- **Implementation started:** Defined a provider-neutral chat contract, local Ollama
  backend, bounded multi-turn conversation engine and CLI. The first local model
  default is Qwen2.5 1.5B Instruct; users install its weights separately. No model
  files or API keys are committed.
- **Verification design:** Mocked backend and HTTP tests cover history, reset,
  bounded context, failure rollback, prompt validation, loopback-only host validation,
  request payloads and malformed/empty provider responses. No live Ollama model was
  available in the CI environment; real local inference still requires owner-side
  installation and execution.
- **Next:** Run repository CI. Then continue with the API and tool contract before
  adding retrieval, mathematics execution, memory, files or creation pipelines.


## 2026-11-17 — Valence VQ-A01: local conversational core verified

- **Done:** Implemented the model-backend contract, local Ollama HTTP adapter,
  bounded multi-turn conversation engine, reset behavior and interactive CLI.
  The backend accepts only a loopback Ollama URL and validates message sizes,
  response JSON and non-empty assistant content.
- **Evidence:** GitHub Actions run 44118599371 passed. Black: 54 files unchanged;
  Ruff: all checks passed; pytest: 173 passed.
- **Limitations:** CI uses mocked model HTTP responses. A real local model has
  not yet been installed or run in this environment. Conversation history is
  process-local memory and disappears when the CLI exits.
- **Next:** VQ-A02, expose the same engine through a loopback-only JSON API.

## 2026-11-17 — Valence VQ-A03: typed tool contract verified

- **Done:** Added `ToolParameter`, `ToolDefinition`, `CapabilityRegistry`,
  `ExecutionPolicy`, `ToolExecutor`, and `ToolResult`. Schemas are exposed
  only for explicitly allowed tools. The default policy denies every tool.
  Required fields, JSON types, enum choices, numeric bounds, string/collection
  limits, unknown keys, and argument sizes are validated before a handler runs.
- **Safety boundary:** Only application-registered Python handlers can execute.
  Sensitive tools require application-side approval. Model output is never
  interpreted as Python or shell code. Handler exceptions are redacted;
  outputs must be JSON-compatible and within a configurable size limit.
  Trusted handlers must enforce native timeouts for their blocking operations.
- **Evidence:** GitHub Actions [run 44122468478](https://github.com/EverlastingUc/R/actions/runs/44122468478) passed.
  Black: 58 files unchanged; Ruff: all checks passed; pytest: 216 passed in
  10.01 seconds. Tests cover default-deny behavior, allowlist filtering,
  validation, approval gates, failure redaction, serialization, and size limits.
- **Limitations:** No external tools are registered by default and the model
  does not yet invoke this executor automatically. This is the trusted tool
  contract only, not a public plugin system or a code-execution sandbox.
- **Next:** VQ-A04, integrate Gluons as a narrowly scoped, source-attributed
  retrieval tool under an explicit policy and bounded query/result sizes.



## 2026-11-17 — Valence VQ-A04: Gluons retrieval adapter verified

- **Done:** Added `create_gluons_tool` and `register_gluons_tool`. The tool
  accepts one bounded text query, requests at most five Gluons search results,
  validates HTTP(S) source URLs (rejecting embedded credentials and invalid
  ports), caps title/snippet/source lengths, and returns source-attributed
  result objects. Every response marks retrieved text as
  `untrusted_web_content`.
- **Policy boundary:** Registering the tool does not enable it. The executor
  still requires an explicit policy allowlist containing `search_web`.
  Search failures are converted into a sanitized, application-authored
  `ToolHandlerError`; generic handler exceptions remain redacted.
- **Evidence:** GitHub Actions [run 44123523142](https://github.com/EverlastingUc/R/actions/runs/44123523142) passed.
  Black: 61 files unchanged; Ruff: all checks passed; pytest: 224 passed in
  9.32 seconds. Tests use an injected fake search provider; no live network
  request is made in CI.
- **Limitations:** Valence's local model still does not automatically request
  or execute tool calls. Applications can explicitly register and permit the
  Gluons tool, but model-facing tool-call orchestration remains a separate
  milestone.
- **Next:** VQ-A05, add a bounded Ollama tool-call loop that validates each
  model request, checks registry/policy, executes allowed tools, and feeds
  structured results back to the model. Limit tool rounds and never evaluate
  model-generated code.


## 2026-11-17 — Valence VQ-A05: model tool-call orchestration verified

- **Done:** Connected Ollama's structured function-call responses to the
  Valence tool registry and executor. The backend validates bounded tool-call
  envelopes and serializes assistant tool requests plus tool-role results.
  The conversation engine enforces the application allowlist, maximum tool
  rounds/calls, bounded tool-result messages, and final tool-free response.
- **Application path:** Added --enable-web-search to both valence and
  valence-api. This explicitly registers the Gluons retrieval tool and grants
  the search_web policy permission. Without the flag, the default remains
  deny-all and no external tool is registered.
- **End-to-end coverage:** Added a deterministic integration test in which a
  fake model requests Gluons search, the provider returns source-attributed
  snippets, and the structured untrusted result is sent back to the model.
- **Safety boundary:** Model output is never evaluated as Python or shell code.
  Only application-registered handlers can run. Unknown, malformed, disabled,
  disallowed, over-budget, or approval-gated tools do not execute. Web snippets
  retain source URLs and are marked untrusted_web_content.
- **Evidence:** GitHub Actions run 44126687355 passed. Black check, Ruff check,
  and the full pytest suite all passed: 244 passed in 9.77 seconds.
- **Limitations:** CI uses fake backends and an injected search provider. A real
  local Ollama model has not been installed or exercised in this environment.
  The local assistant's quality remains bounded by the separately installed
  model. Search remains opt-in; no arbitrary plugin marketplace, remote model
  routing, code execution, or full-page browser is included.
- **Next:** VQ-A06, implement the mathematics subsystem in
  VALENCE_MATH_SPEC.md, beginning with safe expression parsing and exact
  arithmetic before expanding to symbolic algebra and advanced domains.

# GENERAL-PURPOSE AI — MASTER ENGINEERING SPECIFICATION v1.1

**Project type:** Independent AI research and development
**Target:** General-purpose AI assistant
**Development strategy:** Neural network and model training from scratch (research track) + fine-tuning of open-weights models (assistant track)
**Compute strategy:** Hybrid local + cloud
**Languages:** Python, JavaScript, Rust, C++
**Roadmap:** 12 phases, 60 checkpoints

---

## 0. HOW CODEX MUST USE THIS DOCUMENT

This file is the single source of truth for the project. Follow these rules in every session:

1. **Work on exactly one checkpoint at a time**, in order (01 → 60). Do not start a checkpoint before the previous one is accepted.
2. **Tests first.** For every implementation checkpoint, write the tests (derived from the Acceptance line) before or together with the code.
3. **Evidence is mandatory.** A checkpoint is done only when its Acceptance condition is met *and* proof exists in the repo (test output, plot, report, or log under `reports/`).
4. **Never claim a capability that was not measured.** If something was not evaluated, say "not evaluated".
5. **Phases 03 and 04 use NO ML frameworks.** Python + NumPy only (no PyTorch, TensorFlow, JAX, scikit-learn models). Frameworks are allowed as *reference checks in tests only*, and from Phase 06 onward as the main tool where stated.
6. **Keep a research log** (`docs/research_log.md`). After every session append: date, checkpoint, what was done, what failed, what is next.
7. **Ask before big decisions** (dataset choice, model size, paid compute, licensing doubts). Do not guess.
8. **Preserve working versions.** Never overwrite a passing implementation; version it or branch it.
9. At the end of every session output: checkpoint status, tests passed/failed, files changed, next step.

---

## 1. PROJECT SUMMARY

Build a general-purpose AI assistant by (a) learning and implementing neural networks and transformers from first principles, then (b) training a small language model from scratch as a research artifact, and (c) building a usable instruction-following / coding assistant on top, with a proper API, interface, evaluation and security process.

---

## 2. STRATEGIC DECISIONS AND SCOPE NOTES (READ BEFORE STARTING)

These notes are based on an honest review of the original roadmap. They change *how* some phases are executed, not the checkpoint list itself.

### 2.1 Two tracks

| Track | Phases | What it is | Realistic outcome |
|---|---|---|---|
| **Track A — From-scratch research model** | 01–08, 10 | Own neurons, backprop, transformer, small LM trained from zero | A small, genuinely-yours language model. A learning and research success, **not** a competitive assistant. |
| **Track B — Usable assistant** | 09, 11, 12 | Instruction-following, coding, API, UI, tools, security, release | Built by **fine-tuning an open-weights model** (license permitting), optionally comparing against the Track A model. |

**Why:** capability of modern assistants comes from billions of parameters and trillions of tokens. A from-scratch model on hobbyist/limited cloud compute will not become a competent general-purpose or coding assistant. Phase 09 therefore targets (1) the Track A small model for *measured improvement over its own baseline*, and (2) an open-weights model for the actual usable assistant. The owner may override this decision explicitly; if so, record it in the research log.

### 2.2 Honest observations that affect execution

- **Scope reality:** Phases 01–08 are fully achievable. A small decoder-only model trained on a modest corpus is already a strong result.
- **Checkpoint 17 (backpropagation) is the most important checkpoint in the project.** Gradient checking against finite differences is the permanent safety net against subtle bugs. Keep the gradient-check utility and run it in CI for all later layers (including attention).
- **Checkpoint 37 (dataset) is the hidden bottleneck.** Collecting, licensing, deduplicating and filtering text usually takes longer than writing training code. Decide early between existing open datasets and a custom dataset. Do not train on data with unknown licensing.
- **Phase 10 ordering:** Rust/C++ optimisation comes only *after* a correct Python reference implementation and baseline benchmarks exist. Never optimise what is not yet measured. Every optimised kernel must match the reference numerically within a documented tolerance.
- **Compute honesty:** Record real hardware, time and cost for every training run. Pick model sizes that the available compute can actually finish. Distributed/multi-GPU work (Checkpoint 50) is done only if resources justify it.
- **Evaluation honesty:** Small models memorise easily. Always keep a held-out test set untouched until final reporting.
- **Research literacy:** Prefer reproducing known results before inventing new ones. Mark every idea as *established* or *experimental* in the docs.

---

## 3. RECOMMENDED REPOSITORY STRUCTURE

```
project-root/
├── MASTER_SPEC.md            # this file
├── docs/
│   ├── research_log.md
│   ├── checkpoints/          # one short report per checkpoint (cpXX.md)
│   ├── papers/               # paper notes (Checkpoint 05)
│   └── model_cards/
├── src/
│   ├── nn/                   # from-scratch NumPy implementation (Phases 03-05)
│   ├── deep/                 # deep learning experiments (Phase 06)
│   ├── transformer/          # tokenizer, embeddings, attention, blocks (Phase 07)
│   ├── lm/                   # language model + training (Phase 08)
│   ├── native/
│   │   ├── cpp/              # C++ kernels (Phase 10)
│   │   └── rust/             # Rust infrastructure (Phase 10)
│   ├── api/                  # inference API (Phase 11)
│   └── tools/                # tool calling (Phase 11)
├── web/                      # JavaScript interface (Phase 11)
├── data/                     # raw/processed datasets (gitignored, with manifests)
├── configs/                  # versioned, reproducible training configs
├── experiments/              # run outputs, metrics, checkpoints
├── reports/                  # evidence: benchmarks, plots, evaluation reports
└── tests/
```

---

## 4. ENGINEERING CONVENTIONS

- **Reproducibility:** fixed random seeds, pinned dependency versions, every run driven by a config file stored in `configs/`.
- **Versioning:** every trained model gets a version tag, a config hash and a dataset manifest reference.
- **Testing:** `pytest` for Python, native test suites for Rust/C++, integration tests for the API and UI.
- **Numerical checks:** compare against trusted references (NumPy built-ins, analytical values, finite differences) with explicit tolerances.
- **Data hygiene:** train/validation/test always separated; test set never used for tuning.
- **Logging:** log hardware, training duration, compute cost and metrics for every run.

---

# PHASE 01 — FOUNDATIONS OF ARTIFICIAL INTELLIGENCE

**Objective:** Understand the scientific and computational foundations of neural networks.

### Checkpoint 01 — Artificial Intelligence Fundamentals
- Study AI, machine learning, deep learning and neural networks.
- Understand supervised, unsupervised and reinforcement learning.
- **Acceptance:** Produce a technical document distinguishing all four concepts with examples.

### Checkpoint 02 — Biological and Artificial Neurons
- Understand biological neurons at a conceptual level.
- Study artificial neurons, inputs, weights, biases and outputs.
- **Acceptance:** Implement and explain a mathematical artificial neuron.

### Checkpoint 03 — Neural Network Architectures
- Study feedforward networks, recurrent networks and convolutional networks.
- Understand input, hidden and output layers.
- **Acceptance:** Implement a basic feedforward network diagram and architecture specification.

### Checkpoint 04 — Learning Mechanisms
- Understand how models learn from training examples.
- Study training, validation, testing, generalisation and overfitting.
- **Acceptance:** Demonstrate the difference between memorisation and generalisation experimentally.

### Checkpoint 05 — AI Research Literacy
- Study research papers, technical documentation and reproducible experiments.
- Learn how to distinguish established findings from experimental proposals.
- **Acceptance:** Analyse and document three foundational neural network research papers.

---

# PHASE 02 — MATHEMATICAL FOUNDATIONS

**Objective:** Develop the mathematics required to implement and understand neural networks.

### Checkpoint 06 — Linear Algebra
- Vectors, matrices, tensors, dimensions and matrix multiplication.
- Dot products, transposes and matrix transformations.
- **Acceptance:** Implement matrix operations and verify results against a trusted numerical implementation.

### Checkpoint 07 — Calculus
- Functions, derivatives, partial derivatives and chain rule.
- Understand gradients and their relationship to optimisation.
- **Acceptance:** Calculate derivatives of representative neural network functions.

### Checkpoint 08 — Probability and Statistics
- Probability distributions, expectation, variance and sampling.
- Understand entropy and conditional probability.
- **Acceptance:** Implement probability calculations and basic statistical experiments.

### Checkpoint 09 — Numerical Computing
- Floating-point arithmetic, precision and numerical stability.
- Understand overflow, underflow and numerical errors.
- **Acceptance:** Demonstrate and document numerical stability tests.

### Checkpoint 10 — Optimisation Mathematics
- Gradient descent, learning rates and convergence.
- Study local minima, saddle points and optimisation behaviour.
- **Acceptance:** Implement gradient descent and demonstrate convergence on known functions.

> **Execution note:** Keep Phase 02 practical. Every concept gets a small runnable script plus a test. Stability tests from Checkpoint 09 (e.g. softmax with large logits, log-sum-exp) must be reused later in Phases 03–08.

---

# PHASE 03 — NEURAL NETWORK IMPLEMENTATION

**Objective:** Build the first functioning neural network without machine learning frameworks.

> **Constraint:** Python + NumPy only. No ML frameworks except as reference checks inside tests.

### Checkpoint 11 — Artificial Neuron
- Implement weighted inputs, bias and activation.
- **Acceptance:** Correctly calculate outputs for at least 20 test cases.

### Checkpoint 12 — Activation Functions
- Implement sigmoid, tanh, ReLU, leaky ReLU and softmax.
- **Acceptance:** Verify outputs, ranges and derivatives against mathematical reference tests.
- *Note:* Implement softmax in its numerically stable form.

### Checkpoint 13 — Multiple Neurons
- Connect multiple neurons into computational layers.
- **Acceptance:** Build a functioning layer with configurable neuron counts.

### Checkpoint 14 — Multilayer Network
- Implement input, hidden and output layers.
- **Acceptance:** Build and execute a network with at least two hidden layers.

### Checkpoint 15 — Forward Propagation
- Implement forward propagation across all layers.
- **Acceptance:** Verify outputs against independently calculated reference results.

---

# PHASE 04 — TRAINING ENGINE

**Objective:** Make the network learn from examples.

> **Constraint:** Still NumPy only.

### Checkpoint 16 — Loss Functions
- Implement mean squared error and cross-entropy.
- Understand classification and regression objectives.
- **Acceptance:** Verify loss calculations against known numerical examples.

### Checkpoint 17 — Backpropagation  ⭐ CRITICAL
- Implement the chain rule across network layers.
- Calculate gradients for weights and biases.
- **Acceptance:** Compare analytical gradients with numerical finite-difference approximations.
- *Note:* Build a reusable `gradient_check()` utility. It must be re-run in CI for every new layer type added in later phases.

### Checkpoint 18 — Gradient Descent
- Implement parameter updates.
- Experiment with learning rates.
- **Acceptance:** Demonstrate measurable loss reduction on a suitable training dataset.

### Checkpoint 19 — Optimisers
- Implement SGD, momentum, RMSProp and Adam.
- **Acceptance:** Compare convergence behaviour under controlled experiments.

### Checkpoint 20 — Training Pipeline
- Implement batching, epochs, shuffling, validation and checkpoint saving.
- **Acceptance:** Train a reproducible model and successfully restore its saved parameters.

---

# PHASE 05 — FIRST TRAINED MODELS

**Objective:** Establish a working experimental model-development pipeline.

### Checkpoint 21 — Dataset Engineering
- Understand dataset collection, cleaning, labelling and splitting.
- **Acceptance:** Produce a documented, reproducible dataset pipeline.

### Checkpoint 22 — Classification Model
- Train a model on a suitable classification dataset.
- **Acceptance:** Beat a documented baseline on held-out data.

### Checkpoint 23 — Regression Model
- Train a model to predict continuous values.
- **Acceptance:** Evaluate using appropriate regression metrics.

### Checkpoint 24 — Regularisation
- Study dropout, weight decay, early stopping and data augmentation.
- **Acceptance:** Compare regularised and unregularised models using validation results.

### Checkpoint 25 — Model Evaluation
- Study accuracy, precision, recall, F1, confusion matrices and calibration.
- **Acceptance:** Publish a complete experiment report with training, validation and test results.

---

# PHASE 06 — DEEP LEARNING

**Objective:** Develop the knowledge required for large neural architectures.

> **Execution note:** From this phase onward a mainstream framework (e.g. PyTorch) may be used for speed, but every new component should first be verified against the from-scratch implementation or the gradient-check utility where practical.

### Checkpoint 26 — Deep Feedforward Networks
- Implement and experiment with deeper networks.
- **Acceptance:** Train a multilayer model and investigate gradient behaviour.

### Checkpoint 27 — Convolutional Neural Networks
- Study convolution, filters, pooling and feature extraction.
- **Acceptance:** Train a small image-classification CNN.

### Checkpoint 28 — Sequence Modelling
- Study sequential data and recurrent computation.
- **Acceptance:** Build a basic sequence prediction model.

### Checkpoint 29 — LSTM and GRU
- Understand gating mechanisms and long-range dependencies.
- **Acceptance:** Compare an LSTM or GRU against a basic recurrent baseline.

### Checkpoint 30 — Deep Learning Experimentation
- Study architecture selection, hyperparameter optimisation and ablation studies.
- **Acceptance:** Produce a reproducible comparison of at least three model configurations.

---

# PHASE 07 — TRANSFORMER ARCHITECTURE

**Objective:** Understand and implement the architecture underlying modern language models.

### Checkpoint 31 — Tokenisation
- Study tokenisation, vocabulary construction and special tokens.
- **Acceptance:** Build a tokenizer and implement encode/decode round-trip tests.

### Checkpoint 32 — Embeddings
- Study token embeddings, positional information and embedding dimensions.
- **Acceptance:** Implement an embedding layer with verified tensor shapes.

### Checkpoint 33 — Attention Mechanism
- Understand queries, keys, values and attention scores.
- **Acceptance:** Implement scaled dot-product attention and verify its mathematical behaviour.

### Checkpoint 34 — Multi-Head Attention
- Implement multiple attention heads and combine their outputs.
- **Acceptance:** Verify output dimensions and test multiple attention configurations.

### Checkpoint 35 — Transformer Blocks
- Implement residual connections, normalisation, feedforward layers and causal masking.
- **Acceptance:** Build and test a complete transformer block.
- *Note:* Add a causality test proving that changing a future token never changes earlier outputs.

---

# PHASE 08 — LANGUAGE MODEL DEVELOPMENT

**Objective:** Build and train our own small language model.

### Checkpoint 36 — Language Model Architecture
- Design a decoder-only transformer candidate.
- Define context length, hidden dimensions, layers and attention heads.
- **Acceptance:** Publish a complete architecture specification with parameter-count calculations.
- *Note:* Choose a size that the available compute can actually finish training. Document the compute budget next to the architecture.

### Checkpoint 37 — Text Dataset Pipeline  ⚠ HIDDEN BOTTLENECK
- Research data sources, licensing and data quality.
- Implement deduplication, filtering and train/validation splitting.
- **Acceptance:** Produce a documented, legally usable text dataset.
- *Note:* Budget extra time here. Record source, license and provenance for every data source in a dataset manifest. Do not use data of unknown license.

### Checkpoint 38 — Pretraining Objective
- Implement autoregressive next-token prediction.
- Train using cross-entropy loss.
- **Acceptance:** Demonstrate a measurable reduction in held-out validation loss against an untrained baseline.

### Checkpoint 39 — Training Infrastructure
- Implement checkpointing, experiment tracking, gradient accumulation and recovery.
- Investigate mixed precision and GPU memory management.
- **Acceptance:** Successfully resume an interrupted training run without losing previous progress.

### Checkpoint 40 — Language Model Evaluation
- Evaluate perplexity and held-out text generation.
- Investigate repetition, coherence and generalisation.
- **Acceptance:** Produce an evaluation report with reproducible test prompts and results.

---

# PHASE 09 — GENERAL-PURPOSE AND CODING INTELLIGENCE

**Objective:** Expand the language model into a useful instruction-following assistant.

> **Scope note (see Section 2):** The Track A small model is used to demonstrate *measured improvement over its own pretrained baseline*. The usable assistant is built by fine-tuning a suitably licensed open-weights model. Both results are reported separately and honestly.

### Checkpoint 41 — Instruction Tuning
- Prepare instruction-response datasets.
- Study supervised fine-tuning and training objectives.
- **Acceptance:** Demonstrate improved instruction-following performance over the pretrained baseline.

### Checkpoint 42 — Reasoning and Mathematics
- Study mathematical datasets, structured problem-solving and reasoning evaluation.
- **Acceptance:** Establish reproducible mathematics and logical reasoning benchmarks.

### Checkpoint 43 — Programming Language Understanding
- Study Python, JavaScript, Rust and C++ syntax and semantics.
- Prepare properly licensed code datasets.
- **Acceptance:** Evaluate code comprehension across all four languages.

### Checkpoint 44 — Code Generation
- Train and evaluate code-generation capabilities.
- Include syntax, compilation and execution-based evaluation.
- **Acceptance:** Generate programs that pass a predefined suite of automated tests.
- *Note:* Run generated code only inside a sandbox (no network, resource limits, timeouts).

### Checkpoint 45 — Debugging and Software Engineering
- Study bug localisation, error explanation, testing and code modification.
- **Acceptance:** Demonstrate measurable improvement on a held-out debugging benchmark.

---

# PHASE 10 — PERFORMANCE ENGINEERING

**Objective:** Optimise the model and its computational infrastructure.

> **Ordering rule:** Do not start this phase until a correct Python reference implementation and baseline benchmarks exist. Every optimised component must match the reference numerically within a documented tolerance.

### Checkpoint 46 — C++ Computational Components
- Study tensor operations, memory management and performance profiling.
- Implement selected computational kernels in C++.
- **Acceptance:** Demonstrate numerical equivalence and benchmark against the reference implementation.

### Checkpoint 47 — Rust Infrastructure
- Implement selected infrastructure components in Rust.
- Study concurrency, memory safety and efficient data handling.
- **Acceptance:** Pass integration tests and demonstrate stable operation under concurrent requests.

### Checkpoint 48 — GPU Acceleration
- Study GPU architecture and suitable acceleration technologies.
- Evaluate CUDA-compatible execution where hardware permits.
- **Acceptance:** Verify computational correctness and report measured GPU versus CPU performance.

### Checkpoint 49 — Inference Optimisation
- Study batching, quantisation, caching and efficient memory utilisation.
- **Acceptance:** Compare model quality, latency and memory consumption before and after optimisation.

### Checkpoint 50 — Hybrid Compute Infrastructure
- Separate local experimentation from cloud-scale training.
- Implement distributed or multi-GPU training only when resources justify it.
- **Acceptance:** Successfully execute and recover a documented training experiment using the hybrid environment.

---

# PHASE 11 — AI APPLICATION DEVELOPMENT

**Objective:** Make the trained model usable as an AI assistant.

### Checkpoint 51 — Inference API
- Create a stable model-serving interface.
- Implement request validation and response handling.
- **Acceptance:** Pass API integration and error-handling tests.

### Checkpoint 52 — JavaScript Interface
- Develop a responsive conversation interface.
- Implement streaming responses and conversation controls.
- **Acceptance:** Complete end-to-end conversation tests through the interface.

### Checkpoint 53 — Conversation Context
- Implement conversation history, context windows and truncation strategies.
- Investigate persistent memory as a separate system capability.
- **Acceptance:** Demonstrate correct context handling across multi-turn test conversations.

### Checkpoint 54 — Tool Integration
- Design structured tool calling.
- Integrate selected safe utilities, search and coding tools.
- **Acceptance:** Complete tool-selection, argument-validation and result-handling tests.

### Checkpoint 55 — Multimodal Research
- Study image, audio and other modality-specific encoders.
- Define an architecture for future multimodal extensions.
- **Acceptance:** Produce a technical design and a working small proof of concept for at least one additional modality.

---

# PHASE 12 — EVALUATION, SECURITY AND RELEASE

**Objective:** Validate the system and prepare a responsible release.

### Checkpoint 56 — General Capability Benchmarking
- Evaluate language, reasoning, mathematics, instruction following and coding.
- **Acceptance:** Publish a benchmark report with documented baselines and limitations.

### Checkpoint 57 — Reliability and Security
- Test prompt injection, unsafe tool use, data leakage and malformed inputs.
- Implement safeguards and access controls.
- **Acceptance:** Pass the defined security test suite with no unresolved critical vulnerabilities.

### Checkpoint 58 — Deployment
- Prepare reproducible local and cloud deployment configurations.
- Implement logging, monitoring and model versioning.
- **Acceptance:** Deploy a versioned model and demonstrate successful health checks and recovery procedures.

### Checkpoint 59 — Documentation and Reproducibility
- Document architecture, datasets, training configuration and known limitations.
- Publish model cards and experiment records.
- **Acceptance:** An independent developer can reproduce a documented training experiment.

### Checkpoint 60 — General-Purpose AI Release Candidate
- Integrate the model, inference engine, interface and evaluation infrastructure.
- Complete final acceptance tests.
- **Acceptance:** Produce a release candidate with a reproducible benchmark report, security report, deployment guide and explicit capability limitations.

---

# GLOBAL COMPLETION RULES

1. No checkpoint is complete without evidence.
2. Every model must have a version and reproducible configuration.
3. Training, validation and test datasets must remain appropriately separated.
4. Dataset licensing and provenance must be documented.
5. Performance improvements must be measured rather than assumed.
6. Critical bugs must be resolved before advancing.
7. No model capability may be claimed without evaluation.
8. Preserve previous working versions and training checkpoints.
9. Record compute consumption, training duration and hardware configuration.
10. Maintain a research log throughout the project.

**Added in v1.1:**

11. Gradient checks (Checkpoint 17 utility) must pass for every new differentiable component.
12. Native (C++/Rust) and GPU implementations must be numerically validated against the Python reference before use.
13. Generated code must only be executed inside a sandbox.
14. Every claim in documentation is labelled *established* or *experimental*.
15. Track A (from-scratch) and Track B (fine-tuned) results are always reported separately and never mixed.

---

# RISK REGISTER

| Risk | Impact | Mitigation |
|---|---|---|
| Scope too large for available compute | Project stalls in Phase 08–09 | Two-track strategy; size model to compute budget; fine-tune for the assistant |
| Dataset licensing / quality problems | Blocks Checkpoint 37, legal risk | Manifest with source + license per dataset; use open licensed data only |
| Silent backprop bugs | Corrupt every later result | Checkpoint 17 gradient checks in CI forever |
| Premature optimisation | Wasted effort, hard-to-find bugs | Phase 10 only after reference + benchmarks exist |
| Overfitting / test-set leakage | False confidence | Strict splits; test set touched only for final reports |
| Unsafe execution of generated code or tools | Security incident | Sandboxing, argument validation, Checkpoint 57 test suite |
| Undocumented runs | Irreproducible results | Config + seed + dataset manifest per run; research log |

---

# CHECKPOINT TRACKER (UPDATE AFTER EVERY CHECKPOINT)

| # | Checkpoint | Status | Evidence file | Date |
|---|---|---|---|---|
| 01 | AI Fundamentals | ✅ done | docs/checkpoints/cp01.md | 2026-11-08 |
| 02 | Biological and Artificial Neurons | ✅ done | docs/checkpoints/cp02.md | 2026-11-08 |
| 03 | Neural Network Architectures | ✅ done | docs/checkpoints/cp03.md | 2026-11-08 |
| 04 | Learning Mechanisms | ✅ done | reports/memorization_vs_generalization.png | 2026-11-08 |
| 05 | AI Research Literacy | ✅ done | docs/checkpoints/cp05.md | 2026-11-08 |
| 06–10 | Phase 02 — Mathematical Foundations | ✅ done | docs/checkpoints/cp10.md | 2026-11-08 |
| 11–15 | Phase 03 — NN Implementation | ✅ done | docs/checkpoints/cp15.md | 2026-11-08 |
| 16–20 | Phase 04 — Training Engine | ✅ done | docs/checkpoints/cp20.md | 2026-11-08 |
| 21–25 | Phase 05 — First Trained Models | CP21 in progress (Option B approved) | docs/checkpoints/cp21.md | 2026-11-18 |
| 26–30 | Phase 06 — Deep Learning | ☐ | | |
| 31–35 | Phase 07 — Transformer Architecture | ☐ | | |
| 36–40 | Phase 08 — Language Model | ☐ | | |
| 41–45 | Phase 09 — General and Coding Intelligence | ☐ | | |
| 46–50 | Phase 10 — Performance Engineering | ☐ | | |
| 51–55 | Phase 11 — Application Development | ☐ | | |
| 56–60 | Phase 12 — Evaluation, Security, Release | ☐ | | |

---

# FIRST INSTRUCTION FOR CODEX

Start with **Phase 01, Checkpoint 01**. Create the repository structure from Section 3, create `docs/research_log.md`, and complete Checkpoints 01–05 as short documents under `docs/checkpoints/`. Then proceed to Phase 02. Do not skip ahead. After each checkpoint, report status and wait for confirmation before starting the next phase.

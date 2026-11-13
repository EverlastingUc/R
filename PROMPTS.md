# REUSABLE PROMPTS FOR CODEX

Replace the placeholders in <angle brackets>.

## 1. Start of every new session
```
Read AGENTS.md and the last 3 entries of docs/research_log.md. Tell me in 5 lines: the current checkpoint, what is done, what is pending, any open problems, and your plan for this session. Do not write code yet.
```

## 2. Start the next checkpoint
```
Work on Checkpoint <NN> only. Re-read its objective and Acceptance line in MASTER_SPEC.md. Follow the thinking protocol: restate the task, post a short plan and risks, then write tests first, implement in small verified steps, run the full test suite, write docs/checkpoints/cpNN.md, update docs/research_log.md and the MASTER_SPEC.md tracker, and finish with the report format. Do not start the next checkpoint.
```

## 3. Phase 03 / 04 reminder (NumPy only)
```
Reminder: in this phase use Python and NumPy only. No PyTorch, TensorFlow, JAX, Keras, scikit-learn models or autograd. Frameworks may be used inside tests only as reference checks. Use float64 for correctness tests.
```

## 4. Checkpoint 17 (backpropagation, critical)
```
Implement backpropagation for Checkpoint 17 and ALSO build a reusable gradient_check() utility in src/nn/gradcheck.py using centred finite differences. Compare analytical vs numerical gradients for every layer type and every activation; report the maximum relative error and use a documented tolerance. Add the gradient check to the test suite and CI so it runs for every future layer. Do not move on until all gradient checks pass.
```

## 5. Debugging a failure
```
Something fails: <describe error or behaviour>. Follow the debugging protocol in CODEX_BEHAVIOR.md Section 6: reproduce with the smallest input and a fixed seed, read the actual error, check shapes/dtypes/axes/learning rate/label alignment, isolate components, try overfitting a tiny dataset, change ONE thing at a time. Add a regression test and log the cause in docs/research_log.md. Do not change or weaken existing tests.
```

## 6. Model is not learning
```
The model does not learn. Run the training sanity checklist from CODEX_BEHAVIOR.md Section 6: check initial loss against the expected value (ln(num_classes) for classification), overfit a tiny batch, inspect gradient magnitudes, verify label alignment after shuffling, and try learning rates 10x higher and lower. Report findings before changing the architecture.
```

## 7. Dataset work (Checkpoints 21 and 37)
```
Work on the dataset pipeline. Before downloading or using any data, list candidate sources with licence and provenance and ask me to approve. Create a dataset manifest (docs/dataset_manifest_TEMPLATE.md) for each source. Implement deduplication and filtering, and make sure deduplication happens across train/val/test so nothing leaks. Never use data with unknown or incompatible licence.
```

## 8. Before long training runs / spending money
```
Before starting this training run, post: model size, dataset and manifest, config file, expected duration and cost, hardware, checkpoint and resume strategy, and the metric that will tell us it works. Wait for my approval before starting.
```

## 9. Phase 10 (performance engineering)
```
Before optimising anything, confirm that a correct Python reference implementation and a baseline benchmark exist. Implement the kernel/component, validate it numerically against the reference within a documented tolerance, then benchmark. Report speed, memory and any quality change honestly, including cases where it is not faster.
```

## 10. Code review of your own work
```
Review your own work on Checkpoint <NN> as a strict reviewer. Check: Acceptance fully met with evidence, tests can fail, no hardcoded outputs, no leaked test data, shapes and dtypes asserted, seeds fixed, no secrets, no leftover debug code, docs updated, limitations stated honestly. List every problem you find and fix them.
```

## 11. Wrap up a session
```
Wrap up this session: make sure the full test suite passes, update docs/research_log.md, update the tracker in MASTER_SPEC.md, and post the report in the CODEX_BEHAVIOR.md Section 8 format, including honest limitations and the exact next step.
```

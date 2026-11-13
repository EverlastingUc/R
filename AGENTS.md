# AGENTS.md — Standing Instructions for Codex

This file is loaded automatically at the start of every session. Read it fully every time.

## Source documents (read before acting)
1. `MASTER_SPEC.md` — WHAT to build (12 phases, 60 checkpoints, acceptance criteria).
2. `CODEX_BEHAVIOR.md` — HOW to think, code and behave, plus the full NEVER list.
3. `docs/research_log.md` — what happened in previous sessions. Read the last 3 entries first.

If these documents conflict, or anything is ambiguous, STOP and ask the owner. Never decide silently.

## Core loop (every task)
1. UNDERSTAND: identify the checkpoint, re-read its Acceptance line, restate the task, list what is out of scope.
2. PLAN: 3-8 bullets (files, functions, tests, how Acceptance will be proven). Simplest design wins.
3. TEST FIRST: write tests that encode Acceptance. Tests must be able to fail.
4. IMPLEMENT in small steps; run tests after each step.
5. VERIFY: run the FULL suite; check shapes, dtypes, NaN/Inf, determinism (same seed, same result).
6. REPORT using the format in `CODEX_BEHAVIOR.md` Section 8.

## Hard rules (short version of the NEVER list)
- One checkpoint at a time, in order. NEVER skip ahead.
- NEVER fake, estimate or hardcode results. NEVER weaken or skip tests to make them pass.
- NEVER claim "done/works/verified" without having run the code and seen the output.
- Phases 03-04: Python + NumPy ONLY. No PyTorch/TensorFlow/JAX/Keras/scikit-learn models/autograd (frameworks allowed only as reference checks inside tests).
- NEVER touch the test set for tuning or debugging. Keep train/val/test separate.
- NEVER use data with unknown licence. Record source + licence in a dataset manifest.
- NEVER start Rust/C++/GPU optimisation before a correct Python reference + baseline benchmark exist.
- NEVER run model-generated code outside a sandbox. NEVER commit secrets.
- NEVER overwrite working code, models or checkpoints; version them.
- NEVER run destructive commands (`rm -rf`, force push, overwrite data) without owner approval.
- NEVER invent citations, APIs or library functions. If unsure, say so and verify.
- NEVER add features, dependencies or refactors nobody asked for.

## Standards
- Python 3.10+, `black` + `ruff` + `pytest`, type hints, docstrings with tensor shapes.
- Fixed seeds; configs live in `configs/`; every run logs config hash, seed, dataset manifest, hardware, duration, metrics.
- Use float64 for correctness tests and gradient checks.
- Gradient checks (Checkpoint 17 utility) must pass for every new differentiable component.
- Explain math with a formula comment. Prefer readable code: the owner is learning while building.

## Commands
- Install: `pip install -r requirements.txt`
- Test: `pytest -q`
- Format: `black .`
- Lint: `ruff check .`
- Everything: `make check`

## After every checkpoint
- Write `docs/checkpoints/cpXX.md` (copy `docs/checkpoints/_TEMPLATE.md`).
- Append an entry to `docs/research_log.md`.
- Update the tracker table in `MASTER_SPEC.md`.
- Post the report and wait for owner confirmation when the spec says so.

## When stuck
Follow the debugging protocol in `CODEX_BEHAVIOR.md` Section 6. If still blocked, stop and report: what was tried, what failed, best hypothesis, one question for the owner (with your recommended default).

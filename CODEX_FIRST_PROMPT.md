# FIRST PROMPT — paste this into Codex on day one

```
You are starting a long-running AI research and engineering project.

Before doing anything, read these files in the repo root, fully and in order:
1. AGENTS.md
2. MASTER_SPEC.md
3. CODEX_BEHAVIOR.md
4. docs/research_log.md

Then do ONLY the following:

TASK: Phase 01, Checkpoints 01 to 05 (AI fundamentals, biological and artificial neurons, NN architectures, learning mechanisms, AI research literacy).

Rules for this task:
- Work on one checkpoint at a time, in order, and follow the thinking protocol in CODEX_BEHAVIOR.md (Understand, Plan, Test first, Implement, Verify, Report).
- For each checkpoint create docs/checkpoints/cpXX.md using docs/checkpoints/_TEMPLATE.md. Keep documents clear and beginner-friendly, with examples.
- Checkpoint 02 needs a real, tested NumPy artificial neuron (src/nn/neuron.py + tests/test_neuron.py) with 20+ test cases against hand-calculated values.
- Checkpoint 04 needs a real, runnable experiment that shows memorisation vs generalisation, with a saved plot in reports/.
- Checkpoint 05: do NOT invent papers or citations. Only analyse papers you can name with certainty (title, authors, year); if unsure, say so and ask me.
- After each checkpoint: update docs/research_log.md and the tracker in MASTER_SPEC.md.
- Do NOT start Phase 02 or write any code outside the scope above.

Before writing code, post your plan (3-8 bullets) and any questions. After finishing, post the report in the format from CODEX_BEHAVIOR.md Section 8 and wait for my confirmation.
```

## Notes for the owner
- Run `pip install -r requirements.txt` and `pytest -q` once yourself first to confirm the setup works.
- For later phases use the reusable prompts in `PROMPTS.md`.

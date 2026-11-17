# Valence VQ-A05 — Model tool-call orchestration

## Objective

Allow the local model to request approved application tools through a bounded,
verifiable protocol. Model-provided names and arguments are untrusted data; they
must never become executable source code.

## What was implemented

- Added Ollama function-call parsing to the local model backend.
- Added structured assistant tool-call and tool-result message serialization.
- Connected model turns to the existing typed capability registry and executor.
- Enforced registry membership, application policy, parameter validation,
  approval gates, per-turn tool-call budgets, tool-round limits, and message
  size limits.
- Added a final tool-free model turn when the tool budget is exhausted.
- Connected the Gluons adapter to the user-facing CLI and API behind an
  explicit --enable-web-search flag. Normal startup leaves tools disabled.
- Added an end-to-end regression test for model-requested Gluons retrieval and
  source-attributed tool-result feedback.

## How to run

Install Ollama and pull the starter model once:

```powershell
ollama pull qwen2.5:1.5b-instruct
```

Start Valence with web retrieval explicitly enabled:

```powershell
valence --enable-web-search
```

Or start the local API with web retrieval enabled:

```powershell
valence-api --port 8765 --enable-web-search
```

Without --enable-web-search, the Gluons tool is not registered and the
execution policy permits no tools.

## Evidence

GitHub Actions run
[44126687355](https://github.com/EverlastingUc/R/actions/runs/44126687355)
passed all quality gates:
- Black formatting check: passed.
- Ruff lint check: passed.
- Full test suite: 244 passed in 9.77 seconds.

The tests use mocked model responses and an injected fake search provider, so
they do not require internet access or a running Ollama server.

## Limitations

- Real local inference has not yet been tested against an installed Ollama
  model in this environment.
- The implementation supports the current Ollama tool-call contract only.
- Tool access is limited to application-registered handlers. This is not a
  public plugin marketplace or arbitrary code-execution facility.
- Gluons retrieves bounded snippets and URLs; it does not read whole pages.
- Persistent memory, files, creation modes, and mathematical execution are
  separate future milestones.

## Next step

VQ-A06 — Mathematics subsystem: begin with a safe tokenizer/parser and exact
arithmetic, then expand in tested increments through algebra, geometry,
trigonometry, calculus, linear algebra, probability, statistics, and advanced
mathematical domains.

# VQ-A01 — Local conversational core

## Objective

Create the first real Valence conversation path with a replaceable model backend. Keep model execution separate from conversation state and make the local prototype runnable without paid API keys.

## What was built

- `ChatMessage` and `ChatBackend` provider contract.
- `OllamaChatBackend` using the documented non-streaming `POST /api/chat` endpoint.
- `ValenceEngine` with prompt validation, bounded in-memory context, reset, and error rollback.
- Interactive CLI via `valence` or `python -m proton_i.valence`.
- Mocked backend and HTTP regression tests.

The default starter model is `qwen2.5:1.5b-instruct`. Its weights must be installed separately with Ollama and are not stored in Git.

## How to run

```powershell
ollama pull qwen2.5:1.5b-instruct
python -m pip install -e .
valence
```

Type `/reset` to clear the active conversation and `/exit` to quit.

## Evidence

- GitHub Actions: [run 44118599371](https://github.com/EverlastingUc/R/actions/runs/44118599371).
- Black: 54 files unchanged.
- Ruff: all checks passed.
- Pytest: 173 passed.

## Limitations

- The model itself was not executed in CI; transport tests use a mock HTTP response.
- History exists only in memory for this process.
- No user accounts, file uploads, tool execution, web retrieval, mathematics execution, streaming, or multimodal generation yet.

## Next step

VQ-A02 — local JSON HTTP API. Keep it loopback-only and verify session isolation, request validation, and reset behavior.

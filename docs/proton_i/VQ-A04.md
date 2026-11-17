# VQ-A04 — Gluons source-attributed retrieval tool

## Objective

Make Gluons available as a narrowly scoped Valence capability while preserving the separation between website retrieval and the conversational model.

## What was built

- `create_gluons_tool()` creates a typed `search_web` ToolDefinition.
- `register_gluons_tool()` registers it with the Valence CapabilityRegistry.
- Input is a single text query limited to 500 characters.
- At most five results are requested and returned.
- Titles are limited to 300 characters, snippets to 1,000, and source URLs to 2,048 characters.
- Only absolute HTTP(S) source URLs are accepted; URLs with embedded credentials or invalid ports are discarded.
- Results include title, snippet and source URL.
- Every response marks retrieved text as `untrusted_web_content` and carries a reminder not to treat snippets as instructions.
- Provider `SearchError` becomes a sanitized, application-authored `ToolHandlerError`. Generic handler exceptions remain redacted.

## Permission model

Registering Gluons does not grant it execution permission. Its schema is visible only when the current ExecutionPolicy explicitly contains `search_web`, and the ToolExecutor performs the allowlist and argument validation before invoking it. No external tool is enabled by default.

## Evidence

GitHub Actions: [run 44123523142](https://github.com/EverlastingUc/R/actions/runs/44123523142).

- Black: 61 files unchanged.
- Ruff: all checks passed.
- Pytest: 224 passed in 9.32 seconds.
- Tests use an injected deterministic search provider; CI does not make live web requests.

## Current limitations

The local model does not yet emit tool calls or automatically invoke the executor. This milestone makes Gluons available as a registered, policy-controlled retrieval tool, but it is not yet part of the end-to-end chat turn. The next milestone will add a bounded model/tool interaction loop.

## Next step

VQ-A05 — connect Ollama tool-call responses to the CapabilityRegistry and ToolExecutor. Validate tool names and arguments, enforce the active ExecutionPolicy, cap tool rounds and response sizes, and never execute model-generated source code.

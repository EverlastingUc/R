# VQ-A03 — Typed capability registry and guarded tool executor

## Objective

Create a safe, provider-neutral tool contract before connecting Gluons, mathematics, files, or creation workflows to Valence.

## What was built

- `ToolParameter` defines typed JSON arguments with required/optional status, enums, numeric bounds, and string/array/object size limits.
- `ToolDefinition` binds a stable tool name and description to an application-owned handler.
- `CapabilityRegistry` stores definitions and exposes schemas only when the current policy explicitly permits a tool.
- `ExecutionPolicy` is deny-by-default and carries allowed-tool names plus argument/output size limits.
- `ToolExecutor` checks tool registration, enablement, allowlist membership, optional approval, argument structure and types, JSON validity, and payload sizes before invoking a handler.
- `ToolResult` returns structured success/denied/invalid/failed states.
- Handler exception messages are redacted; outputs must be JSON serializable and within the configured output limit.

## Security boundary

Only functions written and registered by the application can run. A model-supplied tool name and arguments are untrusted data; they are matched against a registered definition and validated before handler invocation. The executor never evaluates Python, shell commands, imports, or model-generated code.

The default policy permits no tools. Tools marked `requires_approval` cannot execute until the application supplies explicit approval. This approval flag must come from the application/UI, not from model output.

Handlers are trusted application code and must set native timeouts for any blocking network or filesystem operation. This milestone does not implement a public plugin marketplace, unrestricted tool calling, or a sandbox for untrusted code.

## Evidence

GitHub Actions: [run 44122468478](https://github.com/EverlastingUc/R/actions/runs/44122468478).

- Black: 58 files unchanged.
- Ruff: all checks passed.
- Pytest: 216 passed in 10.01 seconds.
- Regression cases cover deny-by-default policy, catalog filtering, duplicate and malformed definitions, input type/range/enum validation, approval gating, disabled tools, unknown tool names, handler failure redaction, non-JSON outputs, payload limits, huge integers, circular objects, and invalid allowlists.

## Current limitations

No external tools are registered by default. The local chat model does not yet send tool calls to this executor, and the conversation engine does not automatically route model output to it. This is the safe tool foundation, not a working web-search or math tool yet.

## Next step

VQ-A04 — register Gluons as a retrieval-only Valence tool. Keep its search query limits, source URL attribution, and graceful failure handling. It should be visible to the model only under an explicit execution policy.

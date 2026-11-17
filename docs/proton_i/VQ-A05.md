# VQ-A05 — Model tool-call orchestration

## Objective

Connect the local language model's structured function/tool-call protocol to Valence's application-owned CapabilityRegistry and ToolExecutor. A model may request a tool, but it cannot choose arbitrary code or bypass application permissions.

## What was built

- The Ollama backend supports non-streaming `complete_turn()` responses containing assistant text and/or structured `tool_calls`.
- Tool requests are parsed as untrusted data with bounded call counts and argument payload sizes.
- Provider tool schemas are constructed only from registry definitions allowed by the current ExecutionPolicy. Application-only metadata such as `requires_approval` is stripped from the provider schema.
- The engine sends assistant tool-call messages to Ollama, executes the registered handler through ToolExecutor, appends a structured `role: tool` result, and asks the model for its next response.
- Tool execution is bounded by maximum tool rounds and total calls. After the budget is consumed, Valence asks for a final tool-free response.
- Unknown, disabled, denied, malformed, approval-gated, or over-budget calls return structured failures and do not run a handler.
- CLI and local API accept `--enable-web-search`. Only with this explicit flag do they register Gluons and allow the `search_web` tool. Default startup has no external tool permissions.
- The API capability route distinguishes `not_integrated`, `disabled_by_policy`, `model_tool_calling_not_supported`, and `enabled` for web search.

## How to use

Pull the local model once:

```powershell
ollama pull qwen2.5:1.5b-instruct
```

Start normal local chat with no external search:

```powershell
valence
```

Enable Gluons search explicitly for that session:

```powershell
valence --enable-web-search
```

For the loopback API, normal mode is:

```powershell
valence-api --port 8765
```

Enable search for API sessions explicitly:

```powershell
valence-api --port 8765 --enable-web-search
```

The API remains bound to `127.0.0.1`. This is a developer-preview local service, not a public internet endpoint.

## Safety boundary

- Only application-registered Python handlers can execute.
- Tool names and arguments returned by the model are untrusted.
- Tool schemas are visible only when allowed by the application policy.
- Search is opt-in; default policy is deny-all.
- Gluons results remain marked `untrusted_web_content`; source URLs are retained, and retrieved text is not treated as instructions.
- Model-generated Python, shell, JavaScript, or other source code is never evaluated or executed by this loop.
- Tool rounds, call counts, message size, arguments and tool-result output are bounded.

## Evidence

GitHub Actions: [run 44127457691](https://github.com/EverlastingUc/R/actions/runs/44127457691).

- Black: 62 files unchanged.
- Ruff: all checks passed.
- Pytest: 246 passed in 17.65 seconds.
- Tests include Ollama request/response parsing, malformed tool-call rejection, tool-role message serialization, end-to-end API-to-Gluons search with source attribution, deny-policy behavior, approval gates, tool budgets, and redacted failures.

## Limitations

- CI uses fake model and search providers. A real local Ollama model has not yet been installed or exercised in this environment; the quality and correctness of model-selected tool calls must be assessed on the owner's local runtime.
- Only the Ollama function-call protocol is implemented.
- Search is snippet-based retrieval, not full-page browsing.
- No public plugin marketplace, cloud provider router, persistent memory, math solver, file understanding, image/video generation, coding sandbox, or DesignArena-style creation pipeline is completed yet.

## Next step

VQ-A06 — implement Valence's mathematics subsystem, starting with a safe tokenizer/parser and exact arithmetic. Do not use Python `eval()` or `exec()`. Expand in tested modules toward algebra, geometry, trigonometry, calculus, linear algebra, probability/statistics, numerical methods, and the advanced domains in `VALENCE_MATH_SPEC.md`.

# VQ-A02 — Loopback conversation HTTP API

## Objective

Expose Valence's local conversational engine through a stable JSON interface so a future web interface can use the same engine as the terminal client.

## What was built

- A threaded HTTP server in `src/proton_i/valence/api.py`.
- `GET /health` returns service liveness and the active in-memory session count.
- `GET /v1/capabilities` returns explicit implemented/not-integrated capability status.
- `POST /v1/chat` creates a random session on the first message and accepts the returned `session_id` for follow-up messages.
- `POST /v1/reset` clears a conversation while retaining its session identifier.
- Each session has a separate conversation engine and lock, preventing concurrent turns from corrupting one session's history.
- Request JSON is limited to 16 KB; messages are limited to 12,000 characters; the API holds at most 64 sessions.
- The server can bind only to an IPv4 loopback address. Browser requests that include an `Origin` header must use the same loopback origin and port.
- Prompts and session identifiers are not written to the HTTP server logs.
- The installed command is `valence-api --port 8765`.

## How to run

Install the package and pull the local model once:

```powershell
python -m pip install -e .
ollama pull qwen2.5:1.5b-instruct
```

Start the API:

```powershell
valence-api --port 8765
```

In a second PowerShell window, send the first message:

```powershell
$first = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/chat -ContentType "application/json" -Body (@{ message = "Explain neural networks." } | ConvertTo-Json)
$session = $first.session_id
$first.message
```

Continue the same conversation:

```powershell
$next = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/chat -ContentType "application/json" -Body (@{ message = "Give me a simple example."; session_id = $session } | ConvertTo-Json)
$next.message
```

Reset the conversation:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/reset -ContentType "application/json" -Body (@{ session_id = $session } | ConvertTo-Json)
```

## Evidence

GitHub Actions: [run 44120543232](https://github.com/EverlastingUc/R/actions/runs/44120543232).

- Black: 56 files unchanged.
- Ruff: all checks passed.
- Pytest: 188 passed in 9.23 seconds.
- Integration tests use a real ephemeral loopback HTTP server and a deterministic fake model backend.
- Tests cover route responses, session creation, follow-up history, reset, session isolation, session limits, malformed JSON, wrong content type, request-size limits, origin restrictions, non-loopback bind rejection, and sanitized backend failures.

## Security and limitations

This is a local developer-preview API, not a public production server. It binds only to `127.0.0.1`, has no account authentication, has no HTTPS/public-hosting setup, and does not support cross-origin browser clients or CORS preflight. A future frontend should be served from the same local origin or use a separately designed authenticated gateway.

Sessions are in memory, capped at 64, and disappear when the API process stops. There is no session expiry or persistent store yet. CI verifies the HTTP interface with a fake backend; it does not download or run an Ollama model, so real local inference still needs owner-side verification.

## Next step

VQ-A03 — define a typed capability registry and tool contract with explicit permission checks, structured results, timeouts, and output limits. Do not enable arbitrary code execution or an unrestricted tool executor.

# Entropy

Entropy is the from-scratch neural-network research track of this repository. Proton-I is the product track built alongside it. Its stages are Gluons (web retrieval), Valence Quarks (general-purpose assistant), and Sea Quarks (future capability, not yet specified).

## Project setup

Open PowerShell in the repository root:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pip install -e .
.\\.venv\\Scripts\\python.exe -m pytest -q
```

## Proton-I — Gluons

Gluons is the intentionally small web-retrieval component. It searches websites, extracts relevant result snippets, and returns concise plain-text answers with source URLs. It is not the language model and does not calculate mathematics.

```powershell
gluons "What is CSS Flexbox?"
```

See [the Gluons guide](docs/proton_i/GLUONS.md).

## Proton-I — Valence Quarks

Valence is the main conversational assistant. The first working version runs a local model through Ollama, keeps bounded in-memory conversation history, and provides an interactive terminal chat. It does not yet claim the full ChatGPT or DesignArena feature set.

1. Install [Ollama](https://ollama.com/download).
2. Download the starter model:

```powershell
ollama pull qwen2.5:1.5b-instruct
```

3. Start chatting from the repository root:

```powershell
valence
```

Enable optional website search explicitly:

```powershell
valence --enable-web-search
```

Or run it as a Python module:

```powershell
python -m proton_i.valence
```

Use `/reset` to clear the current conversation and `/exit` to quit. The local model must be downloaded separately; model weights are not committed to Git.

You can override the model and host without changing source code:

```powershell
$env:VALENCE_MODEL = "qwen2.5:1.5b-instruct"
$env:VALENCE_OLLAMA_HOST = "http://127.0.0.1:11434"
valence
```

The current Valence prototype has local chat, a loopback API, safe exact basic arithmetic, bounded model tool-call orchestration, a typed tool registry, and a Gluons retrieval adapter. Basic arithmetic is enabled by default; website search remains opt-in with `--enable-web-search`. Retrieved snippets are marked as untrusted web content. Algebra, calculus, geometry, persistent memory, file uploads, coding sandbox, image/video generation, creation modes and model comparisons remain planned milestones.

See [the Valence product specification](docs/proton_i/VALENCE_PRODUCT_SPEC.md) and [its mathematics subsystem specification](docs/proton_i/VALENCE_MATH_SPEC.md).

## Entropy research track

- `neural_network.py` — compact binary-classification network.
- `src/nn/` — NumPy neurons, layers, feedforward network, losses, backpropagation, optimizers, trainer and numerical utilities.
- `experiments/` — reproducible learning experiments.
- `configs/` — versioned experiment settings.
- `reports/` — measured experiment reports and saved checkpoints.
- `docs/checkpoints/` — educational notes and evidence for the Entropy checkpoints.

Run the full research and application test suite with:

```powershell
python -m pytest -q
```

Entropy's small research networks are not represented as general-purpose language models. Valence currently uses an independently installed local language model through Ollama.

## Local Valence API (developer preview)

Start the local HTTP service in a second PowerShell window:

```powershell
valence-api --port 8765
```

Enable the same optional Gluons search integration with
`valence-api --port 8765 --enable-web-search`.

The API binds only to `127.0.0.1`. It currently exposes `GET /health`,
`GET /v1/capabilities`, `POST /v1/chat`, and `POST /v1/reset`.
Chat requests and responses are JSON; the first chat request creates a random
in-memory `session_id`, which must be sent with follow-up requests. Sessions
are not persistent and are cleared when the process exits.

Example:

```powershell
$body = @{ message = "Explain a neural network in simple terms." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/chat -ContentType "application/json" -Body $body
```

See the [VQ-A02 API report](docs/proton_i/VQ-A02.md) and
[VQ-A05 tool-orchestration report](docs/proton_i/VQ-A05.md).

Do not expose this developer-preview server to the public internet. It has no
account authentication or durable session storage.

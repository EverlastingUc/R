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

The current Valence prototype has local chat, a loopback API, a deterministic number-system calculator, bounded model tool-call orchestration, a typed tool registry, and a Gluons retrieval adapter. The calculator supports exact rational arithmetic, bounded integer powers, integer modulo, abs, GCD/LCM, ratios, percentages, exact rational square roots, and approximate irrational square roots. It is enabled by default; Gluons search remains opt-in with `--enable-web-search`. Logarithms, symbolic algebra, equations, geometry, trigonometry, calculus, persistent memory, file uploads, coding sandbox, image/video generation, creation modes and model comparisons remain planned milestones.

See [the Valence product specification](docs/proton_i/VALENCE_PRODUCT_SPEC.md) and [its mathematics subsystem specification](docs/proton_i/VALENCE_MATH_SPEC.md).

## Entropy research track

- `neural_network.py` — compact binary-classification network.
- `src/nn/` — NumPy neurons, layers, feedforward network, regression and softmax-classification training, losses, backpropagation, optimizers, numerical utilities and a reproducible UCI Iris loader.
- `experiments/` — reproducible learning experiments, including the CP22 UCI Iris classifier.
- `configs/` — versioned experiment settings.
- `reports/` — measured experiment reports and saved checkpoints.
- `docs/checkpoints/` — educational notes and evidence for the Entropy checkpoints.

Run the full research and application test suite with:

```powershell
python -m pytest -q
```

Entropy's NumPy research networks are not represented as general-purpose language models. The experimental `entropy_lm` package contains byte and trainable BPE tokenizers, reproducible text and conversation corpus preparation, a decoder-only Transformer, resumable pretraining and supervised fine-tuning, held-out evaluation, and a UTF-8-safe token-streaming local chat interface with per-layer KV-cache decoding. Install the optional PyTorch dependency with `python -m pip install -e ".[llm]"`.

The local chat command streams generated response text by default and preserves the first system message and latest user turn when packing long conversation history; pass `--no-stream` to print only after generation completes. The model also exposes `generate_stream` for token-level consumers. The available commands are `entropy-prepare-corpus`, `entropy-train`, `entropy-evaluate`, `entropy-prepare-chat`, `entropy-chat-train`, `entropy-evaluate-chat`, `entropy-benchmark-chat`, and `entropy-chat`. BPE is fitted only on training partitions. Conversation SFT masks system and user prompt tokens from the loss and scores assistant responses. Training checkpoints preserve model, optimizer, scheduler, progress, and RNG state; both base training and SFT support `--resume` and `--max-steps-this-run`. See [LM06](docs/checkpoints/LM06.md), [LM07](docs/checkpoints/LM07.md), [LM08](docs/checkpoints/LM08.md), [LM09](docs/checkpoints/LM09.md), [LM10](docs/checkpoints/LM10.md), [LM11](docs/checkpoints/LM11.md), [LM12](docs/checkpoints/LM12.md), [LM13](docs/checkpoints/LM13.md), [LM14](docs/checkpoints/LM14.md), and [LM15](docs/checkpoints/LM15.md), plus [LM16](docs/checkpoints/LM16.md), [LM17](docs/checkpoints/LM17.md), and [LM18](docs/checkpoints/LM18.md), plus [LM19](docs/checkpoints/LM19.md), for the milestone details.

This remains an experimental architecture; no ChatGPT-class model or general-purpose Entropy checkpoint has been trained. The chat interface is a local prototype, not evidence of instruction-following quality. Valence currently uses an independently installed local language model through Ollama.

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

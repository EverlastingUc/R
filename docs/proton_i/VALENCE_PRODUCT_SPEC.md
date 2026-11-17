# Valence Quarks — Product and Engineering Specification

**Product role:** Proton-I's user-facing general-purpose AI assistant  
**Product inspiration:** ChatGPT-style conversation and DesignArena-style creative workspace  
**Status:** Owner-approved direction; initial local-chat vertical slice in progress

## 1. Product definition

Valence is not merely a mathematics calculator. It is the main conversational intelligence and task-orchestration layer of Proton-I. Its long-term goal is a capable, general-purpose assistant that can hold multi-turn conversations, understand requests, use approved tools, work with user-provided files, create digital artifacts, and explain its work.

The product has two complementary surfaces:

1. **Assistant:** a general conversational interface for questions, explanations, planning, learning, analysis, and problem solving.
2. **Creation workspace:** a prompt-driven studio for producing and iterating on digital outputs. DesignArena is a reference for the breadth of creation modes and multi-model comparison, not a product to copy or a guarantee that Valence already supports these modes.

All mathematics, including elementary arithmetic, is one Valence capability. The existing [Valence mathematics specification](VALENCE_MATH_SPEC.md) remains the detailed target for that subsystem; it does not define the entire Valence product.

## 2. Product principles

- **Model-agnostic:** keep the conversation engine separate from model providers so local and hosted models can be compared or replaced.
- **Tool-grounded:** calculations, web retrieval, file operations, and generated artifacts must be handled by explicit tools with validated inputs and observable results.
- **Honest:** do not claim browsing, coding, file creation, image generation, or execution unless the corresponding tool actually ran and returned evidence.
- **Incremental:** publish a capability matrix. A mode is not supported until it has an implementation, tests, and a working user path.
- **Privacy-aware:** the first backend is local inference. Remote providers require explicit configuration and a documented data-flow decision.
- **Secure by default:** generated code is never executed directly; tool permissions are narrow; untrusted model output is validated before display or execution.
- **Useful on limited hardware:** start with a small quantized local model and allow a provider to be swapped later without rewriting the product.

## 3. Target capability catalogue

The catalogue is the product destination, not the current implementation claim.

### Conversation and intelligence
- Multi-turn chat with bounded conversation context.
- Explanations, summaries, translation, brainstorming, planning, and tutoring.
- Mathematics: arithmetic, algebra, geometry, trigonometry, calculus, linear algebra, probability, statistics, and advanced extensions.
- Programming assistance: explanation, code drafting, debugging, and test generation, with sandboxed execution as a later capability.
- Web research through Gluons, with source links and retrieval boundaries.
- File understanding and document-grounded answers.

### Creative workspace
- Websites and frontend applications.
- Images and image editing.
- Games and interactive experiences.
- PDFs, documents, presentations, and spreadsheets.
- SVG, UI components, and data visualisations.
- 3D assets and mobile application prototypes.
- Audio/text-to-speech and video-related workflows.

### Multi-model evaluation
- Provider/model registry and capability metadata.
- Side-by-side generation for the same prompt.
- Blind model labels during preference voting.
- User feedback and reproducible leaderboard methodology.
- Cost, latency, quality, and capability comparisons with explicit measurement.

No provider ranking or leaderboard score may be invented. Human-voted ranking is a later, separately instrumented feature.

## 4. Initial architecture

The first implementation should establish stable boundaries before adding more capabilities:

- `ValenceEngine`: conversation state, prompt validation, bounded context, reset.
- `ChatBackend`: provider contract for text generation.
- `OllamaChatBackend`: initial local inference adapter, using Ollama's non-streaming `POST /api/chat` interface.
- `CapabilityRegistry` (planned): declares tools and modes without granting execution permissions automatically.
- `ToolExecutor` (planned): validates typed arguments, enforces permission policy, timeouts and output limits.
- `ConversationStore` (planned): durable user/session history with access controls and deletion semantics.
- `CreationWorkspace` (planned): task-specific artifact pipelines; each mode has a defined input/output format.
- `ModelRouter` (planned): selects among configured providers based on capability, user choice, latency and cost constraints.

The local model adapter is restricted to a loopback host in the first release. Remote model APIs, API keys and cloud routing are not silently enabled.

## 5. First runnable path

The first development slice is a real local text conversation, not a scripted answer:

1. Install Ollama using its official installation instructions.
2. Pull the Qwen2.5 1.5B instruction model: `ollama pull qwen2.5:1.5b-instruct`.
3. Install this repository with `python -m pip install -e .`.
4. Run `python -m proton_i.valence` or the installed `valence` command.
5. Type messages; use `/reset` to clear the in-memory conversation and `/exit` to quit.

The model is a local runtime dependency; its weights are not committed to Git. The default model is a compact starting point, not a claim of ChatGPT-level quality. The Qwen2.5 1.5B quantized artifact is listed at approximately 986 MB on the Ollama library page and is marked Apache 2.0 there. Actual RAM use also includes the runtime, context, and operating system.

## 6. Current implementation boundary

The initial slice supports:
- One local Ollama model.
- Plain-text, multi-turn chat.
- In-memory context with a fixed upper bound.
- Explicit reset and graceful provider error messages.
- Mocked HTTP and orchestration tests that do not require a running model or network.

It does not yet support:
- Web search orchestration (Gluons remains separate until a safe tool adapter is added).
- Mathematics execution (math subsystem implementation is a later Valence milestone).
- Persistent accounts, cloud sync, memory, uploads, or file parsing.
- Tool/function calling, code execution, image/video/audio generation, or the creation modes above.
- Model comparisons, voting, or leaderboards.
- A trained Proton-I language model. This stage calls an external local model runtime; Entropy's research network is not yet a capable conversational LLM.

## 7. Delivery sequence

1. **VQ-A01 — Local conversational core:** backend contract, Ollama adapter, bounded conversation engine, CLI, tests.
2. **VQ-A02 — Application API:** typed request/response schema, local HTTP service, health/readiness, request limits.
3. **VQ-A03 — Tool contract:** typed capability registry, policy checks, structured tool results; no arbitrary code execution.
4. **VQ-A04 — Gluons integration:** retrieval tool returning source-attributed snippets.
5. **VQ-A05 — Mathematical subsystem:** safe expression parsing, exact arithmetic, then the full mathematics roadmap in `VALENCE_MATH_SPEC.md`.
6. **VQ-A06 — Conversation persistence:** storage schema, per-user isolation, export/delete, retention.
7. **VQ-A07 — File workflows:** upload limits, content-type checks, safe parsers, citations to supplied sources.
8. **VQ-A08 — Frontend chat workspace:** streaming, markdown-safe rendering, conversation controls, accessibility.
9. **VQ-A09 — Coding sandbox:** isolated execution with no network, CPU/memory limits, timeouts and filesystem boundaries.
10. **VQ-A10 — Creation modes:** implement each output type as a separately tested artifact pipeline.
11. **VQ-A11 — Multi-model arena:** provider routing, blind pairwise comparisons, preference data and leaderboard methodology.
12. **VQ-A12 — Evaluation and release:** quality benchmarks, safety tests, latency/cost reports, privacy and deployment review.

Each milestone gets its own meaningful commit(s), regression tests, changelog/research-log entry, and verified CI run. Do not create empty commits to inflate the count.

## 8. Definition of done

The MVP is not described as a ChatGPT replacement. It is a local chat prototype that calls a configured local model. Product claims must match the capability matrix and measured evaluations. A release is eligible only when its supported flows have automated tests, explicit limitations, privacy/security review, and reproducible evidence.

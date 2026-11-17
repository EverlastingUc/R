# Valence Quarks — Product and Engineering Specification

**Product role:** Proton-I's user-facing general-purpose AI assistant  
**Product inspiration:** ChatGPT-style conversation and DesignArena-style creative workspace  
**Status:** VQ-A01 through VQ-A04 implemented and CI-verified; broader assistant capabilities remain planned

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
- `CapabilityRegistry` (implemented): stores application-owned tool definitions and exposes only schemas allowed by the current execution policy.
- `ToolExecutor` (implemented): validates typed arguments, applies deny-by-default allowlists and explicit approval gates, enforces input/output size limits, and returns structured redacted results. Each trusted handler must enforce its own native timeout for blocking operations.
- `GluonsSearchTool` (implemented): registers Gluons retrieval as a bounded, source-attributed tool whose snippets are explicitly labeled untrusted web content.
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

The current prototype supports:
- One local Ollama model through the provider-neutral `ChatBackend` contract.
- Plain-text, multi-turn chat through both the CLI and a local HTTP API.
- Bounded in-memory conversation history in the CLI.
- A maximum of 64 in-memory API sessions, each identified by a random session token.
- Explicit conversation reset and graceful provider/API error responses.
- `GET /health` and `GET /v1/capabilities`.
- `POST /v1/chat` and `POST /v1/reset`, using JSON request/response bodies.
- HTTP request bodies capped at 16 KB and chat messages capped at 12,000 characters.
- Server binding restricted to IPv4 loopback (`127.0.0.1`); browser requests with an Origin header must be same-origin loopback requests.
- API tests using an actual ephemeral loopback HTTP server and a deterministic fake model backend. These tests need no running Ollama instance or internet connection.

It does not yet support:
- Authentication, user accounts, durable sessions, session expiry, or database persistence. API sessions disappear when the server process exits.
- Public-network hosting, HTTPS termination, cross-origin browser access, or CORS preflight handling. Keep the developer-preview API on loopback.
- Automatic model-driven web search is not yet implemented. Gluons has a registered, application-callable adapter, but the application must explicitly allow `search_web`; the model does not yet invoke tools automatically.
- Mathematics execution (math subsystem implementation is a later Valence milestone).
- Persistent accounts, cloud sync, memory, uploads, or file parsing.
- Model-driven tool calling is not yet connected to the executor. Gluons can be registered by the application, but it is not auto-invoked by the model; the application must explicitly allow `search_web` in an execution policy. No external tools are enabled automatically.
- Arbitrary code execution, image/video/audio generation, or the creation modes above.
- Model comparisons, voting, or leaderboards.
- A trained Proton-I language model. This stage calls an external local model runtime; Entropy's research network is not yet a capable conversational LLM.

## 7. Delivery sequence

1. **VQ-A01 — Local conversational core:** *implemented* backend contract, Ollama adapter, bounded conversation engine, CLI, tests.
2. **VQ-A02 — Application API:** *implemented* typed JSON routes, loopback-only HTTP service, health/capabilities, session handling and request limits.
3. **VQ-A03 — Tool contract:** *implemented* — typed capability registry, deny-by-default policy checks, approval gates, structured tool results, and output limits; no arbitrary code execution.
4. **VQ-A04 — Gluons integration:** *implemented adapter* — bounded, source-attributed retrieval with untrusted-content labeling and explicit policy permission; automatic model invocation is still pending.
5. **VQ-A05 — Model tool-call orchestration:** connect Ollama's tool-call response protocol to the registry/executor, validate each call, cap tool rounds, and return structured tool results to the model.
6. **VQ-A06 — Mathematical subsystem:** safe expression parsing, exact arithmetic, then the full mathematics roadmap in `VALENCE_MATH_SPEC.md`.
7. **VQ-A07 — Conversation persistence:** storage schema, per-user isolation, export/delete, retention.
8. **VQ-A08 — File workflows:** upload limits, content-type checks, safe parsers, citations to supplied sources.
9. **VQ-A09 — Frontend chat workspace:** streaming, markdown-safe rendering, conversation controls, accessibility.
10. **VQ-A10 — Coding sandbox:** isolated execution with no network, CPU/memory limits, timeouts and filesystem boundaries.
11. **VQ-A11 — Creation modes:** implement each output type as a separately tested artifact pipeline.
12. **VQ-A12 — Multi-model arena:** provider routing, blind pairwise comparisons, preference data and leaderboard methodology.
13. **VQ-A13 — Evaluation and release:** quality benchmarks, safety tests, latency/cost reports, privacy and deployment review.

Each milestone gets its own meaningful commit(s), regression tests, changelog/research-log entry, and verified CI run. Do not create empty commits to inflate the count.

## 8. Definition of done

The MVP is not described as a ChatGPT replacement. It is a local chat prototype that calls a configured local model. Product claims must match the capability matrix and measured evaluations. A release is eligible only when its supported flows have automated tests, explicit limitations, privacy/security review, and reproducible evidence.

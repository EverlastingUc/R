# Proton-I — Three Development Stages

Proton-I is the planned AI assistant in this repository. Its development is organized into three stages named Gluons, Valence Quarks, and Sea Quarks. These are project stage names, not particle-physics simulations.

The updated owner-approved product direction is a general-purpose conversational AI with a prompt-driven creation workspace inspired by DesignArena. The original HTML/CSS-only assistant scope is superseded.

## Stage 1 — Gluons

Gluons is intentionally narrow:
- Search websites and extract relevant information.
- Return short, plain-text, source-attributed answers.
- Report search failures or empty results honestly.
- Do not calculate mathematics or act as the conversational language model.

The current implementation is a retrieval-based baseline, not a trained generative language model. It does not use Entropy's neural-network weights to answer language questions.

## Stage 2 — Valence Quarks

Valence is Proton-I's main general-purpose conversational assistant and task-orchestration layer. It owns multi-turn conversation, model-provider integration, reasoning workflows, mathematical problem solving, programming assistance, document understanding, and the future creative workspace.

The target creation workspace includes websites, apps, games, images, PDFs, spreadsheets, presentations, SVG/UI components, data visualisations, 3D assets, and future audio/video workflows. These are planned capability areas, not current feature claims.

Valence is designed to be model-agnostic: the conversation engine is separate from inference providers. The first working route uses a local Ollama model. The system will grow through tested milestones covering an API, tool permissions, Gluons retrieval integration, mathematics, persistent conversations, file workflows, frontend, sandboxed coding, creation modes, and multi-model comparison.

See [VALENCE_PRODUCT_SPEC.md](VALENCE_PRODUCT_SPEC.md) for the product direction and implementation roadmap. The detailed mathematics target remains in [VALENCE_MATH_SPEC.md](VALENCE_MATH_SPEC.md); mathematics is one subsystem, not the full Valence product.

## Stage 3 — Sea Quarks

Sea Quarks remains reserved for a later Proton-I capability. Its exact role has not yet been defined. It must not duplicate Gluons' website retrieval or Valence's conversational and mathematical responsibilities without an explicit redesign.

## Relationship to Entropy

`R` is the existing Entropy neural-network research repository. Its from-scratch NumPy research track remains distinct from the usable assistant track. Valence's first local-chat adapter invokes a separately installed local model runtime; it does not pretend the existing small Entropy network is already a capable chat model.

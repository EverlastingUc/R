# Proton-I — Three Development Stages

Proton-I is the planned assistant built in this repository. Its development is organized into three stages named Gluons, Valence Quarks, and Sea Quarks. These are project stage names, not particle-physics simulations.

The overall product direction is a beginner-friendly assistant for HTML and CSS, with no JavaScript generation in the initial Proton-I plan. It should answer basic questions and use website search to retrieve supporting information.

## Stage 1 — Gluons

Gluons is the first small version. Its answer behavior is intentionally basic:

- Give a short answer, normally no more than one paragraph.
- Return plain text only: no Markdown headings, lists, tables, or code fences.
- Search the public web and include a source URL when a result is found.
- Report clearly when search is unavailable or returns no useful results.
- Keep JavaScript, TypeScript, Node.js, and framework requests out of scope.
- Do not claim detailed code generation or advanced reasoning.

The current implementation is a retrieval-based baseline, not a trained generative language model. It reads a search-result snippet and selects a concise relevant sentence. It does not yet train or use the existing Entropy neural-network weights for language answering.

## Stage 2 — Valence Quarks

This is a later Proton-I development stage. Its exact capabilities and acceptance criteria have not yet been specified by the owner. Do not begin it or invent its scope while implementing Gluons.

## Stage 3 — Sea Quarks

This is the final planned Proton-I development stage. Its exact capabilities and acceptance criteria have not yet been specified by the owner. Do not begin it or invent its scope while implementing Gluons.

## Relationship to Entropy

`R` is the existing Entropy neural-network research repository. Gluons is implemented inside this repository under `src/proton_i/gluons/` and reuses its Python project packaging. Existing Entropy neural-network code, experiment reports, and checkpoints remain intact.

Future stages may integrate trained components from Entropy, but no language model training or language-answering weights are claimed for Gluons 0.1.

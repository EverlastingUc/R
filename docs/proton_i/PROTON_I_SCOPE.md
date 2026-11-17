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

Valence is Proton-I's complete mathematics capability. It owns mathematical knowledge and computation from basic arithmetic (including direct questions such as `2 + 2`) through algebra, geometry, trigonometry, linear algebra, probability, statistics, calculus, differential equations, discrete mathematics, number theory, numerical methods, optimisation, and advanced mathematical extensions.

Valence is intended to be a deterministic, verifiable mathematics engine. It will parse problems, select an appropriate method, compute exact or approximate results, verify answers, and explain working when useful. It must not rely on a language model's unverified text as the numerical authority.

The detailed scope, proposed architecture, correctness requirements, and implementation milestones are defined in [VALENCE_MATH_SPEC.md](VALENCE_MATH_SPEC.md). The scope is comprehensive, but implementation will be incremental; a domain is not considered supported until its tests and evidence exist.

## Stage 3 — Sea Quarks

Sea Quarks is the final planned Proton-I development stage. Its capabilities have not yet been specified. Under the current stage boundary, mathematics—including elementary arithmetic—belongs to Valence, so Sea Quarks must not duplicate the mathematics engine. Define Sea Quarks separately when its purpose is decided.

## Relationship to Entropy

`R` is the existing Entropy neural-network research repository. Gluons is implemented inside this repository under `src/proton_i/gluons/` and reuses its Python project packaging. Existing Entropy neural-network code, experiment reports, and checkpoints remain intact.

Future Proton-I stages may use components from Entropy, but no language model training or language-answering weights are claimed for Gluons 0.1.

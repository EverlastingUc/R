# Proton-I — First Model Scope

**Proposed model ID:** `proton-i-0.1`
**Parent project:** Entropy (this repository's from-scratch neural-network
research project)
**Status:** Scope proposal; training has not started.
**Target user:** A beginner who wants help with HTML and CSS.

## 1. Product goal

Proton-I's first release is deliberately narrow. It should help with beginner
HTML and CSS tasks, explain basic concepts, produce small HTML/CSS examples,
and look up current documentation on the web.

It must not generate JavaScript in this release. When a request requires
JavaScript, a backend, React, or another unsupported technology, it should say
that the request is outside Proton-I 0.1 rather than silently inventing code.

## 2. First-release capabilities

### HTML

- Explain common HTML elements and attributes.
- Produce basic semantic page structures.
- Create simple links, images, lists, tables, and forms.
- Produce small standalone HTML examples.

### CSS

- Explain selectors, specificity basics, colors, typography, spacing, and the
  box model.
- Produce basic Flexbox and Grid layouts.
- Produce simple responsive rules with media queries.
- Pair CSS with an HTML example when that makes the result usable.

### Basic answers with sources

- Search MDN for HTML/CSS documentation at request time.
- Show the matching document title, short summary, and direct source link.
- Prefer an answer grounded in the retrieved result; say when no useful result
  was found.
- Attribute reused MDN documentation to Mozilla Contributors and link to the
  original MDN page.
- Do not add retrieved web pages to training data or model weights.

The initial search adapter may use MDN's currently described
`/api/v1/search` route as a best-effort integration. It is not treated as a
stable official public API: requests need timeouts, response validation,
failure handling, and a fallback link to the MDN search page.

## 3. What the first model actually is

The repository currently contains a from-scratch NumPy feedforward network and
training pipeline. Proton-I 0.1 should reuse that code rather than introducing
a separate PyTorch model or replacing Entropy's existing experiments.

The first trainable component is a **request-intent classifier**:

1. Normalize a user's prompt.
2. Convert it into a fixed bag-of-words feature vector.
3. Run the vector through the existing NumPy feedforward network.
4. Predict a supported intent, such as `css_flexbox` or
   `html_structure`.
5. Route the request to a constrained HTML/CSS example generator or the MDN
   documentation lookup path.
6. Return the explanation, code, and source links as appropriate.

The classifier learns to select a task category. It does **not** learn to write
arbitrary HTML/CSS from language. Initial code examples are project-authored,
reviewed templates and must be labelled as such in the implementation.

## 4. Initial intent labels

- `html_explanation`
- `html_structure`
- `html_links_images`
- `html_forms_lists_tables`
- `css_selectors`
- `css_colors_typography`
- `css_box_model`
- `css_flexbox`
- `css_grid`
- `css_responsive`
- `html_css_code_request`
- `out_of_scope`

The classifier must be able to reject or flag out-of-scope JavaScript requests.
Confidence thresholds and an unknown/ambiguous fallback must be evaluated
rather than guessed.

## 5. Data plan

The recommended first dataset is Option C in
[`docs/dataset_candidates_cp21.md`](../dataset_candidates_cp21.md).

- Write original, project-authored prompt/label pairs.
- Do not scrape websites or copy question banks for classifier training.
- Deduplicate before splitting.
- Use stratified train/validation/test partitions.
- Keep test examples untouched until the final evaluation.
- Record the source, creation method, licensing status, seed, and split counts
  in a dataset manifest.

**Training is blocked until the owner approves Option C.** The repository
instructions require approval before Checkpoint 21 creates or uses a dataset.

## 6. Runtime web lookup is separate from training

The MDN lookup is a retrieval feature, not model training. Its output should
include source attribution and should not be written into the classifier's
training corpus. MDN's licensing notes distinguish documentation from code
samples; see
[MDN attribution and copyright licensing](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license).

## 7. Explicit non-goals for Proton-I 0.1

- No JavaScript generation.
- No React, Vue, Node.js, or backend code.
- No execution of model-generated code.
- No autonomous browser/computer operation.
- No claim that the small classifier is an LLM or a general-purpose chatbot.
- No claim of arbitrary HTML/CSS code generation.
- No PyTorch/TensorFlow replacement for the NumPy network during Checkpoints
  21–25.

## 8. Milestones

1. **CP21 — Dataset engineering:** owner approves Option C; build, validate,
   split, document, and version the original intent dataset.
2. **CP22 — First classifier:** train the intent classifier with the existing
   NumPy network and compare against a majority-class baseline on held-out
   data.
3. **CP25 — Evaluation:** report per-class precision/recall/F1, confusion
   matrix, out-of-scope rejection, and known limitations.
4. **After the classifier:** add the separate MDN runtime lookup and reviewed
   HTML/CSS template responses, with integration tests and source attribution.
5. **Later language-model track:** arbitrary natural-language code generation
   requires the tokenizer/Transformer/language-model checkpoints (31–38) and
   cannot be claimed from the intent classifier.

## 9. Acceptance language

A first Proton-I prototype is successful only if measured tests show that it
routes held-out beginner HTML/CSS prompts better than the documented baseline,
rejects unsupported JavaScript prompts at the evaluated threshold, and returns
only HTML/CSS in its reviewed code templates. MDN retrieval must display source
links and handle unavailable search results without crashing.

The first classifier's accuracy is not a measure of its ability to write
HTML/CSS. Those are separate capabilities and must be evaluated separately.

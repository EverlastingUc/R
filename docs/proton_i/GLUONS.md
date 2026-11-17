# Gluons — Proton-I Stage 1

Gluons is the first, intentionally limited Proton-I assistant. It accepts a
short question, searches the public web, selects a concise relevant search
snippet, and returns one plain-text paragraph with a source URL. It does not
produce Markdown headings, bullet lists, or fenced code blocks.

## Scope

- Basic, short answers grounded in web search results.
- Plain-text output, normally one concise sentence or paragraph.
- A source URL is included when a result is found.
- Website-search failures and empty result sets are handled without crashing.
- JavaScript, TypeScript, Node.js, React, Vue, and Angular requests are outside this first stage.
- No detailed code-generation feature is claimed by Gluons 0.1.

## How it works

Gluons 0.1 is a retrieval-based baseline, not a trained generative language model. It queries DuckDuckGo's HTML search page using Python's standard library, parses the returned result titles, snippets, and destination URLS, then selects a short sentence with keyword overlap against the question. The search page markup is not a stable API, so the adapter uses a timeout, a response-size limit, URL validation, and graceful error handling.

The search adapter is isolated in `src/proton_i/gluons/search.py``. The plain-text answer policy is in `src/proton_i/gluons/assistant.py``. No external Python package is added for this stage. The existing Entropy neural-network modules and their checkpoints remain unchanged; Gluons does not claim that those weights have been trained to answer language questions.

## Run on Windows

From the repository root, install the project in editable mode and run:

```powershell
python -m pip install -e .
python -m proton_i.gluons "What is CSS Flexbox?"
```
You can also pipe a question into the program:

```powershell
echo What is semantic HTML? | python -m proton_i.gluons
```

If a question contains JavaScript or a framework request, Gluons returns a plain-text scope message without searching. If the search provider is unavailable, it reports that clearly instead of inventing an answer.

## Search provider and limitations

Gluons uses DuckDuckGo's public HTML results page as a best-effort search provider, not a stable documented API. The HTML markup may change, and the provider may rate-limit or block automated requests. A search result snippet is not a substitute for reading and verifying the full source page. Gluons shows the source URL so the user can inspect it directly.

This baseline is not a neural language model, does not reason over full web pages, does not execute code, and does not browse autonomously. A later stage may replace the answer-selection component with a trained model while keeping the retrieval boundary and plain-text response contract.

## Tests

Run the targeted Gluons tests:

```powershell
python -m pytest tests/test_gluons.py -q
```

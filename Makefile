PYTHON ?= python

.PHONY: check format-check lint test

check: format-check lint test

format-check:
	$(PYTHON) -m black --check .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

PYTHON ?= python3
JOBS ?= 4

.PHONY: list integrity test setup proofs gpu mirage-check flashmoe-check luce-check sources

list:
	$(PYTHON) replay.py list

integrity:
	$(PYTHON) replay.py integrity

test:
	$(PYTHON) -m unittest discover -s tests -v

setup:
	$(PYTHON) replay.py setup --jobs $(JOBS)

proofs:
	$(PYTHON) replay.py verify --jobs $(JOBS)

gpu:
	$(PYTHON) run_kernels.py --project all --jobs $(JOBS)

mirage-check:
	$(PYTHON) mirage_check.py --jobs $(JOBS)

flashmoe-check:
	$(PYTHON) flashmoe_check.py --jobs $(JOBS)

luce-check:
	$(PYTHON) luce_check.py --jobs $(JOBS)

sources:
	$(PYTHON) fetch_sources.py

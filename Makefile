PYTHON ?= python
.DEFAULT_GOAL := help

.PHONY: help catalog catalog-local experiment check test smoke
help:
	@echo "make catalog                  List experiments and report paths (read-only)"
	@echo "make catalog-local            Also show local output directory presence"
	@echo "make experiment ID=<id>       Show one experiment's configs and artifacts"
	@echo "make check                    Validate navigation; no PyTorch needed"
	@echo "make test                     Run CPU unittest suite"
	@echo "make smoke                    Three small CPU synthetic checks in new outputs"
	@echo "Set PYTHON=/path/to/python to select an existing environment."

catalog:
	$(PYTHON) tools/catalog.py list

catalog-local:
	$(PYTHON) tools/catalog.py list --local

experiment:
	$(PYTHON) tools/catalog.py show "$(ID)" --local

check:
	$(PYTHON) tools/catalog.py check

test:
	CUDA_VISIBLE_DEVICES='' $(PYTHON) -m unittest discover -s tests -v

smoke:
	CUDA_VISIBLE_DEVICES='' $(PYTHON) train.py --variant all --device cpu --updates 3 --scenarios 4 --sampling-steps 4

PY ?= .venv/bin/python
ARCH ?= gfx942
MODEL ?= $(HOME)/llm/qwen3-30b-a3b

.DEFAULT_GOAL := help

.PHONY: help setup test compile explain site-deploy calib bench

help:
	@echo "Targets: setup test compile explain calib site-deploy bench"
	@echo "  make bench [MODEL=/path/to/qwen3-30b-a3b]  Build, check and time kept ETX on MI300X"

bench:
	@python3 -u -m examples.vllm_llm.bench "$(MODEL)"

setup:
	uv venv -q .venv && uv pip install -q --python .venv/bin/python -e ".[dev]"

test:
	$(PY) -m pytest -q

compile:
	$(PY) -m etx compile examples/moe_layer.py --arch $(ARCH) --out build/moe_$(ARCH) --sim

explain:
	$(PY) -m etx explain examples/moe_layer.py --arch $(ARCH) --sim

site-deploy:
	npx wrangler@latest deploy

calib:
	@echo "run on a GPU box: bash bench/calib/run_calib.sh $(ARCH)"

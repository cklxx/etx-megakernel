PY ?= .venv/bin/python
ARCH ?= gfx942
MODEL ?= $(HOME)/llm/qwen3-30b-a3b
BATCH ?= 1
RECIPE ?= kept
ARMS ?= $(HOME)/arms
GATES ?= $(ARMS)/cpu-gates.json

.DEFAULT_GOAL := help

.PHONY: help setup test compile explain site-deploy calib bench preflight

help:
	@echo "Targets: setup test compile explain calib site-deploy bench preflight"
	@echo "  make bench [MODEL=/path/to/qwen3-30b-a3b] [BATCH=1|2|4|8] [RECIPE=kept|plain]  Build, check and time ETX on MI300X"
	@echo "  make preflight [MODEL=/path] [ARMS=/path] [GATES=/path]  Refuse unstaged boxes (read-only)"

preflight:
	@bash tools/vm_preflight.sh "$(MODEL)" "$(ARMS)" --gates "$(GATES)"

bench:
	@python3 -u -m examples.vllm_llm.bench --batch "$(BATCH)" --recipe "$(RECIPE)" "$(MODEL)"

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

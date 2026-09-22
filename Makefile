PY ?= .venv/bin/python
ARCH ?= gfx942

.PHONY: setup test compile explain site-deploy calib

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

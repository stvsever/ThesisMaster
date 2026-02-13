PYTHON ?= python3

.PHONY: qa-unit qa-integration qa-smoke qa-all pipeline-smoke

qa-unit:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -m "unit"

qa-integration:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg $(PYTHON) -m pytest -m "integration and not smoke"

qa-smoke:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PHOENIX_ENABLE_SMOKE=1 MPLBACKEND=Agg $(PYTHON) -m pytest -m "smoke"

qa-all:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg $(PYTHON) -m pytest -m "not smoke"

pipeline-smoke:
	$(PYTHON) Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1 --pattern pseudoprofile_FTC_ID002 --max-profiles 1 --network-boot 3 --network-block-len 10 --network-jobs 1 --run-id smoke_makefile

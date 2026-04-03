.PHONY: population plots finite-sample tests

population:
	python experiments/identification/run_population_only.py
	python experiments/identification/plot_results.py

plots:
	python experiments/identification/plot_results.py

finite-sample:
	python experiments/identification/run_full_simulation.py

nfxp:
	python -u -c "import sys; sys.path.append('experiments'); import identification.run_nfxp_only as r; r.main()"

tests:
	pytest -q tests/test_appendix_metrics.py

docs:
	python -m sphinx -b html docs docs/_build/html

distclean:
	rm -rf dist build *.egg-info

build:
	python -m pip install --upgrade build twine >/dev/null
	python -m build
	twine check dist/*

publish-test:
	@echo "Uploading to TestPyPI (set TWINE_USERNAME=__token__ and TWINE_PASSWORD=***token***)"
	twine upload --repository testpypi dist/*

publish:
	@echo "Uploading to PyPI (set TWINE_USERNAME=__token__ and TWINE_PASSWORD=***token***)"
	twine upload --repository pypi dist/*

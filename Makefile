.PHONY: benchmark smoke test

benchmark:
	mhealth-benchmark

smoke:
	mhealth-benchmark --synthetic --inner-folds 2 --timing-repetitions 1

test:
	python -m pytest -q

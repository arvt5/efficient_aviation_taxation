PYTHON ?= python

.PHONY: reproduce test verify

verify:
	$(PYTHON) -m aviation_taxation.verify_inputs

reproduce: verify
	$(PYTHON) -m aviation_taxation.reproduce
	MPLBACKEND=Agg $(PYTHON) -m aviation_taxation.welfare_plot
	$(PYTHON) -m aviation_taxation.price_figures

test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg $(PYTHON) -m unittest discover -s tests -v

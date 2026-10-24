PYTHON ?= python
.PHONY: reproduce test verify
verify:
	$(PYTHON) -m aviation_taxation.verify_inputs
	MPLBACKEND=Agg $(PYTHON) -m unittest discover -s part_1/tests -v

reproduce: verify
	MPLBACKEND=Agg $(PYTHON) -m part_1.stationary_monopoly.reproduce
	MPLBACKEND=Agg $(PYTHON) -m part_1.time_varying_demand.reproduce
	MPLBACKEND=Agg $(PYTHON) -m part_1.duopoly.reproduce
	$(PYTHON) -m aviation_taxation.reproduce
	$(PYTHON) -m aviation_taxation.eu_policy_brief
	MPLBACKEND=Agg $(PYTHON) -m aviation_taxation.welfare_plot
	$(PYTHON) -m aviation_taxation.price_figures

test:
	MPLBACKEND=Agg $(PYTHON) -m unittest discover -s part_1/tests -v
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg $(PYTHON) -m unittest discover -s part_2/tests -v

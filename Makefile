PYTHON ?= python
.PHONY: reproduce test
reproduce:
	MPLBACKEND=Agg $(PYTHON) -m part_1.stationary_monopoly.reproduce
	MPLBACKEND=Agg $(PYTHON) -m part_1.time_varying_demand.reproduce
	MPLBACKEND=Agg $(PYTHON) -m part_1.duopoly.reproduce
	$(MAKE) -C part_2 reproduce PYTHON="$(PYTHON)"

test:
	MPLBACKEND=Agg $(PYTHON) -m unittest discover -s part_1/tests -v
	$(MAKE) -C part_2 test PYTHON="$(PYTHON)"

PYTHON=python

latex = tex/
img = img/

.PHONY: all run_1 run_2a run_2b tex clean

all: tp1_1.py tp1_2a.py tp1_2b.py

run_1: tp1_1.py
	@$(PYTHON) tp1_1.py > 1.out

run_2a: tp1_2a.py
	@$(PYTHON) tp1_2a.py > 2a.out

run_2b: tp1_2b.py
	@$(PYTHON) tp1_2b.py > 2b.out

clean:
	@rm -f *.out
	@cd $(img) && rm -f *.png
	@cd $(latex) && $(MAKE) --no-print-directory clean

tex:
	@$(MAKE) clean --no-print-directory
	@$(MAKE) run_1 --no-print-directory
	@$(MAKE) run_2a --no-print-directory
	@$(MAKE) run_2b --no-print-directory
	@cd $(latex) && $(MAKE) --no-print-directory

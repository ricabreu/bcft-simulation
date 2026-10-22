# Makefile for the BCFT simulation companion
# Usage: make reproduce | make report | make clean

PY ?= python3

.PHONY: all reproduce report clean help

help:
	@echo "Targets:"
	@echo "  reproduce  - run the pipeline: regenerate figures, dataset, results"
	@echo "  report     - compile report/main.tex to PDF (needs pdflatex)"
	@echo "  clean      - remove LaTeX build artifacts and caches"

all: reproduce report

reproduce:
	cd src && $(PY) run_experiments.py

report:
	cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f report/*.aux report/*.log report/*.out report/*.toc report/_*.png
	rm -rf src/__pycache__

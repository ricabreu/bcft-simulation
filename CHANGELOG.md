# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-07

### Added
- Core library `src/bcft.py`: divergences (Kullback–Leibler, Jensen–Shannon),
  coherence functional, Fisher–Rao distance, geodesic interpolation, and the
  natural-gradient (replicator) coherence flow on the probability simplex.
- Reproducible experiment driver `src/run_experiments.py` (fixed seed) that
  regenerates all figures, the synthetic dataset, and the results summary.
- Three executed experiments: (1) monotonic divergence and core-vs-peripheral
  cost; (2) Fisher–Rao geodesic vs. Euclidean movement; (3) convergent
  natural-gradient coherence flows.
- Synthetic 100-observation brand-field dataset and ideal distribution.
- Technical report `report/main.tex` and compiled `report/main.pdf`.
- Optional real-data extension `src/run_climatebert_evaluation.py` applying BCFT
  divergences to the ClimateBERT `environmental_claims` dataset.
- Packaging: `README.md`, `LICENSE` (MIT), `CITATION.cff`, `.zenodo.json`,
  `requirements.txt`, `requirements-realdata.txt`, `Makefile`, data dictionary,
  and CI reproducibility workflow.

### Notes
- Notation is unified to `o` (observation point), `O` (set of observation
  points), `P_o` (brand state), `P*` (ideal), `C(o) = -D(P_o ‖ P*)`, and
  `F_B = {P_o : o ∈ O}`, locked to the conceptual companion paper.

[1.0.0]: https://github.com/ricabreu/bcft-simulation/releases/tag/v1.0.0

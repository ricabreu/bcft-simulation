# Synthetic Semantic Dynamics Under Brand Coherence Field Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![reproduce](https://github.com/ricabreu/bcft-simulation/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ricabreu/bcft-simulation/actions/workflows/reproduce.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23570568.svg)](https://doi.org/10.5281/zenodo.23570568)

> **Computational companion** to *Brand Coherence Field Theory* (BCFT), the
> conceptual working paper (SSRN). This repository contains the mathematical
> pipeline, **executed** experiments, figures, synthetic dataset, and the
> compiled technical report.

BCFT models brand meaning as a family of probability distributions over a
semantic space and defines **coherence as (negative) divergence from an ideal
brand distribution** on a statistical manifold. This report *executes* the
theory: real divergences, Fisher–Rao geodesics, and natural-gradient coherence
flows on the probability simplex.

## Canonical definition (abridged — see the paper / report for the full statement)

> Brand Coherence Field Theory (BCFT) rests on four primitives. (i) *Semantic
> space*: a brand's possible meanings are represented in a semantic space.
> (ii) *Brand state*: at each observation point `o` the realised meaning of the
> brand is a probability distribution `P_o` over the semantic space.
> (iii) *Ideal distribution*: the brand's intended identity is a reference
> distribution `P*`. (iv) *Coherence*: `C(o) = -D(P_o ‖ P*)`, the negative
> divergence of the observed state from the ideal. The brand is a field
> `F_B = {P_o : o ∈ O}` over the set `O` of observation points, and coherence is
> a graded, relational, measurable property of that field.

## Repository structure

```
arxiv_bcft_simulation/
├── src/
│   ├── bcft.py                        # core library: divergences, Fisher-Rao geometry, NG flow
│   ├── run_experiments.py             # reproduces all figures, dataset, results (fixed seed)
│   └── run_climatebert_evaluation.py  # optional real-data extension (ClimateBERT)
├── figures/                           # generated figures (PNG)
├── data/
│   ├── synthetic_brand_field.csv      # 100-observation synthetic brand field
│   ├── ideal_distribution.json        # P*
│   ├── results_summary.json           # headline numbers (seeded)
│   └── DATA_DICTIONARY.md             # column-level documentation
├── report/
│   ├── main.tex                       # technical report source (LaTeX)
│   └── main.pdf                       # compiled report (Zenodo/GitHub companion PDF)
├── .github/workflows/reproduce.yml    # CI: re-runs the pipeline on every push
├── requirements.txt                   # core dependencies
├── requirements-realdata.txt          # extra deps for the ClimateBERT script
├── Makefile                           # make reproduce | report | clean
├── CITATION.cff                       # how to cite (GitHub "Cite this repository")
├── .zenodo.json                       # Zenodo deposit metadata
├── CHANGELOG.md
├── LICENSE                            # MIT
└── README.md
```

## Reproduce everything

```bash
pip install -r requirements.txt
make reproduce          # or: cd src && python run_experiments.py
```

All randomness is seeded (`SEED = 20260628`); outputs are reproducible.

## Build the report

```bash
make report             # needs a LaTeX distribution (pdflatex)
```

## Experiments

1. **Divergence and incoherence** — coherence declines monotonically with
   distortion; core-dimension distortion is ~35× more costly than peripheral.
2. **Geodesic vs. Euclidean movement** — the Fisher–Rao geodesic length equals
   the Fisher–Rao distance exactly; the Euclidean path is longer and strays from
   it (max deviation 0.120).
3. **Convergent coherence flows** — the natural-gradient flow converges
   geometrically to `P*` from all tested initial states (final KL ≈ 1e-16).

## Optional: real-data extension

`src/run_climatebert_evaluation.py` applies the same BCFT divergences to the
public [`climatebert/environmental_claims`](https://huggingface.co/datasets/climatebert/environmental_claims)
dataset as a real-data illustration (e.g., greenwashing / consistency
diagnostics). It requires extra dependencies and network access:

```bash
pip install -r requirements-realdata.txt
python src/run_climatebert_evaluation.py
```

This script is an illustrative extension and is **not** part of the seeded,
self-contained synthetic pipeline behind the report.

## How to cite

Please cite **both** the software/dataset and the conceptual companion paper.
GitHub will surface a "Cite this repository" button from [`CITATION.cff`](CITATION.cff).

```
Abreu, R. (2026). Synthetic Semantic Dynamics Under Brand Coherence Field
Theory: A Simulation-Based Technical Report (Version 1.0.0) [Software].
Zenodo. https://doi.org/10.5281/zenodo.23570568

Abreu, R. (2026). Brand Coherence Field Theory: An Information-Geometric
Framework for Modelling Brand Meaning. Working paper (SSRN).
```

## Archive and versions

Version 1.0.0 is archived on Zenodo:
[10.5281/zenodo.23570568](https://doi.org/10.5281/zenodo.23570568).

To release a new version:

1. Update the code/report, bump `version` in `CITATION.cff`, and add a
   `CHANGELOG.md` entry.
2. On the Zenodo record, choose **New version**, upload the new snapshot, and
   reserve its DOI.
3. Put the new DOI in the README badge and `CITATION.cff` (`doi:`), then tag
   the release on GitHub.

Once the SSRN paper is live, add its URL/DOI to the report's reference list
(`report/main.tex`) and to the Zenodo record's *Related works*.

## License

MIT — see [LICENSE](LICENSE).

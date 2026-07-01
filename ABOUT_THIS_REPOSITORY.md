# About this repository — `specificity_index` (public)

**Repository:** https://github.com/Baeksw98/specificity_index · **Maintainer:** Asclep Inc., Republic of Korea · **License:** PolyForm Noncommercial 1.0.0 (commercial use by written approval of Asclep Inc.) · **Status:** reference implementation accompanying the methods paper *"Baek's Specificity Index: A Network-Analytic, Game-Theoretic Measure of AI Rater Reasoning Concentration"* (preprint forthcoming).

## What this repository is for

When an automated rater (an LLM judge, or a human) scores an open-ended answer, it also produces *reasoning* — the considerations it cites. This repository provides a small, dependency-light Python library that measures **how concentrated that reasoning is on the content a question is actually about**, versus drifting to off-question content. It turns a rater's cited categories into a weighted network and reads, from the network's structure, a single interpretable number per (question, scoring condition).

It is the shared, citable method behind two studies: a technical paper that defines and proves the measure, and a clinical paper that applies it to AI-rater evaluation in type 2 diabetes care. The library is the artifact others install to compute the measure on their own data.

## What it computes

1. **A co-occurrence network** over content categories: an edge links two categories cited together in the same reasoning trace, weighted by how often that happens. The **tie strength** of an edge is its share of the network's total connective mass; a node's **weighted degree** integrates its connections. (This formalism is inherited and extended from Baek et al., *Sci Rep* 2025;15:5415.)
2. **The Specificity Index `SI`** ∈ [0, 1]: the share of connective mass carried by edges that touch the question's in-scope categories. `SI = 1` is fully on-question; `SI = 0` is fully off-question.
3. **Anchoring Credit `ψ_c` (AC):** a per-category contribution to `SI`, defined as the Shapley value of the edge-induced "anchoring game." It has a closed form and satisfies the exact identity `Σ_c ψ_c = SI`, so the index decomposes transparently into which categories drive it.
4. **Inference:** a patient/cluster-clustered nonparametric bootstrap for 95% confidence intervals on `SI` and on the between-condition difference, with Benjamini–Hochberg correction across questions.

The measure is **descriptive**; the library makes no causal claim.

## How to use it

```bash
pip install -e .            # numpy + pandas only; Python >= 3.10
```
```python
import pandas as pd, specificity_index as si
df = pd.read_csv("claim_assignments.csv")   # patient_id, question, protocol, trace_id, claim_code
cell = df[(df["question"] == "Q3.1") & (df["protocol"] == "GR")]
si.specificity_index(cell, "Q3.1")          # the index for one cell
si.anchoring_credit(cell, "Q3.1")           # {category: psi_c}, sums to SI
si.specificity_table(df)                    # per-question SI by condition + difference
si.specificity_bootstrap(df, n_boot=2000)   # the same with 95% CIs and BH-adjusted p
```
The in-scope category map (`specificity_index.scope.IN_SCOPE`) is domain-specific and **must be validated by a domain expert** before the index is trusted on a new task.

## Repository layout
- `specificity_index/` — the library (`network.py`, `index.py`, `bootstrap.py`, `scope.py`).
- `tests/` — unit tests (run from inside this directory: `python -m pytest tests/`).
- `README.md` — quickstart; this file — the fuller explanation.
- `LICENSE` / `LICENSE.md` — PolyForm Noncommercial 1.0.0 (verbatim) + the Asclep Inc. commercial notice.
- `examples/` — a worked example reproducing the headline numbers from the accompanying paper.

## How to cite
Until the preprint is posted, cite this repository and the method anchor: Baek S, Kim K, Park SY, et al. *Sci Rep*. 2025;15(1):5415. doi:10.1038/s41598-025-89340-2. The defining paper will supersede this notice on release.

## Relationship to the other study
The clinical companion study (a separate, private reproduction repository) uses this library as a secondary, structure-aware measure alongside its primary rate-based metrics, and cites the methods paper. Together the two form a small connected reference network rather than standalone artifacts.

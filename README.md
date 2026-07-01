# Baek's Specificity Index

| Field | Value |
|---|---|
| Repository | https://github.com/Baeksw98/specificity_index |
| Maintainer | Asclep Inc., Republic of Korea |
| Status | Public reference implementation for the methods paper in preparation |
| License | PolyForm Noncommercial License 1.0.0; commercial use requires written approval from Asclep Inc. |

Baek's Specificity Index is a network-analytic, game-theoretic measure of how concentrated a rater's reasoning is on the in-scope content of an evaluation question. Given the clinical or semantic categories cited by an AI rater, LLM judge, or human rater, the index quantifies whether the reasoning stays anchored to the question's intended scope or drifts into off-question content.

The package is intentionally lightweight: Python >= 3.10 with `numpy` and `pandas` only.

---

## What This Repository Is For

This repository provides the reusable method artifact behind the companion methods paper:

> *Baek's Specificity Index: A Network-Analytic, Game-Theoretic Measure of AI Rater Reasoning Concentration* (preprint forthcoming).

It is designed to be used by researchers who have a long-format table of rater reasoning traces and category citations. The clinical Type 2 diabetes qualitative assessment study uses this package as a secondary, structure-aware measure and cites the methods paper for the formal definition.

This repository does **not** contain proprietary patient records, raw AI-rater traces, OpenAI Batch API outputs, or the private qualitative reproduction package. It contains the metric implementation, tests, a demo, and small result tables for the full40 clinical application.

---

## Method Summary

For one `(question q, protocol p)` cell, the cited content categories form a weighted undirected co-occurrence network:

- **Nodes:** content categories, e.g. `C1` to `C28`.
- **Edges:** two categories are linked when they are cited together in the same reasoning trace.
- **Edge weight:** `w(a,b)`, the number of traces citing both categories.
- **Tie strength:** `ts(a,b) = w(a,b) / sum(w)`, so all edge tie strengths sum to 1.
- **Question scope:** `S(q)`, the domain-validated set of categories in scope for question `q`.

The primary scalar is:

```text
SI(q,p) = sum ts(a,b) over edges where a in S(q) or b in S(q)
```

Interpretation:

- `SI = 1`: all connective reasoning mass touches question-relevant content.
- `SI = 0`: no connective reasoning mass touches question-relevant content.
- The measure is descriptive. The package does not make causal claims.

The package also computes **Anchoring Credit** (`AC`, symbol `psi_c`), a closed-form Shapley-value decomposition of `SI` by category:

```text
sum_c psi_c = SI
```

This gives per-category attribution without exponential Shapley enumeration.

---

## Public API

| Function | Purpose |
|---|---|
| `cooccurrence(cell_df)` | Build category co-occurrence counts for one cell. |
| `tie_strength(cooc)` | Normalize co-occurrence counts into edge tie strengths. |
| `weighted_degree(ts)` | Compute node weighted degree from tie strengths. |
| `specificity_index(cell_df, question)` | Compute `SI` for one `(question, protocol)` cell. |
| `anchoring_credit(cell_df, question)` | Compute per-category `psi_c`; values sum to `SI`. |
| `specificity_table(assignments_df)` | Compute GR, Non-GR, and delta SI point estimates by question. |
| `specificity_bootstrap(assignments_df, n_boot=2000)` | Compute patient-clustered bootstrap CIs and BH-adjusted p-values. |

---

## Input Data Contract

Public functions expect a long-format `pandas.DataFrame` with one row per `(trace, distinct cited category)`.

Required columns:

| Column | Meaning |
|---|---|
| `patient_id` | Cluster/resampling unit for inference. |
| `question` | Evaluation question key, e.g. `Q3.1`. |
| `protocol` | Scoring/evaluation condition, e.g. `GR` or `Non-GR`. |
| `trace_id` | Unique reasoning-trace identifier. |
| `claim_code` | Cited content category, e.g. `C1`. |

Example:

```csv
patient_id,question,protocol,trace_id,claim_code
P001,Q3.1,GR,T0001,C1
P001,Q3.1,GR,T0001,C2
P001,Q3.1,GR,T0002,C1
```

The in-scope map (`specificity_index.scope.IN_SCOPE`) is domain-specific and must be reviewed by a qualified domain expert before results are interpreted in a new domain.

---

## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

---

## Basic Usage

```python
import pandas as pd
import specificity_index as si

df = pd.read_csv("claim_assignments.csv")

cell = df[(df["question"] == "Q3.1") & (df["protocol"] == "GR")]
si_value = si.specificity_index(cell, "Q3.1")
credit = si.anchoring_credit(cell, "Q3.1")
table = si.specificity_table(df)
ci_table = si.specificity_bootstrap(df, n_boot=2000)

print(si_value)
print(sum(credit.values()))  # equals si_value, up to floating-point tolerance
```

Worked example:

```bash
python examples/demo.py
```

---

## Included Repository Files

```text
specificity_index/
├── README.md
├── ABOUT_THIS_REPOSITORY.md
├── LICENSE
├── LICENSE.md
├── pyproject.toml
├── specificity_index/
│   ├── __init__.py
│   ├── network.py
│   ├── index.py
│   ├── bootstrap.py
│   └── scope.py
├── tests/
│   ├── test_network.py
│   ├── test_index.py
│   └── test_bootstrap.py
├── examples/
│   └── demo.py
├── scripts/
│   ├── validate_si_full40.py
│   └── run_bootstrap_full40.py
└── results/
    ├── specificity_index_full40.csv
    ├── specificity_index_full40_ci.csv
    ├── anchoring_credit_full40.csv
    └── validation_summary.json
```

The `results/` directory contains small reproducibility outputs for the full40 clinical application. It does not contain raw patient data or raw rater outputs.

---

## Reproducing Tests

```bash
python -m pytest tests/ -q
```

Current validation result from the publication check:

```text
61 passed
```

If your local pytest version warns about an unknown `asyncio_mode` option, the warning is harmless for this package; the tests are synchronous.

---

## Full40 Clinical Result Summary

The included `results/specificity_index_full40_ci.csv` table summarizes the clinical companion application:

- Mean `SI_GR`: `0.9198`.
- Mean `SI_Non-GR`: `0.7158`.
- Mean `delta_SI`: `+0.2041`.
- Point-estimate direction: `SI_GR > SI_Non-GR` in `14/15` questions.
- Inferential summary:
  - `13/15` questions had significant positive GR delta.
  - `1/15` question had a significant reversal (`Q3.1`).
  - `1/15` question was null (`Q3.8`).

These are descriptive comparisons of reasoning concentration, not causal estimates.

---

## Relationship to the Qualitative Assessment Repository

The companion clinical paper has a separate private reproduction repository for editorial review. That private repository contains de-identified aggregate CSVs and a deterministic script that regenerates Paper 2 tables and figures. It uses this public `specificity_index` package for the network-analytic SI table and cites the methods paper for the formal definition.

In short:

- This public repository defines and implements the metric.
- The private qualitative repository reproduces the clinical paper's figure/table package.

---

## Citation

Until the Specificity Index preprint is posted, cite this repository and the method anchor:

Baek S, Kim K, Park SY, et al. Application of network analysis and association rule mining for visualizing the lymph node metastasis patterns in esophageal squamous cell carcinoma. *Scientific Reports*. 2025;15(1):5415. doi:10.1038/s41598-025-89340-2.

The defining Specificity Index paper will supersede this temporary citation notice after release.

---

## License

Released for noncommercial use under the **PolyForm Noncommercial License 1.0.0**. See `LICENSE` and `LICENSE.md`.

Academic and noncommercial research use is permitted. Commercial use, redistribution for commercial purposes, or integration into commercial products requires separate written approval from Asclep Inc., Republic of Korea.

© 2026 Asclep Inc., Republic of Korea.

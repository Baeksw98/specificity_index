# Baek's Specificity Index

A network-analytic, game-theoretic measure of how concentrated a rater's reasoning is on the *in-scope* content of an evaluation question. Given the categories an AI rater (or human rater) cites when judging answers, the Specificity Index quantifies whether that reasoning anchors on the question's intended content or drifts to off-question content.

> **Citation / preprint forthcoming.** The defining paper — *Baek's Specificity Index: A Network-Analytic, Game-Theoretic Measure of AI Rater Reasoning Concentration* — is in preparation (arXiv, stat.ME; cross-list cs.SI, cs.GT). Until it is posted, cite this repository and the method anchor below.

## Method

For one (question `q`, protocol `p`) cell, the cited clinical-content categories form a weighted **co-occurrence network**: an edge links two categories cited together, weighted by co-occurrence count. Following the method anchor, the **tie strength** of an edge is its share of total connective mass, `ts(a,b) = w(a,b) / Σ w`, and a node's **weighted degree** integrates degree with tie strength.

- **Specificity Index** `SI(q,p) = Σ ts(a,b)` over edges with at least one endpoint in the question's in-scope set `S(q)`. `SI ∈ [0,1]`; `SI = 1` is fully anchored, `SI = 0` is fully off-question.
- **Anchoring Credit** `AC`, symbol `ψ_c` — the per-category contribution to `SI`, the Shapley value of the edge-induced anchoring game: `ψ_c = ½ Σ_{b: in-scope-incident} ts(c,b)`. It satisfies the exact decomposition `Σ_c ψ_c = SI` (efficiency), in closed form (no exponential enumeration).

Attribution is **descriptive**; the package makes no causal claim.

## Method anchor (inherited)

Baek S, Kim K, Park SY, et al. Application of network analysis and association rule mining for visualizing the lymph node metastasis patterns in esophageal squamous cell carcinoma. *Sci Rep*. 2025;15(1):5415. doi:[10.1038/s41598-025-89340-2](https://doi.org/10.1038/s41598-025-89340-2). The tie-strength / weighted-degree / network-heatmap formalism is inherited and extended from this work.

## Install

```bash
pip install -e .        # numpy + pandas only; Python >= 3.10
```

## Usage

```python
import pandas as pd
import specificity_index as si

# one row per (trace, distinct cited category)
df = pd.read_csv("claim_assignments.csv")   # patient_id, question, protocol, trace_id, claim_code

cell = df[(df.question == "Q3.1") & (df.protocol == "GR")]
si.specificity_index(cell, "Q3.1")          # -> SI in [0, 1]
si.anchoring_credit(cell, "Q3.1")           # -> {category: psi_c}, sums to SI
si.specificity_table(df)                    # per-question SI_GR, SI_Non-GR, delta_SI (+ SI_wd secondary)
```

The in-scope map (`specificity_index.scope.IN_SCOPE`) **must be validated by a domain expert** before the index is trusted for a new domain.

## Reproducing the tests

```bash
python -m pytest tests/ -q
```

## License

Released for **noncommercial use** under the **PolyForm Noncommercial License 1.0.0** (see `LICENSE` and `LICENSE.md`, placed verbatim from the official PolyForm source). Academic and non-commercial research use is permitted. **Commercial use or redistribution for commercial purposes requires separate written approval from Asclep Inc.** (Republic of Korea), a research corporation — not an academic institution. For commercial licensing, contact Asclep Inc.

© Asclep Inc., Republic of Korea.

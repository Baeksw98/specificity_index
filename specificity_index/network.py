"""Network-analytic building blocks for the Specificity Index.

Overview
--------
A rater's reasoning for a given (question, protocol) cell is modeled as a
**weighted undirected network** whose nodes are clinical-content categories
(C1–C28) and whose edges encode how often pairs of categories co-occur within
the same reasoning trace.  Three building blocks are implemented here:

``cooccurrence``
    Counts how many reasoning traces in a cell mention each *pair* of
    categories together.  Each non-empty trace contributes one count to every
    pair of distinct categories it cites.

``tie_strength``
    Converts raw co-occurrence counts to a normalized weight for each edge.
    Formally, for category pair (a, b):

        ts(a, b) = cooc(a, b) / sum_{i < j} cooc(i, j)

    This is the Sci-Rep-2025 form (Baek S, Kim K, et al. Sci Rep. 2025;15:5415,
    doi:10.1038/s41598-025-89340-2).  Because the denominator is the sum of all
    pair counts, the tie-strength values sum to exactly 1 over all edges — they
    form a probability distribution over edges, interpretable as the *share of
    the network's total connective mass* carried by each edge.

``weighted_degree``
    Measures the overall connectivity of each node by summing the tie strengths
    of all edges incident to it:

        wdeg(c) = sum_{b != c} ts(c, b)

    A node with high weighted degree is a **hub**: it co-occurs frequently with
    many other categories, contributing a large share of the network's total
    connective mass.  In the context of the Specificity Index, an in-scope
    category that is a hub indicates that the rater's reasoning is anchored on
    the question's target content.

Terminology
-----------
- **Edge**: a link between two category nodes that co-occur in at least one
  reasoning trace within the cell.
- **Tie strength**: the normalized weight of an edge, expressing what fraction
  of all pairwise co-occurrences it represents.  Inherits the Sci-Rep-2025
  definition by methodological lineage.
- **Weighted degree**: the sum of tie strengths of all edges touching a node;
  integrates a node's connection count with the intensity of those connections.
- **Hub**: a node with disproportionately high weighted degree; in clinical
  terms, a category that frequently co-occurs with many other cited categories.

Open decisions (gate-A)
-----------------------
The edge definition (G1) and the "main judgement" filter (G2) are undecided
at the time of this skeleton release.  Provisional definitions are used:

- G1 (edge): two categories co-occur if they appear in the same trace.
- G2 (main judgement): any trace that cites at least one category.

These are noted with [OPEN] in the docstrings below and MUST NOT be treated
as locked specifications.  See docs/Baek_Specificity_Index_Design_Consideration.md
Section 10 for the full list of gate-A decisions.

Input contract
--------------
All public functions accept a long-format ``pandas.DataFrame`` with one row
per (trace, distinct cited category) and the following required columns:

    patient_id  : str  — patient identifier
    question    : str  — evaluation question key (e.g., "Q3.1")
    protocol    : str  — protocol label (e.g., "GR" or "Non-GR")
    trace_id    : str  — unique reasoning trace identifier
    claim_code  : str  — category code (e.g., "C1")

The ``cell`` parameter throughout refers to a subset of this table already
filtered to a single (question, protocol) cell by the caller.

Method anchor
-------------
Baek S, Kim K, et al. Sci Rep. 2025;15:5415 (doi:10.1038/s41598-025-89340-2).
The tie-strength and weighted-degree formulas used here are inherited from
that work; this package extends them to the reasoning-specificity domain.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Mapping

import pandas as pd


def _trace_category_sets(cell: pd.DataFrame) -> list[frozenset[str]]:
    """Return the distinct cited category set for each trace in *cell*.

    Traces that cite no categories are omitted (they carry no edge information).
    This implements the provisional G2 "main judgement" definition: any trace
    with at least one cited category.  [OPEN G2]
    """
    out: list[frozenset[str]] = []
    for _trace_id, sub in cell.groupby("trace_id"):
        codes = frozenset(str(c) for c in sub["claim_code"].unique())
        if codes:
            out.append(codes)
    return out


def cooccurrence(cell_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Undirected category co-occurrence counts within one (question, protocol) cell.

    An **edge** between two categories exists when both are cited within the
    same reasoning trace.  For each trace the function increments the count of
    every ordered pair (a, b) with a < b (lexicographic) by one.  The result
    is a dictionary mapping each such pair to its total count across all traces
    in the cell.

    Parameters
    ----------
    cell_df:
        Long-format DataFrame for a single (question, protocol) cell.
        Required columns: ``trace_id``, ``claim_code``.

    Returns
    -------
    dict mapping (category_a, category_b) -> count, with category_a < category_b
    (lexicographic).  An empty dict is returned when no cell contains two or
    more distinct categories.

    Notes
    -----
    [OPEN G1] The edge definition (co-occurrence within a trace) is provisional.
    [OPEN G2] "Main judgement" is provisionally any trace with >= 1 category.
    Both are gate-A decisions and must not be treated as finalized.
    """
    cooc: Counter[tuple[str, str]] = Counter()
    for codes in _trace_category_sets(cell_df):
        for a, b in itertools.combinations(sorted(codes), 2):
            cooc[(a, b)] += 1
    return dict(cooc)


def tie_strength(
    cooc: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], float]:
    """Sci-Rep-2025 tie strength: edge co-occurrence count divided by total.

    Formally, for each category pair (a, b):

        ts(a, b) = cooc(a, b) / sum_{i < j} cooc(i, j)

    The denominator is the sum of co-occurrence counts over **all** pairs in
    the network, so the tie-strength values sum to exactly 1.  They form a
    probability distribution over edges and express each edge's share of the
    network's total connective mass.

    Parameters
    ----------
    cooc:
        Mapping from category pair (a, b) to co-occurrence count, as returned
        by :func:`cooccurrence`.

    Returns
    -------
    dict mapping (category_a, category_b) -> tie_strength in [0, 1].
    Returns an empty dict when *cooc* is empty or all counts are zero.

    References
    ----------
    Baek S, Kim K, et al. Sci Rep. 2025;15:5415.
    doi: 10.1038/s41598-025-89340-2
    """
    total = sum(cooc.values())
    if total == 0:
        return {}
    return {pair: count / total for pair, count in cooc.items()}


def weighted_degree(
    ts: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    """Node weighted degree: sum of tie strengths of all incident edges.

    For each category node *c*:

        wdeg(c) = sum_{b != c} ts(c, b)

    where the sum runs over all edges in the network that touch *c*.  This
    integrates a node's raw connection count with the intensity (tie strength)
    of those connections.

    A category with high weighted degree is a **hub**: it co-occurs frequently
    with many other cited categories and contributes a large share of the
    network's connective mass.  In the context of the Specificity Index, an
    in-scope hub indicates reasoning that is anchored on the question's target
    content.

    Parameters
    ----------
    ts:
        Mapping from category pair to tie strength, as returned by
        :func:`tie_strength`.

    Returns
    -------
    dict mapping category_code -> weighted_degree >= 0.
    Returns an empty dict when *ts* is empty.

    References
    ----------
    Baek S, Kim K, et al. Sci Rep. 2025;15:5415.
    doi: 10.1038/s41598-025-89340-2
    """
    wdeg: dict[str, float] = defaultdict(float)
    for (a, b), strength in ts.items():
        wdeg[a] += strength
        wdeg[b] += strength
    return dict(wdeg)

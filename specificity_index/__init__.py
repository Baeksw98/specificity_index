"""Specificity Index — public Python package.

This package implements Baek's Specificity Index, a network-analytic measure
of how concentrated an AI rater's clinical reasoning is on the in-scope
content of an evaluation question.

The mathematical specification is logic-locked (gate B).  See
docs/program_planning/specificity_index_FORMAL_SPEC.md for the authoritative
definition.

Method anchor
-------------
Baek S, Kim K, et al. Sci Rep. 2025;15:5415.
doi: 10.1038/s41598-025-89340-2

Exported names
--------------
Network building blocks (decided, stable)::

    cooccurrence(cell_df)        -> dict[(str, str), int]
    tie_strength(cooc)           -> dict[(str, str), float]
    weighted_degree(ts)          -> dict[str, float]

In-scope category map (clinician validation required)::

    DEFAULT_MAIN          — primary in-scope categories per question
    DEFAULT_CROSS_CUTTING — cross-cutting additions per question
    IN_SCOPE              — merged map used in all computations

Locked index functions (spec §3–§4, gate B)::

    specificity_index(cell_df, question, in_scope=IN_SCOPE) -> float
    anchoring_credit(cell_df, question, in_scope=IN_SCOPE)  -> dict[str, float]
    specificity_table(assignments_df, gr_label, non_gr_label) -> pd.DataFrame

Backward-compatibility CANDIDATE scalars (gate-A, not the locked formula)::

    candidate_indices(cell_df, question)             -> dict
    candidate_index_tie_strength_concentration(...)  -> float  [CANDIDATE]
    candidate_index_weighted_degree_centrality(...)  -> float  [CANDIDATE]
    network_index_table(assignments)                 -> pd.DataFrame
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Asclep Inc."

from .scope import DEFAULT_CROSS_CUTTING, DEFAULT_MAIN, IN_SCOPE
from .network import cooccurrence, tie_strength, weighted_degree
from .index import (
    # Locked (gate B)
    specificity_index,
    anchoring_credit,
    specificity_table,
    # Backward-compatibility CANDIDATE shims
    candidate_indices,
    candidate_index_tie_strength_concentration,
    candidate_index_weighted_degree_centrality,
    network_index_table,
)
from .bootstrap import specificity_bootstrap

__all__ = [
    "__version__",
    # scope
    "DEFAULT_MAIN",
    "DEFAULT_CROSS_CUTTING",
    "IN_SCOPE",
    # network building blocks
    "cooccurrence",
    "tie_strength",
    "weighted_degree",
    # locked index functions (gate B)
    "specificity_index",
    "anchoring_credit",
    "specificity_table",
    # bootstrap inference (spec §6)
    "specificity_bootstrap",
    # backward-compatibility CANDIDATE scalars
    "candidate_indices",
    "candidate_index_tie_strength_concentration",
    "candidate_index_weighted_degree_centrality",
    "network_index_table",
]

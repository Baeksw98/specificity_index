"""Specificity Index — locked mathematical implementation (gate B).

This module implements Baek's Specificity Index (SI), the per-category
Anchoring Credit (AC, symbol ψ_c), and the cross-protocol comparison table,
following the **logic-locked** formal specification:
docs/program_planning/specificity_index_FORMAL_SPEC.md.

Mathematical reference
-----------------------
Baek S, Kim K, et al. Sci Rep. 2025;15:5415.
doi: 10.1038/s41598-025-89340-2

Core objects (§3–§4 of the formal spec)
-----------------------------------------
``specificity_index``
    PRIMARY scalar.  SI(q,p) = Σ_{a∈S(q) ∨ b∈S(q)} ts(a,b).
    The share of the network's tie-strength mass carried by edges that are
    *in-scope-incident* (at least one endpoint lies in S(q)).  Range [0,1].
    Returns float('nan') when the cell has no co-occurrence edges (spec §3,
    Empty-network property).

``anchoring_credit``
    Per-category Anchoring Credit (AC), symbol ψ_c.  For each category c:

        ψ_c(q,p) = ½ · Σ_{b≠c : c∈S(q) ∨ b∈S(q)} ts(c,b)

    This is the Shapley value of the edge-induced anchoring game (spec §4).
    Each in-scope-incident edge's tie-strength is split equally between its
    two endpoints.

    **Key identity (efficiency / index decomposition):**

        Σ_c ψ_c(q,p) = SI(q,p)

    Proof: Σ_c ψ_c = ½ Σ_c Σ_{b≠c, in-scope-incident} ts(c,b)
           = ½ · 2 · Σ_{in-scope-incident {a,b}} ts(a,b) = SI.  ∎

    (Each in-scope-incident edge {a,b} contributes ts(a,b) to both c=a and
    c=b; the factor ½ cancels the double-count.)  Reference: spec §4.

    Causal interpretation is **out of scope**.  ψ_c is a *descriptive*
    attribution of observed connective mass.  No causal claims are made or
    implied (spec §5).

``specificity_table``
    Per-question point estimates: SI for GR and Non-GR and
    ΔSI = SI_GR − SI_NonGR.  SI_wd (the weighted-degree centrality variant,
    spec §3 secondary scalar) is included as clearly-labelled secondary
    columns for sensitivity analysis.

Backward-compatibility shims
-----------------------------
The gate-A CANDIDATE functions (``candidate_index_tie_strength_concentration``,
``candidate_index_weighted_degree_centrality``, ``candidate_indices``,
``network_index_table``) are retained with unchanged signatures and are
marked CANDIDATE in their docstrings.  They should not be cited as the
locked formula.

Input contract
--------------
All public functions accept a long-format ``pandas.DataFrame`` with one row
per (trace, distinct cited category) and the following required columns:

    patient_id  : str  — patient identifier
    question    : str  — evaluation question key (e.g., "Q3.1")
    protocol    : str  — protocol label (e.g., "GR" or "Non-GR")
    trace_id    : str  — unique reasoning trace identifier
    claim_code  : str  — category code (e.g., "C1")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import pandas as pd

from .network import cooccurrence, tie_strength, weighted_degree
from .scope import IN_SCOPE


# ---------------------------------------------------------------------------
# Primary locked scalar — §3
# ---------------------------------------------------------------------------

def specificity_index(
    cell_df: pd.DataFrame,
    question: str,
    in_scope: Mapping[str, frozenset[str]] = IN_SCOPE,
) -> float:
    """Baek's Specificity Index: in-scope connective concentration (spec §3).

    Computes the share of the network's tie-strength mass that is carried by
    *in-scope-incident* edges — edges where at least one endpoint belongs to
    S(q), the question's in-scope category set:

        SI(q,p) = Σ_{ {a,b} : a∈S(q) ∨ b∈S(q) } ts(a,b)

    Because tie strengths sum to 1 (they form a probability distribution over
    edges), SI lies in [0, 1].  SI = 1 means every edge is in-scope-incident
    (fully anchored reasoning); SI = 0 means no edge touches S(q).

    Parameters
    ----------
    cell_df:
        Long-format DataFrame for a single (question, protocol) cell.
        Required columns: ``trace_id``, ``claim_code``.
    question:
        Question key used to look up S(q) from *in_scope*.
    in_scope:
        Mapping from question key to frozen set of in-scope category codes.
        Defaults to :data:`~specificity_index.scope.IN_SCOPE`.

    Returns
    -------
    float in [0, 1], or ``float("nan")`` when the cell has no co-occurrence
    edges (undefined per spec §3 Empty-network property; not imputed).

    References
    ----------
    Formal spec §3; Baek S, et al. Sci Rep. 2025;15:5415.
    doi: 10.1038/s41598-025-89340-2
    """
    scope = in_scope.get(question, frozenset())
    cooc = cooccurrence(cell_df)
    ts = tie_strength(cooc)
    if not ts:
        return float("nan")
    return float(sum(s for (a, b), s in ts.items() if a in scope or b in scope))


# ---------------------------------------------------------------------------
# Per-category attribution — §4
# ---------------------------------------------------------------------------

def anchoring_credit(
    cell_df: pd.DataFrame,
    question: str,
    in_scope: Mapping[str, frozenset[str]] = IN_SCOPE,
) -> dict[str, float]:
    """Per-category Anchoring Credit (AC), symbol ψ_c (spec §4).

    The Shapley value of the edge-induced anchoring cooperative game with
    player set C and value function

        v(T) = Σ_{ {a,b}⊆T : a∈S(q) ∨ b∈S(q) } ts(a,b)

    Because the value function is an edge-induced sum, the Shapley value
    admits the closed form (spec §4):

        ψ_c(q,p) = ½ · Σ_{ b≠c : c∈S(q) ∨ b∈S(q) } ts(c,b)

    That is, each in-scope-incident edge's tie-strength is attributed equally
    to its two endpoints.

    **Efficiency / index decomposition identity** (key property, spec §4):

        Σ_c ψ_c(q,p) = SI(q,p)

    The Anchoring Credit (AC) decomposes the Specificity Index exactly into
    per-category contributions with a closed form (no 2^28 enumeration).
    This is the principled, axiomatically-justified (efficiency, symmetry,
    null-player, additivity) attribution of the index (spec §4).

    **Causal interpretation is out of scope.**  ψ_c is a *descriptive*
    measure of observed connective mass; no causal claims are made or implied
    (spec §5).

    Parameters
    ----------
    cell_df:
        Long-format DataFrame for a single (question, protocol) cell.
        Required columns: ``trace_id``, ``claim_code``.
    question:
        Question key used to look up S(q) from *in_scope*.
    in_scope:
        Mapping from question key to frozen set of in-scope category codes.
        Defaults to :data:`~specificity_index.scope.IN_SCOPE`.

    Returns
    -------
    dict mapping category code -> ψ_c (float >= 0), containing only
    categories with nonzero Anchoring Credit.  Returns an empty dict when
    the cell has no co-occurrence edges.

    References
    ----------
    Formal spec §4; Baek S, et al. Sci Rep. 2025;15:5415.
    doi: 10.1038/s41598-025-89340-2
    """
    scope = in_scope.get(question, frozenset())
    cooc = cooccurrence(cell_df)
    ts_map = tie_strength(cooc)
    if not ts_map:
        return {}

    psi: dict[str, float] = defaultdict(float)
    for (a, b), s in ts_map.items():
        # Edge {a,b} is in-scope-incident if at least one endpoint is in S(q).
        if a in scope or b in scope:
            psi[a] += 0.5 * s
            psi[b] += 0.5 * s

    return {c: v for c, v in psi.items() if v != 0.0}


# ---------------------------------------------------------------------------
# Cross-protocol comparison table — §6
# ---------------------------------------------------------------------------

def specificity_table(
    assignments_df: pd.DataFrame,
    gr_label: str = "GR",
    non_gr_label: str = "Non-GR",
) -> pd.DataFrame:
    """Per-question Specificity Index point estimates for GR vs Non-GR (spec §6).

    For each evaluation question present in *assignments_df*, computes:

    - ``SI_{gr_label}``:     SI for the GR protocol (primary scalar, spec §3).
    - ``SI_{non_gr_label}``: SI for the Non-GR protocol.
    - ``delta_SI``:          ΔSI = SI_GR − SI_NonGR (descriptive contrast).
    - ``SI_wd_{gr_label}``:  SI_wd (secondary scalar) for GR — see below.
    - ``SI_wd_{non_gr_label}``: SI_wd for Non-GR.

    ``SI_wd`` (secondary / sensitivity scalar, spec §3):
        SI_wd(q,p) = ( Σ_{c∈S(q)} wd(c) ) / ( Σ_c wd(c) )
                   = ½ · Σ_{c∈S(q)} wd(c)
    Included as a clearly-labelled secondary column for robustness analysis;
    the primary headline scalar is SI (tie-strength incidence).

    Point estimates only.  Bootstrap confidence intervals and BH-FDR
    correction across questions (spec §6) are not implemented here.

    Parameters
    ----------
    assignments_df:
        Long-format DataFrame with columns:
        patient_id, question, protocol, trace_id, claim_code.
    gr_label:
        Protocol label for the GR arm (default ``"GR"``).
    non_gr_label:
        Protocol label for the Non-GR arm (default ``"Non-GR"``).

    Returns
    -------
    pandas.DataFrame with one row per question and columns:
        question, SI_{gr_label}, SI_{non_gr_label}, delta_SI,
        SI_wd_{gr_label} [secondary], SI_wd_{non_gr_label} [secondary].
    NaN appears wherever a cell has no co-occurrence edges.

    References
    ----------
    Formal spec §3, §6; Baek S, et al. Sci Rep. 2025;15:5415.
    doi: 10.1038/s41598-025-89340-2
    """
    rows: list[dict] = []
    for q in sorted(assignments_df["question"].unique()):
        q_df = assignments_df[assignments_df["question"] == q]
        gr_df = q_df[q_df["protocol"] == gr_label]
        non_gr_df = q_df[q_df["protocol"] == non_gr_label]

        si_gr = specificity_index(gr_df, q)
        si_non_gr = specificity_index(non_gr_df, q)
        delta = (
            float("nan")
            if (si_gr != si_gr or si_non_gr != si_non_gr)  # NaN check
            else si_gr - si_non_gr
        )

        # Secondary scalar: weighted-degree centrality of in-scope hubs.
        si_wd_gr = _si_wd(gr_df, q)
        si_wd_non_gr = _si_wd(non_gr_df, q)

        rows.append({
            "question": q,
            f"SI_{gr_label}": si_gr,
            f"SI_{non_gr_label}": si_non_gr,
            "delta_SI": delta,
            f"SI_wd_{gr_label} [secondary]": si_wd_gr,
            f"SI_wd_{non_gr_label} [secondary]": si_wd_non_gr,
        })

    return pd.DataFrame(rows)


def _si_wd(cell_df: pd.DataFrame, question: str) -> float:
    """Internal helper: SI_wd secondary scalar (spec §3)."""
    scope = IN_SCOPE.get(question, frozenset())
    cooc = cooccurrence(cell_df)
    ts = tie_strength(cooc)
    if not ts:
        return float("nan")
    wdeg = weighted_degree(ts)
    total = sum(wdeg.values())
    if total == 0.0:
        return float("nan")
    in_scope_wdeg = sum(v for c, v in wdeg.items() if c in scope)
    return float(in_scope_wdeg / total)


# ---------------------------------------------------------------------------
# Backward-compatibility CANDIDATE shims (gate-A, not the locked formula)
# ---------------------------------------------------------------------------

def candidate_index_tie_strength_concentration(
    cell_df: pd.DataFrame,
    question: str,
) -> float:
    """CANDIDATE scalar A: share of tie-strength mass incident to in-scope nodes.

    .. deprecated::
        This is the gate-A CANDIDATE version.  The locked formula is
        :func:`specificity_index`, which implements the same computation
        but with the correct signature (includes ``in_scope`` parameter).
        Retained for backward compatibility only.

    Computes:

        SI_A(q, p) = Σ_{ {a,b} : a∈S(q) ∨ b∈S(q) } ts(a,b)

    CANDIDATE — not independently authoritative; use :func:`specificity_index`.

    Parameters
    ----------
    cell_df:
        Long-format DataFrame for a single (question, protocol) cell.
    question:
        Question key used to look up S(q) from :data:`~.scope.IN_SCOPE`.

    Returns
    -------
    float in [0, 1], or ``float("nan")`` when the cell has no edges.
    """
    return specificity_index(cell_df, question)


def candidate_index_weighted_degree_centrality(
    cell_df: pd.DataFrame,
    question: str,
) -> float:
    """CANDIDATE scalar B: in-scope share of total weighted degree.

    Computes the secondary scalar from spec §3:

        SI_wd(q, p) = Σ_{c∈S(q)} wd(c) / Σ_c wd(c)  = ½ Σ_{c∈S(q)} wd(c)

    CANDIDATE — retained for backward compatibility only.

    Parameters
    ----------
    cell_df:
        Long-format DataFrame for a single (question, protocol) cell.
    question:
        Question key used to look up S(q) from :data:`~.scope.IN_SCOPE`.

    Returns
    -------
    float in [0, 1], or ``float("nan")`` when the cell has no edges.
    """
    return _si_wd(cell_df, question)


def candidate_indices(
    cell_df: pd.DataFrame,
    question: str,
) -> dict[str, object]:
    """Return both CANDIDATE scalars and diagnostic counts for one cell.

    CANDIDATE — retained for backward compatibility only.

    Parameters
    ----------
    cell_df:
        Long-format DataFrame for a single (question, protocol) cell.
    question:
        Question key (e.g., "Q3.1").

    Returns
    -------
    dict with keys:
        ``si_tie_strength_concentration`` (float): primary SI scalar.
        ``si_weighted_degree_centrality``  (float): secondary SI_wd scalar.
        ``n_traces`` (int): number of distinct traces in the cell.
        ``n_edges``  (int): number of co-occurrence edges.
    """
    cooc = cooccurrence(cell_df)
    ts = tie_strength(cooc)
    scope = IN_SCOPE.get(question, frozenset())

    n_traces = int(cell_df["trace_id"].nunique())
    n_edges = len(ts)

    if not ts:
        return {
            "si_tie_strength_concentration": float("nan"),
            "si_weighted_degree_centrality": float("nan"),
            "n_traces": n_traces,
            "n_edges": n_edges,
        }

    in_scope_mass = sum(s for (a, b), s in ts.items() if a in scope or b in scope)
    wdeg = weighted_degree(ts)
    total_wdeg = sum(wdeg.values())
    in_scope_wdeg = sum(v for c, v in wdeg.items() if c in scope)

    return {
        "si_tie_strength_concentration": float(in_scope_mass),
        "si_weighted_degree_centrality": (
            float(in_scope_wdeg / total_wdeg) if total_wdeg else float("nan")
        ),
        "n_traces": n_traces,
        "n_edges": n_edges,
    }


def network_index_table(assignments: pd.DataFrame) -> pd.DataFrame:
    """Per-(question, protocol) CANDIDATE index point estimates.

    Iterates over every (question, protocol) cell in *assignments*, calls
    :func:`candidate_indices`, and returns a tidy DataFrame of point estimates.

    NOTE: Retained for backward compatibility.  For the locked formula use
    :func:`specificity_table`.

    Parameters
    ----------
    assignments:
        Long-format DataFrame with columns:
        patient_id, question, protocol, trace_id, claim_code.

    Returns
    -------
    DataFrame with columns: question, protocol, si_tie_strength_concentration,
    si_weighted_degree_centrality, n_traces, n_edges.
    """
    rows = []
    for (q, p), cell in assignments.groupby(["question", "protocol"]):
        rows.append({"question": q, "protocol": p, **candidate_indices(cell, q)})
    return pd.DataFrame(rows).sort_values(["question", "protocol"]).reset_index(drop=True)

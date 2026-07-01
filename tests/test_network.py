"""Tests for the network-analytic building blocks.

Ported from the reference implementation at:
Qualitative_Assessment/Meeting_070126/02_specificity_index/baek_specificity_index_network.py

Three property-based tests are covered:

1. tie_strength sums to ~1; weighted degree sums to ~2 (probability-distribution
   properties of the Sci-Rep-2025 normalization).
2. in-scope hub detection — a category that co-occurs broadly with many other
   categories should have the highest weighted degree (hub property).  NOTE:
   the share of weighted degree held by the in-scope node is NOT guaranteed to
   exceed 50% — that claim is not a mathematical theorem for arbitrary cell
   constructions (see test below for the correct property).
3. co-occurrence — cooccurrence counts are correct for a known input.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure the package root is on sys.path whether tests run via conftest or
# --noconftest (e.g., when pytest is invoked from a parent repo's venv).
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import pandas as pd
import pytest

from specificity_index.network import cooccurrence, tie_strength, weighted_degree
from specificity_index.scope import IN_SCOPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cell(traces: list[list[str]], question: str = "Q3.1") -> pd.DataFrame:
    """Build a minimal cell DataFrame from a list of per-trace category lists."""
    rows = []
    for i, codes in enumerate(traces):
        for code in codes:
            rows.append({
                "patient_id": "P001",
                "question": question,
                "protocol": "GR",
                "trace_id": f"T{i:04d}",
                "claim_code": code,
            })
    return pd.DataFrame(rows, columns=["patient_id", "question", "protocol", "trace_id", "claim_code"])


# ---------------------------------------------------------------------------
# Test 1: tie_strength sums to 1
# ---------------------------------------------------------------------------

class TestTieStrengthSumsToOne:
    """Tie strengths must sum to ~1 and weighted degree must sum to ~2.

    These are the normalization properties of the Sci-Rep-2025 tie-strength
    definition (spec §2):

        Σ_{i<j} ts(i,j) = 1   (probability distribution over edges)
        Σ_c wd(c)         = 2  (each edge contributes to two nodes)
    """

    def test_simple_three_trace_cell(self):
        # Three traces, each citing two categories — produces three distinct edges.
        traces = [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        assert ts, "Expected non-empty tie strengths"
        total = sum(ts.values())
        assert math.isclose(total, 1.0, abs_tol=1e-12), (
            f"Tie strengths should sum to 1.0, got {total}"
        )

    def test_single_dominant_pair(self):
        # All traces cite the same pair — only one edge, strength must be 1.
        traces = [["C1", "C2"]] * 10
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        assert len(ts) == 1
        total = sum(ts.values())
        assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_many_categories(self):
        # Five categories cited together in every trace — C(5,2)=10 edges.
        codes = ["C1", "C2", "C3", "C4", "C5"]
        traces = [codes for _ in range(20)]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        assert len(ts) == 10
        total = sum(ts.values())
        assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_weighted_degree_sums_to_two(self):
        # Σ_c wd(c) = 2 because each edge {a,b} contributes ts(a,b) to both
        # wd(a) and wd(b), so total = 2 * Σ ts = 2 * 1 = 2.
        traces = [["C1", "C2", "C3"], ["C1", "C4"], ["C2", "C4", "C5"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        wdeg = weighted_degree(ts)
        total = sum(wdeg.values())
        assert math.isclose(total, 2.0, abs_tol=1e-12), (
            f"Weighted degree should sum to 2.0, got {total}"
        )

    def test_weighted_degree_sums_to_two_symmetric_cell(self):
        # Simple 3-node clique: each node has wd = 2/3; total = 2.
        traces = [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        wdeg = weighted_degree(ts)
        total = sum(wdeg.values())
        assert math.isclose(total, 2.0, abs_tol=1e-12)

    def test_empty_cell_returns_empty(self):
        # A cell with no traces should yield empty dicts.
        cell = _make_cell([])
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        assert cooc == {}
        assert ts == {}

    def test_single_category_traces_no_edges(self):
        # Traces with only one category produce no pairs.
        traces = [["C1"], ["C2"], ["C3"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        assert cooc == {}
        assert ts == {}


# ---------------------------------------------------------------------------
# Test 2: in-scope hub detection
# ---------------------------------------------------------------------------

class TestInScopeHubDetection:
    """An in-scope category that co-occurs broadly should have high weighted degree.

    NOTE on the replaced test
    -------------------------
    A previous version of this class asserted that the in-scope node would hold
    >50% of the total weighted degree mass.  This is NOT a mathematical theorem:
    the weighted-degree share of one node depends on the full edge structure, and
    with off-scope edges present the in-scope node can have less than half the
    total mass.  The correct property is that the designed hub has the *highest*
    individual weighted degree (i.e. it is the maximum-degree node), which IS
    guaranteed by construction when C1 connects to more partners at equal or
    higher multiplicity than any other single node.
    """

    def test_inscope_hub_has_maximum_weighted_degree(self):
        # Q3.1's in-scope set is {C1}.
        # Construct a cell where C1 co-occurs with many other categories, making
        # it the hub by degree.  The remaining edges involve only out-of-scope.
        question = "Q3.1"
        scope = IN_SCOPE[question]  # {"C1"}

        # C1 co-occurs with C2, C3, C4, C5 across 8 traces (4 distinct edges,
        # each with count 2).
        hub_traces = [["C1", f"C{i}"] for i in range(2, 6)] * 2  # 8 traces
        # One additional off-scope edge (C2, C3) with count 1.
        oos_traces = [["C2", "C3"]]
        cell = _make_cell(hub_traces + oos_traces, question=question)

        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        wdeg = weighted_degree(ts)

        assert wdeg, "Expected non-empty weighted degree"
        # C1 connects to 4 partners; every other node connects to at most 2.
        # Therefore C1 must be the node with the highest weighted degree.
        max_node = max(wdeg, key=wdeg.__getitem__)
        assert max_node in scope, (
            f"Expected the in-scope hub (C1) to have maximum weighted degree; "
            f"got max node {max_node!r} with wd={wdeg[max_node]:.4f}"
        )

    def test_inscope_node_higher_than_each_individual_oos_node(self):
        # Weaker than >50% share but still a meaningful property: the in-scope
        # hub node C1 should have a strictly higher weighted degree than each
        # individual out-of-scope node when C1 has more edge connections.
        question = "Q3.1"
        scope = IN_SCOPE[question]  # {"C1"}

        hub_traces = [["C1", f"C{i}"] for i in range(2, 6)] * 2
        oos_traces = [["C2", "C3"]]
        cell = _make_cell(hub_traces + oos_traces, question=question)

        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        wdeg = weighted_degree(ts)

        c1_wd = wdeg["C1"]
        oos_nodes = [c for c in wdeg if c not in scope]
        for c in oos_nodes:
            assert c1_wd > wdeg[c], (
                f"C1 weighted degree ({c1_wd:.4f}) should exceed {c} "
                f"({wdeg[c]:.4f})"
            )

    def test_oos_dominant_cell_low_inscope_share(self):
        # Build a cell where out-of-scope categories co-occur densely and C1
        # (in-scope for Q3.1) is isolated.
        question = "Q3.1"
        scope = IN_SCOPE[question]  # {"C1"}

        # Dense out-of-scope clique: C2, C3, C4, C5 all co-occur.
        oos_traces = [["C2", "C3", "C4", "C5"]] * 10
        # One trace with C1 paired with a single other category.
        inscope_traces = [["C1", "C2"]]
        cell = _make_cell(oos_traces + inscope_traces, question=question)

        cooc = cooccurrence(cell)
        ts = tie_strength(cooc)
        wdeg = weighted_degree(ts)

        in_scope_wdeg = sum(v for c, v in wdeg.items() if c in scope)
        total_wdeg = sum(wdeg.values())
        share = in_scope_wdeg / total_wdeg
        # In-scope node contributes little when out-of-scope nodes dominate.
        assert share < 0.5, (
            f"Out-of-scope dominant cell: in-scope share should be <50%, got {share:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 3: co-occurrence correctness
# ---------------------------------------------------------------------------

class TestCooccurrence:
    """cooccurrence should count pair appearances across traces correctly."""

    def test_pair_counts_two_traces(self):
        # (C1, C2) appears twice; (C1, C3) and (C2, C3) appear once each.
        traces = [["C1", "C2", "C3"], ["C1", "C2"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        assert cooc[("C1", "C2")] == 2
        assert cooc[("C1", "C3")] == 1
        assert cooc[("C2", "C3")] == 1

    def test_pair_ordering_is_lexicographic(self):
        # Keys should always be (smaller, larger) lexicographically.
        traces = [["C2", "C1"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        assert ("C1", "C2") in cooc
        assert ("C2", "C1") not in cooc

    def test_duplicate_codes_in_trace_counted_once(self):
        # Duplicate claim_code rows within the same trace should not inflate counts.
        rows = [
            {"patient_id": "P1", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0001", "claim_code": "C1"},
            {"patient_id": "P1", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0001", "claim_code": "C1"},  # duplicate
            {"patient_id": "P1", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0001", "claim_code": "C2"},
        ]
        cell = pd.DataFrame(rows)
        cooc = cooccurrence(cell)
        # C1 appears twice in the same trace but should count once per trace.
        assert cooc.get(("C1", "C2"), 0) == 1
        assert ("C1", "C1") not in cooc

    def test_single_category_per_trace_no_pairs(self):
        traces = [["C1"], ["C2"], ["C3"]]
        cell = _make_cell(traces)
        cooc = cooccurrence(cell)
        assert cooc == {}

    def test_empty_dataframe(self):
        cell = _make_cell([])
        cooc = cooccurrence(cell)
        assert cooc == {}

    def test_multi_patient_aggregation(self):
        # Co-occurrence is aggregated across traces regardless of patient.
        rows = [
            {"patient_id": "P1", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0001", "claim_code": "C1"},
            {"patient_id": "P1", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0001", "claim_code": "C2"},
            {"patient_id": "P2", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0002", "claim_code": "C1"},
            {"patient_id": "P2", "question": "Q3.1", "protocol": "GR",
             "trace_id": "T0002", "claim_code": "C2"},
        ]
        cell = pd.DataFrame(rows)
        cooc = cooccurrence(cell)
        assert cooc[("C1", "C2")] == 2

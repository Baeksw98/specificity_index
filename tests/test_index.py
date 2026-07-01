"""Tests for the locked Specificity Index functions (gate B).

Covers:
1. SI range [0, 1].
2. SI == 1.0 when every edge is in-scope-incident (fully anchored cell).
3. SI == 0.0 when no edge touches S(q) (fully off-scope cell).
4. SI strictly lower on a diffuse off-scope cell than on a focused cell.
5. Efficiency identity: Σ_c ψ_c == SI (abs_tol 1e-9).
6. Anchoring Credit (AC) is non-negative and only contains nonzero entries.
7. specificity_table produces SI_GR > SI_NonGR for a well-designed synthetic
   dataset (the GR protocol concentrates co-occurrence on S(q)).
8. Empty-cell contract: SI == nan; anchoring_credit == {}.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import pandas as pd
import pytest

from specificity_index.index import anchoring_credit, specificity_index, specificity_table
from specificity_index.scope import IN_SCOPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cell(
    traces: list[list[str]],
    question: str = "Q3.1",
    protocol: str = "GR",
) -> pd.DataFrame:
    """Build a minimal cell DataFrame from a list of per-trace category lists."""
    rows = []
    for i, codes in enumerate(traces):
        for code in codes:
            rows.append({
                "patient_id": f"P{i:03d}",
                "question": question,
                "protocol": protocol,
                "trace_id": f"T{i:04d}",
                "claim_code": code,
            })
    if not rows:
        return pd.DataFrame(
            columns=["patient_id", "question", "protocol", "trace_id", "claim_code"]
        )
    return pd.DataFrame(rows)


def _make_assignments(
    gr_traces: list[list[str]],
    non_gr_traces: list[list[str]],
    question: str = "Q3.1",
) -> pd.DataFrame:
    """Build a combined assignments DataFrame with both GR and Non-GR cells."""
    gr_df = _make_cell(gr_traces, question=question, protocol="GR")
    non_gr_df = _make_cell(non_gr_traces, question=question, protocol="Non-GR")
    return pd.concat([gr_df, non_gr_df], ignore_index=True)


# ---------------------------------------------------------------------------
# 1 & 2. Range and fully-anchored cell
# ---------------------------------------------------------------------------

class TestSpecificityIndexRange:
    """SI must lie in [0, 1]; boundary values achievable by construction."""

    def test_range_mixed_cell(self):
        # Q3.1: S(q) = {C1}.  Mix of in-scope and off-scope edges.
        traces = [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]]
        cell = _make_cell(traces, question="Q3.1")
        si = specificity_index(cell, "Q3.1")
        assert not math.isnan(si), "SI should not be nan for a cell with edges"
        assert 0.0 <= si <= 1.0, f"SI out of range: {si}"

    def test_fully_anchored_cell_si_equals_one(self):
        # Q3.1: S(q) = {C1}.
        # Every trace cites C1 (the only in-scope category) plus one OOS category.
        # All edges are {C1, Cx} — all in-scope-incident.
        # Therefore SI = 1.0.
        traces = [["C1", "C2"], ["C1", "C3"], ["C1", "C4"], ["C1", "C5"]]
        cell = _make_cell(traces, question="Q3.1")
        si = specificity_index(cell, "Q3.1")
        assert math.isclose(si, 1.0, abs_tol=1e-12), (
            f"Fully anchored cell should give SI=1.0, got {si}"
        )

    def test_fully_anchored_multi_inscope_cell_si_equals_one(self):
        # Q3.5: S(q) = {C5, C6, C7} (multiple in-scope categories).
        # Traces cite only in-scope categories — all edges are trivially
        # in-scope-incident, so SI = 1.0.
        traces = [["C5", "C6"], ["C5", "C7"], ["C6", "C7"], ["C5", "C6", "C7"]]
        cell = _make_cell(traces, question="Q3.5")
        si = specificity_index(cell, "Q3.5")
        assert math.isclose(si, 1.0, abs_tol=1e-12), (
            f"All-inscope cell should give SI=1.0, got {si}"
        )

    def test_fully_offscope_cell_si_equals_zero(self):
        # Q3.1: S(q) = {C1}.
        # No trace cites C1 — no edge touches S(q) — SI = 0.0.
        traces = [["C2", "C3"], ["C4", "C5"], ["C2", "C4"]]
        cell = _make_cell(traces, question="Q3.1")
        si = specificity_index(cell, "Q3.1")
        assert math.isclose(si, 0.0, abs_tol=1e-12), (
            f"Fully off-scope cell should give SI=0.0, got {si}"
        )

    def test_diffuse_oos_cell_has_lower_si_than_focused_inscope_cell(self):
        # Focused: C1 co-occurs with many categories — high SI.
        focused_traces = [["C1", "C2"], ["C1", "C3"], ["C1", "C4"], ["C1", "C5"]]
        # Diffuse: Dense OOS clique plus a single C1 trace — low SI.
        diffuse_traces = [["C2", "C3", "C4", "C5"]] * 8 + [["C1", "C2"]]
        focused_cell = _make_cell(focused_traces, question="Q3.1")
        diffuse_cell = _make_cell(diffuse_traces, question="Q3.1")
        si_focused = specificity_index(focused_cell, "Q3.1")
        si_diffuse = specificity_index(diffuse_cell, "Q3.1")
        assert si_focused > si_diffuse, (
            f"Focused SI ({si_focused:.4f}) should exceed diffuse SI ({si_diffuse:.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Empty-cell contract
# ---------------------------------------------------------------------------

class TestEmptyCellContract:
    """An empty cell (no edges) must return nan / empty dict."""

    def test_si_nan_when_no_traces(self):
        cell = _make_cell([], question="Q3.1")
        si = specificity_index(cell, "Q3.1")
        assert math.isnan(si), "SI should be nan when no traces exist"

    def test_si_nan_when_single_category_per_trace(self):
        # Single-category traces produce no pairs.
        traces = [["C1"], ["C2"], ["C3"]]
        cell = _make_cell(traces, question="Q3.1")
        si = specificity_index(cell, "Q3.1")
        assert math.isnan(si), "SI should be nan when no edges exist"

    def test_anchoring_credit_empty_when_no_edges(self):
        cell = _make_cell([], question="Q3.1")
        ac = anchoring_credit(cell, "Q3.1")
        assert ac == {}, f"AC should be empty dict when no edges, got {ac}"

    def test_anchoring_credit_empty_single_category_traces(self):
        traces = [["C1"], ["C2"]]
        cell = _make_cell(traces, question="Q3.1")
        ac = anchoring_credit(cell, "Q3.1")
        assert ac == {}


# ---------------------------------------------------------------------------
# 4. Efficiency identity: Σ_c ψ_c == SI
# ---------------------------------------------------------------------------

class TestEfficiencyIdentity:
    """Σ_c anchoring_credit[c] == specificity_index for every cell (spec §4).

    This is the key index-decomposition property of the Anchoring Credit (AC):
    ψ_c is a descriptive attribution of observed connective mass with the
    Shapley efficiency axiom (spec §4).
    """

    def _check_efficiency(self, traces: list[list[str]], question: str) -> None:
        cell = _make_cell(traces, question=question)
        si = specificity_index(cell, question)
        ac = anchoring_credit(cell, question)
        if math.isnan(si):
            assert ac == {}, "AC must be empty when SI is nan"
            return
        ac_sum = sum(ac.values())
        assert math.isclose(ac_sum, si, abs_tol=1e-9), (
            f"Efficiency violated for {question}: Σψ_c={ac_sum:.12f} vs SI={si:.12f}"
        )

    def test_efficiency_fully_anchored_cell(self):
        # All edges touch C1 (in-scope for Q3.1) — SI=1.0, Σψ_c must be 1.0.
        traces = [["C1", "C2"], ["C1", "C3"], ["C1", "C4"]]
        self._check_efficiency(traces, "Q3.1")

    def test_efficiency_fully_offscope_cell(self):
        # No edge touches C1 — SI=0.0, AC should be empty or sum to 0.
        traces = [["C2", "C3"], ["C4", "C5"]]
        self._check_efficiency(traces, "Q3.1")

    def test_efficiency_mixed_cell(self):
        # Mixed in-scope and off-scope edges.
        traces = [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]]
        self._check_efficiency(traces, "Q3.1")

    def test_efficiency_multi_inscope_question(self):
        # Q3.5: S(q) = {C5, C6, C7}.
        traces = [["C5", "C6"], ["C5", "C7"], ["C6", "C7"],
                  ["C8", "C9"], ["C5", "C8"]]
        self._check_efficiency(traces, "Q3.5")

    def test_efficiency_dense_mixed_cell(self):
        # Dense clique of 5 categories, 2 in-scope (for Q3.5: C5, C6).
        traces = [["C5", "C6", "C7", "C8", "C9"]] * 5
        self._check_efficiency(traces, "Q3.5")

    def test_efficiency_single_edge(self):
        # Trivial case: one edge, one or zero endpoints in S(q).
        # Q3.1 (S={C1}): edge C1-C2 → SI=1.0, Σψ=1.0.
        self._check_efficiency([["C1", "C2"]], "Q3.1")
        # Edge C2-C3 → SI=0.0, AC empty.
        self._check_efficiency([["C2", "C3"]], "Q3.1")


# ---------------------------------------------------------------------------
# 5. Anchoring Credit non-negativity and nonzero-only entries
# ---------------------------------------------------------------------------

class TestAnchoringCreditProperties:
    """AC values must be non-negative; returned dict contains only nonzero entries."""

    def test_ac_nonnegative(self):
        traces = [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]]
        cell = _make_cell(traces, question="Q3.1")
        ac = anchoring_credit(cell, "Q3.1")
        for c, v in ac.items():
            assert v >= 0.0, f"AC for {c} is negative: {v}"

    def test_ac_only_nonzero_entries(self):
        traces = [["C2", "C3"], ["C4", "C5"]]  # all OOS for Q3.1
        cell = _make_cell(traces, question="Q3.1")
        ac = anchoring_credit(cell, "Q3.1")
        # SI=0 → AC should be empty (no in-scope-incident edge)
        for c, v in ac.items():
            assert v != 0.0, f"AC returned a zero entry for {c}"

    def test_ac_covers_both_endpoints_of_inscope_incident_edge(self):
        # Edge {C1, C2}: C1 in S(q), so edge is in-scope-incident.
        # Both C1 and C2 should receive AC = 0.5 * ts(C1,C2).
        traces = [["C1", "C2"], ["C1", "C2"]]  # one edge, count=2
        cell = _make_cell(traces, question="Q3.1")
        ac = anchoring_credit(cell, "Q3.1")
        # Only one edge, ts = 1.0; each endpoint gets 0.5.
        assert math.isclose(ac.get("C1", 0.0), 0.5, abs_tol=1e-12), (
            f"C1 (in-scope) should have AC=0.5, got {ac.get('C1')}"
        )
        assert math.isclose(ac.get("C2", 0.0), 0.5, abs_tol=1e-12), (
            f"C2 (out-of-scope endpoint of in-scope-incident edge) should have "
            f"AC=0.5, got {ac.get('C2')}"
        )

    def test_ac_oos_node_zero_when_no_inscope_incident_edge(self):
        # Pure OOS edge: no in-scope-incident edges, so no AC.
        traces = [["C2", "C3"]]
        cell = _make_cell(traces, question="Q3.1")
        ac = anchoring_credit(cell, "Q3.1")
        assert "C2" not in ac
        assert "C3" not in ac


# ---------------------------------------------------------------------------
# 6. specificity_table: GR > Non-GR for a well-designed synthetic dataset
# ---------------------------------------------------------------------------

class TestSpecificityTable:
    """specificity_table should discriminate GR from Non-GR when SI_GR > SI_NonGR."""

    def test_gr_higher_si_than_non_gr(self):
        # Q3.1: S(q) = {C1}.
        # GR: C1 co-occurs with everything — fully anchored (SI should be 1.0).
        gr_traces = [["C1", "C2"], ["C1", "C3"], ["C1", "C4"], ["C1", "C5"]] * 3
        # Non-GR: dense OOS clique, C1 appears rarely — low SI.
        non_gr_traces = [["C2", "C3", "C4", "C5"]] * 6 + [["C1", "C2"]]
        df = _make_assignments(gr_traces, non_gr_traces, question="Q3.1")
        table = specificity_table(df)

        assert len(table) == 1
        row = table.iloc[0]
        assert row["question"] == "Q3.1"
        si_gr = row["SI_GR"]
        si_non_gr = row["SI_Non-GR"]
        assert not math.isnan(si_gr), "SI_GR should not be nan"
        assert not math.isnan(si_non_gr), "SI_NonGR should not be nan"
        assert si_gr > si_non_gr, (
            f"Expected SI_GR ({si_gr:.4f}) > SI_NonGR ({si_non_gr:.4f})"
        )
        assert math.isclose(row["delta_SI"], si_gr - si_non_gr, abs_tol=1e-12), (
            "delta_SI must equal SI_GR - SI_NonGR"
        )

    def test_table_delta_si_equals_gr_minus_non_gr(self):
        gr_traces = [["C1", "C2"], ["C1", "C3"], ["C2", "C3"]]
        non_gr_traces = [["C2", "C3"], ["C4", "C5"], ["C2", "C4"]]
        df = _make_assignments(gr_traces, non_gr_traces, question="Q3.1")
        table = specificity_table(df)
        row = table.iloc[0]
        si_gr = row["SI_GR"]
        si_non_gr = row["SI_Non-GR"]
        expected_delta = si_gr - si_non_gr
        assert math.isclose(row["delta_SI"], expected_delta, abs_tol=1e-12), (
            f"delta_SI mismatch: got {row['delta_SI']:.12f}, expected {expected_delta:.12f}"
        )

    def test_table_contains_secondary_si_wd_columns(self):
        gr_traces = [["C1", "C2"], ["C1", "C3"]]
        non_gr_traces = [["C2", "C3"], ["C4", "C5"]]
        df = _make_assignments(gr_traces, non_gr_traces, question="Q3.1")
        table = specificity_table(df)
        assert "SI_wd_GR [secondary]" in table.columns, (
            "Table should contain the secondary SI_wd column for GR"
        )
        assert "SI_wd_Non-GR [secondary]" in table.columns, (
            "Table should contain the secondary SI_wd column for Non-GR"
        )

    def test_table_nan_when_no_edges_in_cell(self):
        # Non-GR cell has no multi-category traces → no edges → SI = nan.
        gr_traces = [["C1", "C2"], ["C1", "C3"]]
        non_gr_traces = [["C1"], ["C2"], ["C3"]]  # single-category only
        df = _make_assignments(gr_traces, non_gr_traces, question="Q3.1")
        table = specificity_table(df)
        row = table.iloc[0]
        assert math.isnan(row["SI_Non-GR"]), (
            "SI_NonGR should be nan when Non-GR cell has no edges"
        )
        assert math.isnan(row["delta_SI"]), (
            "delta_SI should be nan when either SI is nan"
        )

    def test_table_one_row_per_question(self):
        # Two questions; table should have two rows.
        rows = []
        for q, in_sc, oos in [("Q3.1", "C1", "C2"), ("Q3.2", "C2", "C3")]:
            for p in ["GR", "Non-GR"]:
                for i in range(3):
                    rows.append({
                        "patient_id": f"P{i}",
                        "question": q,
                        "protocol": p,
                        "trace_id": f"T{i}_{q}_{p}",
                        "claim_code": in_sc if p == "GR" else oos,
                    })
                    rows.append({
                        "patient_id": f"P{i}",
                        "question": q,
                        "protocol": p,
                        "trace_id": f"T{i}_{q}_{p}",
                        "claim_code": oos,
                    })
        df = pd.DataFrame(rows)
        table = specificity_table(df)
        assert len(table) == 2, f"Expected 2 rows, got {len(table)}"
        assert set(table["question"]) == {"Q3.1", "Q3.2"}

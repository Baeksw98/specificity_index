"""Tests for specificity_bootstrap (spec §6, patient-clustered bootstrap).

Synthetic dataset tests:
1. Point estimates from specificity_bootstrap match specificity_index().
2. delta_SI point = SI_GR - SI_Non-GR (within float precision).
3. CI is well-formed: lo <= hi (non-empty).
4. p_value lies in [0, 1] (non-NaN for cells with edges).
5. p_adj >= p_value (BH always inflates or keeps equal).
6. BH correction: running with two questions produces p_adj values.
7. Runs with small n_boot (no error, correct shape).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import pandas as pd
import pytest

from specificity_index import specificity_index, specificity_bootstrap


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def _make_synthetic_df(seed: int = 0) -> pd.DataFrame:
    """Build a small but realistic synthetic dataset.

    5 patients, 2 questions (Q3.1, Q3.2), 2 protocols (GR, Non-GR).
    Each patient has 6–10 traces per (question, protocol); each trace
    cites 2–5 categories from C1–C12.  No NONE rows are included.
    """
    rng = np.random.default_rng(seed)
    patients = [f"P{i:02d}" for i in range(5)]
    questions = ["Q3.1", "Q3.2"]
    protocols = ["GR", "Non-GR"]
    categories = [f"C{i}" for i in range(1, 13)]

    rows: list[dict] = []
    for pat in patients:
        for q in questions:
            for p in protocols:
                n_traces = int(rng.integers(6, 11))
                for t in range(n_traces):
                    n_cats = int(rng.integers(2, 6))
                    cats = rng.choice(categories, size=n_cats, replace=False)
                    trace_id = f"{pat}_{q}_{p}_{t}"
                    for cat in cats:
                        rows.append(
                            {
                                "patient_id": pat,
                                "question": q,
                                "protocol": p,
                                "trace_id": trace_id,
                                "claim_code": cat,
                                "score": 1.0,
                                "score_bucket": "high",
                                "stance_code": "S1",
                            }
                        )
    return pd.DataFrame(rows)


_DF = _make_synthetic_df(seed=0)
_RESULT = specificity_bootstrap(_DF, n_boot=100, seed=42)


# ---------------------------------------------------------------------------
# 1. Point estimates match specificity_index()
# ---------------------------------------------------------------------------

class TestPointEstimatesMatch:
    """Bootstrap point estimates must equal specificity_index() for each cell."""

    @pytest.mark.parametrize("q", ["Q3.1", "Q3.2"])
    @pytest.mark.parametrize(
        "p_label, col",
        [("GR", "SI_GR"), ("Non-GR", "SI_Non-GR")],
    )
    def test_point_estimate_matches_specificity_index(
        self, q: str, p_label: str, col: str
    ) -> None:
        cell_df = _DF[
            (_DF["question"] == q)
            & (_DF["protocol"] == p_label)
            & (_DF["claim_code"] != "NONE")
        ]
        expected = specificity_index(cell_df, q)
        row = _RESULT[_RESULT["question"] == q].iloc[0]
        got = float(row[col])

        if math.isnan(expected):
            assert math.isnan(got), (
                f"Expected NaN for {q}/{p_label}, got {got}"
            )
        else:
            assert abs(got - expected) < 1e-9, (
                f"Point-estimate mismatch for {q}/{p_label}: "
                f"bootstrap={got:.12f} vs specificity_index={expected:.12f}"
            )


# ---------------------------------------------------------------------------
# 2. delta_SI = SI_GR − SI_Non-GR
# ---------------------------------------------------------------------------

class TestDeltaSI:
    @pytest.mark.parametrize("q", ["Q3.1", "Q3.2"])
    def test_delta_si_equals_difference(self, q: str) -> None:
        row = _RESULT[_RESULT["question"] == q].iloc[0]
        si_gr = float(row["SI_GR"])
        si_non = float(row["SI_Non-GR"])
        delta = float(row["delta_SI"])
        if math.isnan(si_gr) or math.isnan(si_non):
            assert math.isnan(delta), "delta_SI must be NaN when either SI is NaN"
        else:
            assert math.isclose(delta, si_gr - si_non, abs_tol=1e-12), (
                f"delta_SI mismatch for {q}: {delta} != {si_gr} - {si_non}"
            )


# ---------------------------------------------------------------------------
# 3. CI is well-formed: lo <= hi
# ---------------------------------------------------------------------------

class TestCIWellFormed:
    @pytest.mark.parametrize("q", ["Q3.1", "Q3.2"])
    @pytest.mark.parametrize(
        "prefix",
        ["SI_GR", "SI_Non-GR", "delta_SI"],
    )
    def test_ci_lo_le_hi(self, q: str, prefix: str) -> None:
        row = _RESULT[_RESULT["question"] == q].iloc[0]
        lo = float(row[f"{prefix}_lo"])
        hi = float(row[f"{prefix}_hi"])
        if math.isnan(lo) or math.isnan(hi):
            return  # NaN is acceptable when there are no edges
        assert lo <= hi + 1e-12, (
            f"CI is empty for {q}/{prefix}: lo={lo:.6f} > hi={hi:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. p_value in [0, 1]
# ---------------------------------------------------------------------------

class TestPValue:
    @pytest.mark.parametrize("q", ["Q3.1", "Q3.2"])
    def test_p_value_in_unit_interval(self, q: str) -> None:
        row = _RESULT[_RESULT["question"] == q].iloc[0]
        pval = float(row["p_value"])
        if math.isnan(pval):
            return  # NaN only when CI could not be computed
        assert 0.0 <= pval <= 1.0, f"p_value out of [0,1] for {q}: {pval}"


# ---------------------------------------------------------------------------
# 5. p_adj >= p_value (BH inflates)
# ---------------------------------------------------------------------------

class TestBHFDR:
    @pytest.mark.parametrize("q", ["Q3.1", "Q3.2"])
    def test_p_adj_ge_p_value(self, q: str) -> None:
        row = _RESULT[_RESULT["question"] == q].iloc[0]
        pval = float(row["p_value"])
        padj = float(row["p_adj"])
        if math.isnan(pval) or math.isnan(padj):
            return
        # BH adjusted p-value is always >= raw p-value (step-up procedure).
        assert padj >= pval - 1e-12, (
            f"p_adj ({padj:.6f}) < p_value ({pval:.6f}) for {q}"
        )

    def test_p_adj_column_exists(self) -> None:
        assert "p_adj" in _RESULT.columns

    def test_result_has_one_row_per_question(self) -> None:
        assert len(_RESULT) == 2
        assert set(_RESULT["question"]) == {"Q3.1", "Q3.2"}


# ---------------------------------------------------------------------------
# 6. Deterministic: same seed → same result
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        r1 = specificity_bootstrap(_DF, n_boot=20, seed=123)
        r2 = specificity_bootstrap(_DF, n_boot=20, seed=123)
        for col in r1.columns:
            if col == "question":
                continue
            v1 = r1[col].values
            v2 = r2[col].values
            both_nan = np.isnan(v1.astype(float)) & np.isnan(v2.astype(float))
            np.testing.assert_array_equal(
                v1[~both_nan], v2[~both_nan],
                err_msg=f"Column {col} differs between identical seeds",
            )


# ---------------------------------------------------------------------------
# 7. NONE rows are dropped internally
# ---------------------------------------------------------------------------

class TestNoneFiltering:
    def test_none_rows_ignored(self) -> None:
        df_with_none = _DF.copy()
        # Inject NONE rows that should be silently discarded.
        extra_rows = _DF.iloc[:20].copy()
        extra_rows["claim_code"] = "NONE"
        df_with_none = pd.concat([df_with_none, extra_rows], ignore_index=True)

        r_without = specificity_bootstrap(_DF, n_boot=20, seed=7)
        r_with = specificity_bootstrap(df_with_none, n_boot=20, seed=7)

        for col in ["SI_GR", "SI_Non-GR", "delta_SI"]:
            np.testing.assert_array_almost_equal(
                r_without[col].values,
                r_with[col].values,
                decimal=12,
                err_msg=f"NONE rows should not affect point estimates ({col})",
            )


# ---------------------------------------------------------------------------
# 8. Fully-anchored cell: SI = 1 everywhere → delta CI includes 0
# ---------------------------------------------------------------------------

class TestFullyAnchoredCell:
    """When GR is fully anchored and Non-GR is not, delta_SI > 0 is expected."""

    def _make_contrasted_df(self) -> pd.DataFrame:
        patients = [f"P{i:02d}" for i in range(4)]
        rows: list[dict] = []
        for pat in patients:
            # GR: all traces cite C1 (in-scope for Q3.1) + one OOS → SI_GR = 1.0
            for t in range(5):
                for cat in ["C1", "C2"]:
                    rows.append(
                        dict(
                            patient_id=pat, question="Q3.1", protocol="GR",
                            trace_id=f"{pat}_GR_{t}", claim_code=cat,
                            score=1.0, score_bucket="high", stance_code="S1",
                        )
                    )
            # Non-GR: dense OOS clique → SI_NonGR < 1.0
            for t in range(5):
                for cat in ["C2", "C3", "C4", "C5"]:
                    rows.append(
                        dict(
                            patient_id=pat, question="Q3.1", protocol="Non-GR",
                            trace_id=f"{pat}_NGR_{t}", claim_code=cat,
                            score=1.0, score_bucket="high", stance_code="S1",
                        )
                    )
        return pd.DataFrame(rows)

    def test_gr_higher_si(self) -> None:
        df = self._make_contrasted_df()
        result = specificity_bootstrap(df, n_boot=50, seed=0)
        row = result[result["question"] == "Q3.1"].iloc[0]
        assert float(row["SI_GR"]) > float(row["SI_Non-GR"]), (
            "Expected SI_GR > SI_Non-GR for this contrasted dataset"
        )
        assert float(row["delta_SI"]) > 0, "delta_SI should be positive"

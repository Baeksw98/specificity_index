"""Patient-clustered nonparametric bootstrap for Baek's Specificity Index.

Implements the inference procedure from formal spec §6 (logic-locked):

  Estimand  : population mean SI(q,p) over the trace-generating process.
  Inference  : resample the 40 patients with replacement; rebuild network;
               recompute SI(q,p) and ΔSI(q) = SI_GR − SI_NonGR.
  BH-FDR    : Benjamini–Hochberg FDR correction across the 15 questions.

Efficient algorithm (one-pass precompute, then O(n_boot × n_cells) numpy):
  1. One pass: for each (question, protocol, patient) and each trace, form
     the distinct cited category set, accumulate per-patient co-occurrence
     contributions: cooc_contrib[(q,p,patient)][(cat_a, cat_b)] += 1 for
     every unordered pair of distinct categories co-cited in a trace.
  2. Build numpy matrices: for each (q,p) cell, a (n_patients × n_pairs)
     integer matrix whose row p is patient p's contribution vector.
  3. Point estimate: column-sum the matrix → total co-occurrence counts;
     SI = in_scope_count / total_count.
  4. Bootstrap: draw n_patients row indices with replacement (numpy
     default_rng), fancy-index the matrix, column-sum → boot SI.

References
----------
Formal spec §6: docs/program_planning/specificity_index_FORMAL_SPEC.md
Baek S, Kim K, et al. Sci Rep. 2025;15:5415.
doi: 10.1038/s41598-025-89340-2
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Mapping

import numpy as np
import pandas as pd

from .scope import IN_SCOPE


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR correction
# ---------------------------------------------------------------------------

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg step-up FDR correction (no NaN in input).

    Parameters
    ----------
    pvals : 1-D float array of raw p-values (must be NaN-free).

    Returns
    -------
    1-D float array of BH-adjusted p-values, clipped to [0, 1].
    """
    n = len(pvals)
    order = np.argsort(pvals)
    pvals_sorted = pvals[order]

    adjusted = np.zeros(n)
    cummin = np.inf
    for i in range(n - 1, -1, -1):
        bh = pvals_sorted[i] * n / (i + 1)
        cummin = min(cummin, bh)
        adjusted[i] = min(cummin, 1.0)

    result = np.empty(n)
    result[order] = adjusted
    return result


# ---------------------------------------------------------------------------
# Percentile CI helper
# ---------------------------------------------------------------------------

def _percentile_ci(
    arr: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    """Return (lo_pct, hi_pct) after dropping NaNs; (nan, nan) if empty."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


# ---------------------------------------------------------------------------
# Step 1: one-pass precomputation of per-patient co-occurrence contributions
# ---------------------------------------------------------------------------

def _precompute_cooc_contributions(
    df: pd.DataFrame,
) -> tuple[list[str], dict[tuple[str, str, str], dict[tuple[str, str], int]]]:
    """Precompute per-(question, protocol, patient) co-occurrence contributions.

    For each trace in the data, forms the distinct cited category set
    (identical logic to network.cooccurrence / network._trace_category_sets)
    and increments the co-occurrence count for every unordered pair of
    distinct categories that co-appear in that trace.  The count is stored
    under the patient that owns the trace so that patient contributions can
    be independently summed or resampled.

    The caller must have already filtered rows where claim_code == "NONE".

    Parameters
    ----------
    df : DataFrame with columns patient_id, question, protocol, trace_id,
         claim_code.  Rows with claim_code == "NONE" must already be removed.

    Returns
    -------
    patients : sorted list of all unique patient IDs found in df.
    cooc_contrib : dict mapping (question, protocol, patient_id) to a
                   plain dict mapping (cat_a, cat_b) → int count, where
                   cat_a < cat_b lexicographically and count is the number
                   of traces in that (question, protocol, patient_id) cell
                   that co-cite both cat_a and cat_b.
    """
    # Deduplicate: each (q, p, patient, trace, claim_code) row must be unique
    # (matches network._trace_category_sets which uses .unique() within a trace).
    df_dedup = df.drop_duplicates(
        ["question", "protocol", "patient_id", "trace_id", "claim_code"]
    )

    # Sort so that consecutive rows with the same (q,p,patient,trace) are
    # adjacent, and claim_codes within a trace appear in lexicographic order
    # (matches itertools.combinations(sorted(codes), 2) in network.py).
    df_s = df_dedup.sort_values(
        ["question", "protocol", "patient_id", "trace_id", "claim_code"],
        kind="mergesort",
    )

    q_arr = df_s["question"].to_numpy(dtype=object)
    p_arr = df_s["protocol"].to_numpy(dtype=object)
    pat_arr = df_s["patient_id"].to_numpy(dtype=object)
    trace_arr = df_s["trace_id"].to_numpy(dtype=object)
    cat_arr = df_s["claim_code"].to_numpy(dtype=object)
    n = len(df_s)

    patients = sorted(df["patient_id"].unique().tolist())

    if n == 0:
        return patients, {}

    # Locate trace boundaries using integer codes (fast vectorized comparison).
    q_codes = pd.factorize(q_arr, sort=False)[0]
    p_codes = pd.factorize(p_arr, sort=False)[0]
    pat_codes = pd.factorize(pat_arr, sort=False)[0]
    trace_codes = pd.factorize(trace_arr, sort=False)[0]

    boundary_mask = (
        (q_codes[:-1] != q_codes[1:])
        | (p_codes[:-1] != p_codes[1:])
        | (pat_codes[:-1] != pat_codes[1:])
        | (trace_codes[:-1] != trace_codes[1:])
    )
    changes = np.where(boundary_mask)[0] + 1
    trace_starts = np.concatenate([[0], changes])
    trace_ends = np.concatenate([changes, [n]])

    # Accumulate per-patient co-occurrence contributions.
    cooc_contrib: dict[
        tuple[str, str, str], dict[tuple[str, str], int]
    ] = defaultdict(lambda: defaultdict(int))

    for i in range(len(trace_starts)):
        s = int(trace_starts[i])
        e = int(trace_ends[i])
        # cats is already sorted (lexicographic) due to sort above.
        cats = cat_arr[s:e]
        if len(cats) < 2:
            continue  # single-category trace: no pairs (matches G2 in network.py)
        key = (str(q_arr[s]), str(p_arr[s]), str(pat_arr[s]))
        for a, b in itertools.combinations(cats, 2):
            cooc_contrib[key][(str(a), str(b))] += 1

    return patients, {k: dict(v) for k, v in cooc_contrib.items()}


# ---------------------------------------------------------------------------
# Step 2: build per-cell numpy matrices
# ---------------------------------------------------------------------------

def _build_cell_matrices(
    patients: list[str],
    cooc_contrib: dict[tuple[str, str, str], dict[tuple[str, str], int]],
    questions: list[str],
    protocols: list[str],
    in_scope: Mapping[str, frozenset[str]],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Build (n_patients × n_pairs) integer matrices per (question, protocol).

    Each row of the matrix is a patient's co-occurrence contribution vector
    for that cell.  Column-summing the matrix reproduces the whole-cell
    co-occurrence counts used by specificity_index().

    Returns
    -------
    cell_mats : dict mapping (question, protocol) to
                  (matrix, in_scope_mask), where:
                    matrix        — np.int64 array, shape (n_patients, n_pairs).
                    in_scope_mask — np.bool_ array, shape (n_pairs,).
                    n_pairs == 0 when the cell has no co-occurrence data.
    """
    n_patients = len(patients)
    pat_idx = {pat: i for i, pat in enumerate(patients)}

    cell_mats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for q in questions:
        scope = in_scope.get(q, frozenset())
        for p in protocols:
            # Collect all pairs seen across all patients for this cell.
            all_pairs: set[tuple[str, str]] = set()
            for pat in patients:
                key = (q, p, pat)
                if key in cooc_contrib:
                    all_pairs.update(cooc_contrib[key].keys())

            if not all_pairs:
                empty = np.zeros((n_patients, 0), dtype=np.int64)
                cell_mats[(q, p)] = (empty, np.zeros(0, dtype=bool))
                continue

            pairs = sorted(all_pairs)
            pair_idx = {pr: i for i, pr in enumerate(pairs)}
            n_pairs = len(pairs)

            matrix = np.zeros((n_patients, n_pairs), dtype=np.int64)
            for pat in patients:
                pi = pat_idx[pat]
                key = (q, p, pat)
                if key in cooc_contrib:
                    for pr, cnt in cooc_contrib[key].items():
                        matrix[pi, pair_idx[pr]] = cnt

            in_scope_mask = np.array(
                [a in scope or b in scope for (a, b) in pairs], dtype=bool
            )
            cell_mats[(q, p)] = (matrix, in_scope_mask)

    return cell_mats


# ---------------------------------------------------------------------------
# SI from a count vector
# ---------------------------------------------------------------------------

def _si_from_counts(
    counts: np.ndarray, in_scope_mask: np.ndarray
) -> float:
    """Compute SI = in_scope_count / total_count (NaN when total == 0)."""
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    in_scope_total = int(counts[in_scope_mask].sum())
    return in_scope_total / total


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def specificity_bootstrap(
    assignments_df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 20260701,
    gr_label: str = "GR",
    non_gr_label: str = "Non-GR",
    in_scope: Mapping[str, frozenset[str]] = IN_SCOPE,
) -> pd.DataFrame:
    """Patient-clustered nonparametric bootstrap for Baek's Specificity Index.

    Implements spec §6 (logic-locked):

    1. Precompute per-(q, p, patient) co-occurrence contributions in one pass.
    2. Point estimates: sum all patients' contributions, compute SI.
    3. Bootstrap (n_boot replicates, seed via numpy default_rng):
       draw n_patients indices with replacement; sum those rows; recompute
       SI(q,p) and ΔSI(q) = SI_GR − SI_NonGR.
    4. For each question: 2.5/97.5 percentile CIs; two-sided bootstrap
       p-value = 2 · min(mean(ΔSI* > 0), mean(ΔSI* < 0)) clipped to [0, 1].
    5. Benjamini–Hochberg FDR across the 15 questions' raw p-values.

    Parameters
    ----------
    assignments_df : long-format DataFrame with columns patient_id, question,
                     protocol, trace_id, claim_code.  Rows where
                     claim_code == "NONE" are filtered out internally.
    n_boot : number of bootstrap replicates (default 2000).
    seed : seed for numpy.random.default_rng (default 20260701).
    gr_label : protocol label for the GR arm (default "GR").
    non_gr_label : protocol label for the Non-GR arm (default "Non-GR").
    in_scope : mapping from question key to frozenset of in-scope category
               codes; defaults to scope.IN_SCOPE.

    Returns
    -------
    pd.DataFrame with one row per question and columns:
        question,
        SI_{gr_label},    SI_{gr_label}_lo,    SI_{gr_label}_hi,
        SI_{non_gr_label}, SI_{non_gr_label}_lo, SI_{non_gr_label}_hi,
        delta_SI, delta_SI_lo, delta_SI_hi,
        p_value, p_adj
    where _lo/_hi are 2.5th/97.5th bootstrap percentile CIs and p_adj is
    the BH-FDR adjusted p-value.

    References
    ----------
    Formal spec §6; Baek S, et al. Sci Rep. 2025;15:5415.
    doi: 10.1038/s41598-025-89340-2
    """
    # ------------------------------------------------------------------ #
    # Pre-processing: filter NONE, keep only needed columns               #
    # ------------------------------------------------------------------ #
    df = assignments_df.loc[
        assignments_df["claim_code"] != "NONE",
        ["patient_id", "question", "protocol", "trace_id", "claim_code"],
    ].copy()

    # ------------------------------------------------------------------ #
    # Step 1: precompute per-patient co-occurrence contributions          #
    # ------------------------------------------------------------------ #
    patients, cooc_contrib = _precompute_cooc_contributions(df)
    n_patients = len(patients)

    questions = sorted(df["question"].unique().tolist())
    protocols = [gr_label, non_gr_label]

    # ------------------------------------------------------------------ #
    # Step 2: build (n_patients × n_pairs) matrices per cell             #
    # ------------------------------------------------------------------ #
    cell_mats = _build_cell_matrices(
        patients, cooc_contrib, questions, protocols, in_scope
    )

    # ------------------------------------------------------------------ #
    # Point estimates                                                      #
    # ------------------------------------------------------------------ #
    point_si: dict[tuple[str, str], float] = {}
    for q in questions:
        for p in protocols:
            matrix, mask = cell_mats[(q, p)]
            point_si[(q, p)] = _si_from_counts(matrix.sum(axis=0), mask)

    # ------------------------------------------------------------------ #
    # Bootstrap                                                            #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed)

    boot_gr = {q: np.full(n_boot, np.nan) for q in questions}
    boot_non = {q: np.full(n_boot, np.nan) for q in questions}
    boot_delta = {q: np.full(n_boot, np.nan) for q in questions}

    for b in range(n_boot):
        # Resample 40 patient indices with replacement (spec §6).
        drawn = rng.integers(0, n_patients, size=n_patients)

        for q in questions:
            si_g: float
            si_n: float

            matrix_g, mask_g = cell_mats[(q, gr_label)]
            if matrix_g.shape[1] == 0:
                si_g = float("nan")
            else:
                si_g = _si_from_counts(matrix_g[drawn, :].sum(axis=0), mask_g)

            matrix_n, mask_n = cell_mats[(q, non_gr_label)]
            if matrix_n.shape[1] == 0:
                si_n = float("nan")
            else:
                si_n = _si_from_counts(matrix_n[drawn, :].sum(axis=0), mask_n)

            boot_gr[q][b] = si_g
            boot_non[q][b] = si_n
            if not (np.isnan(si_g) or np.isnan(si_n)):
                boot_delta[q][b] = si_g - si_n

    # ------------------------------------------------------------------ #
    # CIs and two-sided bootstrap p-values                                 #
    # ------------------------------------------------------------------ #
    rows: list[dict] = []
    raw_pvals: list[float] = []

    for q in questions:
        si_gr_pt = point_si[(q, gr_label)]
        si_non_pt = point_si[(q, non_gr_label)]
        delta_pt = (
            float("nan")
            if (np.isnan(si_gr_pt) or np.isnan(si_non_pt))
            else si_gr_pt - si_non_pt
        )

        gr_lo, gr_hi = _percentile_ci(boot_gr[q])
        non_lo, non_hi = _percentile_ci(boot_non[q])
        delta_lo, delta_hi = _percentile_ci(boot_delta[q])

        valid_delta = boot_delta[q][~np.isnan(boot_delta[q])]
        if len(valid_delta) > 0:
            p_pos = float(np.mean(valid_delta > 0))
            p_neg = float(np.mean(valid_delta < 0))
            pval = min(2.0 * min(p_pos, p_neg), 1.0)
        else:
            pval = float("nan")

        raw_pvals.append(pval)
        rows.append(
            {
                "question": q,
                f"SI_{gr_label}": si_gr_pt,
                f"SI_{gr_label}_lo": gr_lo,
                f"SI_{gr_label}_hi": gr_hi,
                f"SI_{non_gr_label}": si_non_pt,
                f"SI_{non_gr_label}_lo": non_lo,
                f"SI_{non_gr_label}_hi": non_hi,
                "delta_SI": delta_pt,
                "delta_SI_lo": delta_lo,
                "delta_SI_hi": delta_hi,
                "p_value": pval,
            }
        )

    # ------------------------------------------------------------------ #
    # Benjamini–Hochberg FDR across the 15 questions (spec §6)           #
    # ------------------------------------------------------------------ #
    pvals_arr = np.array(raw_pvals, dtype=float)
    valid_mask = ~np.isnan(pvals_arr)
    p_adj = np.full(len(raw_pvals), float("nan"))
    if valid_mask.any():
        p_adj[valid_mask] = _bh_fdr(pvals_arr[valid_mask])

    result = pd.DataFrame(rows)
    result["p_adj"] = p_adj
    return result

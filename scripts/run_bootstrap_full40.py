"""Run patient-clustered bootstrap for Baek's Specificity Index on the full
40-patient semantic claim-assignment dataset.

Outputs
-------
results/specificity_index_full40_ci.csv   — per-question SI CIs and p-values
(console)                                  — comparison table

Safety gates
------------
The script performs two sanity checks before (or immediately after) running
the bootstrap and stops with a non-zero exit code if either fails:

1. Point-estimate sanity check: bootstrap-derived SI point estimates must
   match the locked values in results/specificity_index_full40.csv to within
   1e-6.  Any mismatch is reported verbatim; the script stops (spec §2).

2. Internal consistency check: SI_GR and SI_Non-GR point estimates from the
   bootstrap DataFrame must equal specificity_table() output to within 1e-9.

Usage
-----
    python scripts/run_bootstrap_full40.py \\
        --claim-assignments /path/to/claim_assignments.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable without installation.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

import specificity_index as si  # noqa: E402
from specificity_index.bootstrap import specificity_bootstrap  # noqa: E402

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run patient-clustered SI bootstrap on a claim-assignment CSV."
    )
    parser.add_argument(
        "--claim-assignments",
        type=Path,
        required=True,
        help=(
            "Long-format CSV with patient_id, question, protocol, trace_id, "
            "and claim_code columns. The raw full40 table is not distributed "
            "with the public repository."
        ),
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=REPO / "results" / "specificity_index_full40.csv",
        help="Locked point-estimate CSV used for the pre-bootstrap sanity check.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "specificity_index_full40_ci.csv",
        help="Output CSV path for bootstrap CIs and p-values.",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=20260701, help="Bootstrap RNG seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
args = _parse_args()
CSV = args.claim_assignments
EXISTING = args.existing
OUT_CI = args.out
if not CSV.exists():
    raise SystemExit(f"claim-assignment CSV not found: {CSV}")
if not EXISTING.exists():
    raise SystemExit(f"locked point-estimate CSV not found: {EXISTING}")

MISMATCH_TOL = 1e-6   # sanity check against existing locked CSV
INTERNAL_TOL = 1e-9   # internal self-consistency check

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
t0 = time.time()
print(f"[{time.time()-t0:.0f}s] Loading CSV …", flush=True)

df_raw = pd.read_csv(
    CSV,
    usecols=["patient_id", "question", "protocol", "trace_id", "claim_code"],
    dtype={
        "patient_id": "category",
        "question": "category",
        "protocol": "category",
        "claim_code": "category",
        # trace_id: leave as string (too many unique values for category)
    },
)
print(
    f"[{time.time()-t0:.1f}s] Loaded {len(df_raw):,} rows; "
    f"{df_raw['patient_id'].nunique()} patients.",
    flush=True,
)

# Convert categorical columns to str for downstream compatibility.
for col in ["patient_id", "question", "protocol", "claim_code"]:
    df_raw[col] = df_raw[col].astype(str)

df = df_raw[df_raw["claim_code"] != "NONE"].copy()
print(
    f"[{time.time()-t0:.1f}s] After NONE filter: {len(df):,} rows; "
    f"unique traces: {df['trace_id'].nunique():,}.",
    flush=True,
)

# ---------------------------------------------------------------------------
# SAFETY GATE 1: verify against existing locked results before bootstrap
# ---------------------------------------------------------------------------
print(f"[{time.time()-t0:.1f}s] Verifying point estimates via specificity_table()…", flush=True)
tbl_locked = si.specificity_table(df)

existing = pd.read_csv(EXISTING)

mismatches: list[str] = []
for _, ex_row in existing.iterrows():
    q = ex_row["question"]
    tbl_row = tbl_locked[tbl_locked["question"] == q]
    if tbl_row.empty:
        mismatches.append(f"  {q}: question not found in recomputed table")
        continue
    tbl_row = tbl_row.iloc[0]
    for col_ex, col_tbl in [
        ("SI_GR", "SI_GR"),
        ("SI_Non-GR", "SI_Non-GR"),
    ]:
        got = float(tbl_row[col_tbl])
        exp = float(ex_row[col_ex])
        if np.isnan(got) and np.isnan(exp):
            continue
        diff = abs(got - exp)
        if diff > MISMATCH_TOL:
            mismatches.append(
                f"  {q}/{col_ex}: recomputed={got:.10f}, "
                f"locked={exp:.10f}, diff={diff:.2e}"
            )

if mismatches:
    print("\nFATAL: point-estimate mismatch vs locked results/specificity_index_full40.csv")
    for m in mismatches:
        print(m)
    print("\nSTOP: correct the implementation before running the bootstrap.")
    sys.exit(1)

print(
    f"[{time.time()-t0:.1f}s] Point estimates match locked CSV to within {MISMATCH_TOL:.0e}. OK.",
    flush=True,
)

# ---------------------------------------------------------------------------
# Run bootstrap
# ---------------------------------------------------------------------------
print(
    f"[{time.time()-t0:.1f}s] Starting bootstrap (n_boot={args.n_boot}, seed={args.seed}) …",
    flush=True,
)

result = specificity_bootstrap(
    df,
    n_boot=args.n_boot,
    seed=args.seed,
    gr_label="GR",
    non_gr_label="Non-GR",
)
print(f"[{time.time()-t0:.1f}s] Bootstrap complete.", flush=True)

# ---------------------------------------------------------------------------
# SAFETY GATE 2: bootstrap point estimates must match specificity_table()
# ---------------------------------------------------------------------------
internal_mismatches: list[str] = []
for _, tbl_row in tbl_locked.iterrows():
    q = tbl_row["question"]
    boot_row = result[result["question"] == q]
    if boot_row.empty:
        internal_mismatches.append(f"  {q}: not in bootstrap output")
        continue
    boot_row = boot_row.iloc[0]
    for col_tbl, col_boot in [
        ("SI_GR", "SI_GR"),
        ("SI_Non-GR", "SI_Non-GR"),
    ]:
        got = float(boot_row[col_boot])
        exp = float(tbl_row[col_tbl])
        if np.isnan(got) and np.isnan(exp):
            continue
        diff = abs(got - exp)
        if diff > INTERNAL_TOL:
            internal_mismatches.append(
                f"  {q}/{col_tbl}: bootstrap={got:.12f}, "
                f"specificity_table={exp:.12f}, diff={diff:.2e}"
            )

if internal_mismatches:
    print("\nFATAL: bootstrap point estimates disagree with specificity_table():")
    for m in internal_mismatches:
        print(m)
    print("\nSTOP: there is a bug in bootstrap.py point-estimate computation.")
    sys.exit(1)

print(
    f"[{time.time()-t0:.1f}s] Internal consistency check passed (tol {INTERNAL_TOL:.0e}).",
    flush=True,
)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
OUT_CI.parent.mkdir(exist_ok=True)
result.to_csv(OUT_CI, index=False)
print(f"[{time.time()-t0:.1f}s] Saved → {OUT_CI}", flush=True)

# ---------------------------------------------------------------------------
# Print per-question table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("Per-question Specificity Index: 95% CIs and FDR-adjusted p-values")
print(f"(spec §6: patient-clustered bootstrap, n_boot={args.n_boot}, seed={args.seed})")
print("=" * 90)

header = (
    f"{'Question':<8}  "
    f"{'SI_GR':>7} {'[95% CI]':^19}  "
    f"{'SI_Non-GR':>9} {'[95% CI]':^19}  "
    f"{'ΔSI':>7} {'[95% CI]':^19}  "
    f"{'p':>7}  {'p_BH':>7}"
)
print(header)
print("-" * 90)

for _, row in result.iterrows():
    q = row["question"]

    def _fmt_ci(pt_col: str, lo_col: str, hi_col: str) -> str:
        pt = row[pt_col]
        lo = row[lo_col]
        hi = row[hi_col]
        if np.isnan(pt):
            return f"{'NaN':>7} {'[         NaN         ]':>21}"
        return (
            f"{pt:7.4f} [{lo:7.4f}, {hi:7.4f}]"
        )

    si_gr_str  = _fmt_ci("SI_GR",       "SI_GR_lo",       "SI_GR_hi")
    si_non_str = _fmt_ci("SI_Non-GR",   "SI_Non-GR_lo",   "SI_Non-GR_hi")
    delta_str  = _fmt_ci("delta_SI",    "delta_SI_lo",    "delta_SI_hi")

    pval = row["p_value"]
    padj = row["p_adj"]
    pval_str = f"{pval:7.4f}" if not np.isnan(pval) else "    NaN"
    padj_str = f"{padj:7.4f}" if not np.isnan(padj) else "    NaN"

    print(f"{q:<8}  {si_gr_str}  {si_non_str}  {delta_str}  {pval_str}  {padj_str}")

print("=" * 90)
print(
    f"\n[{time.time()-t0:.1f}s] Done. "
    f"Questions with SI_GR > SI_Non-GR: "
    f"{int((result['delta_SI'] > 0).sum())}/{len(result)}."
)
print(f"Results written to: {OUT_CI}")

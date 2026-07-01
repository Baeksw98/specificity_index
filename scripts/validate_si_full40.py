"""Validate Baek's Specificity Index on the full 40-patient semantic data.

Loads the complete claim-assignment table, computes SI / AC per (question,
protocol) with the locked library, and writes the per-question table plus a
top-Anchoring-Credit table. This is the gate-B empirical validation and the
source of Paper 1's results. Deterministic; no fabricated values.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

REPO = Path("/NHNHOME/WORKSPACE/0226010051_A/[Self Research-#2] Specificity_index")
sys.path.insert(0, str(REPO))
import pandas as pd  # noqa: E402
import specificity_index as si  # noqa: E402
from specificity_index.scope import IN_SCOPE  # noqa: E402

CSV = Path("/NHNHOME/WORKSPACE/0226010051_A/data/reports/protocol_study/qualitative_analysis/question_deep_dive_full40_semantic/claim_assignments.csv")
OUT = REPO / "results"
OUT.mkdir(exist_ok=True)

t0 = time.time()
df = pd.read_csv(
    CSV,
    usecols=["patient_id", "question", "protocol", "trace_id", "claim_code"],
    dtype={"patient_id": "category", "question": "category", "protocol": "category",
           "trace_id": "string", "claim_code": "category"},
)
print(f"[{time.time()-t0:.0f}s] loaded {len(df):,} rows", flush=True)
df = df[df["claim_code"] != "NONE"].copy()
df["question"] = df["question"].astype(str)
df["protocol"] = df["protocol"].astype(str)
df["claim_code"] = df["claim_code"].astype(str)
print(f"[{time.time()-t0:.0f}s] after NONE filter: {len(df):,} rows; traces {df['trace_id'].nunique():,}", flush=True)

tbl = si.specificity_table(df)
tbl.to_csv(OUT / "specificity_index_full40.csv", index=False)
print(f"[{time.time()-t0:.0f}s] specificity_table written", flush=True)

# Anchoring Credit: top category per (question, GR) — confirm the in-scope category dominates
ac_rows = []
for (q, p), cell in df.groupby(["question", "protocol"]):
    ac = si.anchoring_credit(cell, q)
    scope = sorted(IN_SCOPE.get(q, set()))
    top = sorted(ac.items(), key=lambda kv: -kv[1])[:3]
    ac_rows.append({
        "question": q, "protocol": p, "in_scope": ";".join(scope),
        "top_AC": "; ".join(f"{c}={v:.3f}" for c, v in top),
        "in_scope_AC_share": round(sum(v for c, v in ac.items() if c in IN_SCOPE.get(q, set())), 4),
    })
ac_df = pd.DataFrame(ac_rows).sort_values(["question", "protocol"])
ac_df.to_csv(OUT / "anchoring_credit_full40.csv", index=False)

pos = int((tbl["delta_SI"] > 0).sum())
summary = {
    "n_assignment_rows": int(len(df)),
    "n_traces": int(df["trace_id"].nunique()),
    "questions": int(tbl["question"].nunique()),
    "si_gr_gt_nongr_count": pos,
    "si_gr_gt_nongr_total": int(len(tbl)),
    "mean_delta_SI": round(float(tbl["delta_SI"].mean()), 4),
    "mean_SI_GR": round(float(tbl["SI_GR"].mean()), 4),
    "mean_SI_NonGR": round(float(tbl["SI_Non-GR"].mean()), 4),
    "elapsed_sec": round(time.time() - t0, 1),
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("=== SI per question ===", flush=True)
print(tbl[["question", "SI_GR", "SI_Non-GR", "delta_SI"]].to_string(index=False), flush=True)
print("=== SUMMARY ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)

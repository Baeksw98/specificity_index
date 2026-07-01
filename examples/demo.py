"""Self-contained demo of Baek's Specificity Index.

Runs without the (large) study data: it builds a small synthetic claim-assignment
table where, for one question, the "GR" condition concentrates its reasoning on
the question's in-scope category while the "Non-GR" condition spreads across
off-question categories. It then shows the three core calls and the efficiency
identity Σ_c psi_c = SI.

Run:  python examples/demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import specificity_index as si  # noqa: E402

# Q3.1 in-scope set is {C1} (see specificity_index.scope.IN_SCOPE).
rows = []
for i in range(10):
    p = f"P{i:02d}"
    # GR: each trace pairs the in-scope category C1 with one neighbor -> anchored
    for t, nb in enumerate(["C17", "C2", "C4", "C27"]):
        rows += [(p, "Q3.1", "GR", f"{p}-g-{t}", c) for c in ("C1", nb)]
    # Non-GR: traces pair off-scope categories, rarely touching C1 -> diffuse
    for t, (a, b) in enumerate([("C3", "C17"), ("C20", "C25"), ("C5", "C22"), ("C9", "C13")]):
        rows += [(p, "Q3.1", "Non-GR", f"{p}-n-{t}", c) for c in (a, b)]

df = pd.DataFrame(rows, columns=["patient_id", "question", "protocol", "trace_id", "claim_code"])

gr = df[df.protocol == "GR"]
ng = df[df.protocol == "Non-GR"]

print("Specificity Index (in-scope connective concentration), Q3.1:")
print(f"  GR     SI = {si.specificity_index(gr, 'Q3.1'):.3f}")
print(f"  Non-GR SI = {si.specificity_index(ng, 'Q3.1'):.3f}")

ac = si.anchoring_credit(gr, "Q3.1")
print("\nAnchoring Credit psi_c (top categories, GR):")
for c, v in sorted(ac.items(), key=lambda kv: -kv[1])[:4]:
    print(f"  {c}: {v:.3f}")
print(f"  sum psi_c = {sum(ac.values()):.6f}  ==  SI = {si.specificity_index(gr, 'Q3.1'):.6f}  (efficiency)")

print("\nPer-question table (point estimates):")
print(si.specificity_table(df)[["question", "SI_GR", "SI_Non-GR", "delta_SI"]].to_string(index=False))

print("\nWith 95% CIs and BH-adjusted p (patient-clustered bootstrap, small n_boot for the demo):")
print(si.specificity_bootstrap(df, n_boot=200, seed=20260701).to_string(index=False))

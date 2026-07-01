"""Question-aligned in-scope category map for the Specificity Index.

This module defines which clinical-content categories (C1–C28) are considered
"in scope" for each evaluation question. Two layers are distinguished:

- ``DEFAULT_MAIN``: the primary category or categories that the question
  directly targets (e.g., Q3.1 is about C1 — Symptom Recognition).
- ``DEFAULT_CROSS_CUTTING``: categories that are in scope for a question
  because they apply across multiple clinical domains (e.g., C27 spans
  several questions).

``IN_SCOPE[q]`` is the union of both layers for question ``q``, and is the
set used when computing tie-strength concentration and weighted-degree
centrality for that question.

IMPORTANT — CLINICIAN VALIDATION REQUIRED
------------------------------------------
This mapping MUST be reviewed and signed off by a qualified clinician before
the index values are interpreted or reported. The category-to-question
alignment reflects the research team's current understanding of the Gold
Rubric; it is not a finalized clinical specification. Changes to the mapping
directly alter all computed index values.

Source: ported from
``Qualitative_Assessment/Meeting_070126/02_specificity_index/specificity_index.py``
with no mathematical modification.
"""

from __future__ import annotations

DEFAULT_MAIN: dict[str, frozenset[str]] = {
    "Q3.1": frozenset({"C1"}),
    "Q3.2": frozenset({"C2"}),
    "Q3.3": frozenset({"C3"}),
    "Q3.4": frozenset({"C4"}),
    "Q3.5": frozenset({"C5", "C6", "C7"}),
    "Q3.6": frozenset({"C8", "C9", "C10"}),
    "Q3.7": frozenset({"C11", "C12"}),
    "Q3.8": frozenset({"C13", "C14", "C15"}),
    "Q4.1": frozenset({"C16"}),
    "Q4.2": frozenset({"C17", "C18"}),
    "Q4.3": frozenset({"C19"}),
    "Q5.1": frozenset({"C20", "C21", "C26"}),
    "Q5.2": frozenset({"C22"}),
    "Q5.3": frozenset({"C23", "C24"}),
    "Q5.4": frozenset({"C25", "C26"}),
}
"""Primary in-scope category set for each evaluation question.

Keys are pipeline question identifiers (e.g., "Q3.1"). Values are frozensets
of category codes (e.g., "C1"). A reasoning trace that cites only categories
in this set is maximally anchored on the question's intended clinical content.
"""

DEFAULT_CROSS_CUTTING: dict[str, frozenset[str]] = {
    "Q3.3": frozenset({"C27"}),
    "Q3.4": frozenset({"C27"}),
    "Q3.5": frozenset({"C27"}),
    "Q4.1": frozenset({"C27"}),
    "Q4.3": frozenset({"C27"}),
    "Q5.1": frozenset({"C27"}),
    "Q5.4": frozenset({"C27"}),
}
"""Cross-cutting categories that are additionally in scope for specific questions.

C27 is a cross-cutting category relevant to multiple clinical domains. For
the questions listed here it is counted as in-scope in addition to the
question's primary categories. For all other questions C27 is out-of-scope.
"""

IN_SCOPE: dict[str, frozenset[str]] = {
    q: DEFAULT_MAIN[q] | DEFAULT_CROSS_CUTTING.get(q, frozenset())
    for q in DEFAULT_MAIN
}
"""Merged in-scope set per question: union of main and cross-cutting categories.

This is the operationalized ``S(q)`` used in all index computations. See the
design consideration document (docs/Baek_Specificity_Index_Design_Consideration.md)
Section 3 for the formal definition.
"""

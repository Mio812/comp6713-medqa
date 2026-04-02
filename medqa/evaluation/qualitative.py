"""
Qualitative error analysis for the MedQA system.

Examines misclassified / low-scoring examples and groups them into
interpretable error categories. Results are written to a JSON file for
inclusion in the project report.

Error taxonomy (medical QA specific):
  - WRONG_RETRIEVAL    : correct answer not in retrieved context
  - TERMINOLOGY_MISMATCH: question uses informal terms; model uses formal ones (or vice versa)
  - PARTIAL_ANSWER     : answer partially correct but incomplete
  - HALLUCINATION      : model generates plausible-sounding but incorrect information
  - AMBIGUOUS_QUESTION : question is under-specified or has multiple valid answers
  - OTHER              : does not fit the above categories
"""

import json
from pathlib import Path
from typing import Any

from medqa.data.preprocessor import normalise_answer
from medqa.evaluation.metrics import token_f1


# ── Error categorisation ──────────────────────────────────────────────────────

def categorise_error(
    question: str,
    prediction: str,
    gold: str,
    context: str,
    f1_score: float,
) -> str:
    """
    Heuristically assign an error category to a misclassified example.

    The rules below are intentionally simple keyword/overlap checks.
    They provide a structured starting point for the qualitative section
    of the report; manual review of the output is expected.
    """
    pred_norm = normalise_answer(prediction)
    gold_norm = normalise_answer(gold)
    ctx_norm  = normalise_answer(context)

    # Gold answer not present at all in retrieved context
    gold_words = set(gold_norm.split())
    ctx_words  = set(ctx_norm.split())
    overlap    = gold_words & ctx_words
    if len(overlap) / max(len(gold_words), 1) < 0.2:
        return "WRONG_RETRIEVAL"

    # Partial answer: F1 > 0 but < 0.5, and some overlap exists
    if 0 < f1_score < 0.5 and overlap:
        return "PARTIAL_ANSWER"

    # Hallucination proxy: prediction is long but shares < 10 % tokens with gold
    pred_words = set(pred_norm.split())
    if len(pred_words) > 10 and len(pred_words & gold_words) / len(pred_words) < 0.1:
        return "HALLUCINATION"

    # Terminology mismatch: short prediction, zero overlap with gold
    if f1_score == 0.0 and len(pred_norm.split()) <= 5:
        return "TERMINOLOGY_MISMATCH"

    return "OTHER"


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_errors(
    results: list[dict[str, Any]],
    f1_threshold: float = 0.5,
    max_examples: int = 20,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Identify low-scoring examples, categorise errors, and return a summary.

    Args:
        results       : list of dicts, each with keys:
                          question, predicted_answer, gold_answer,
                          context, token_f1 (float)
        f1_threshold  : examples with token_f1 below this are treated as errors
        max_examples  : cap on how many error examples to store (for report size)
        output_path   : if given, write JSON report to this file

    Returns:
        {
            "total":          int,
            "errors":         int,
            "error_rate":     float,
            "category_counts": {category: count, ...},
            "examples":       [list of error dicts],
        }
    """
    errors = []
    for r in results:
        f1 = r.get("token_f1", token_f1(r["predicted_answer"], r["gold_answer"]))
        if f1 >= f1_threshold:
            continue   # correct enough — skip

        category = categorise_error(
            question   = r.get("question", ""),
            prediction = r.get("predicted_answer", ""),
            gold       = r.get("gold_answer", ""),
            context    = r.get("context", ""),
            f1_score   = f1,
        )
        errors.append({
            "question":          r.get("question", ""),
            "predicted_answer":  r.get("predicted_answer", ""),
            "gold_answer":       r.get("gold_answer", ""),
            "token_f1":          round(f1, 4),
            "error_category":    category,
        })

    # Count categories
    category_counts: dict[str, int] = {}
    for e in errors:
        cat = e["error_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    summary = {
        "total":           len(results),
        "errors":          len(errors),
        "error_rate":      round(len(errors) / max(len(results), 1), 4),
        "category_counts": category_counts,
        "examples":        errors[:max_examples],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Qualitative] Error report saved to {output_path}")

    _print_summary(summary)
    return summary


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{'='*50}")
    print("  Qualitative Error Analysis")
    print(f"{'='*50}")
    print(f"  Total samples : {summary['total']}")
    print(f"  Errors        : {summary['errors']} ({summary['error_rate']*100:.1f}%)")
    print(f"  Error types:")
    for cat, cnt in sorted(summary["category_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / max(summary["errors"], 1) * 100
        print(f"    {cat:<25} {cnt:>4}  ({pct:.1f}%)")
    print(f"{'='*50}\n")

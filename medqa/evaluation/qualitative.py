"""
Qualitative error analysis: categorises low-scoring predictions into error types
(WRONG_RETRIEVAL, TERMINOLOGY_MISMATCH, PARTIAL_ANSWER, HALLUCINATION, etc.).
"""

import json
from pathlib import Path
from typing import Any

from medqa.data.preprocessor import normalise_answer
from medqa.evaluation.metrics import token_f1


def categorise_error(
    question: str,
    prediction: str,
    gold: str,
    context: str,
    f1_score: float,
) -> str:
    """Assign a heuristic error category to a misclassified example."""
    pred_norm = normalise_answer(prediction)
    gold_norm = normalise_answer(gold)
    ctx_norm  = normalise_answer(context)

    gold_words = set(gold_norm.split())
    ctx_words  = set(ctx_norm.split())
    overlap    = gold_words & ctx_words
    if len(overlap) / max(len(gold_words), 1) < 0.2:
        return "WRONG_RETRIEVAL"

    if 0 < f1_score < 0.5 and overlap:
        return "PARTIAL_ANSWER"

    pred_words = set(pred_norm.split())
    if len(pred_words) > 10 and len(pred_words & gold_words) / len(pred_words) < 0.1:
        return "HALLUCINATION"

    if f1_score == 0.0 and len(pred_norm.split()) <= 5:
        return "TERMINOLOGY_MISMATCH"

    return "OTHER"


def analyse_errors(
    results: list[dict[str, Any]],
    f1_threshold: float = 0.5,
    max_examples: int = 20,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Identify low-scoring examples, categorise errors, and return a summary."""
    errors = []
    for r in results:
        f1 = r.get("token_f1", token_f1(r["predicted_answer"], r["gold_answer"]))
        if f1 >= f1_threshold:
            continue

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

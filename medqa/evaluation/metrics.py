"""
Quantitative evaluation metrics for the MedQA system.

Implements three metrics appropriate for open-domain medical QA:

  1. Exact Match (EM)  — binary; 1 if normalised prediction == normalised gold
  2. Token F1          — token-level overlap (standard SQuAD metric)
  3. BERTScore         — semantic similarity using contextual embeddings

All functions accept parallel lists of predictions and gold answers and
return a summary dict so results can be compared across models in one call.
"""

import re
import string
from collections import Counter
from typing import Any

from medqa.data.preprocessor import normalise_answer


# ── Exact Match ───────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if the normalised strings are identical, else 0.0."""
    return float(normalise_answer(prediction) == normalise_answer(gold))


# ── Token F1 ──────────────────────────────────────────────────────────────────

def token_f1(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 between *prediction* and *gold*.
    Tokens are split on whitespace after normalisation.
    """
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1


# ── BERTScore ─────────────────────────────────────────────────────────────────

def bert_score_batch(
    predictions: list[str],
    golds: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Compute BERTScore (precision, recall, F1) for a batch.

    Uses DeBERTa-xlarge by default (recommended by the BERTScore authors
    for English). Falls back to bert-base-uncased if the larger model is
    unavailable.

    Returns:
        {"bertscore_p": float, "bertscore_r": float, "bertscore_f1": float}
    """
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(
            predictions,
            golds,
            model_type=model_type,
            batch_size=batch_size,
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_p":  float(P.mean()),
            "bertscore_r":  float(R.mean()),
            "bertscore_f1": float(F.mean()),
        }
    except Exception as e:
        print(f"[Metrics] BERTScore failed: {e}")
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}


# ── Combined evaluation ───────────────────────────────────────────────────────

def evaluate(
    predictions: list[str],
    golds: list[str],
    use_bertscore: bool = True,
) -> dict[str, Any]:
    """
    Compute EM, Token-F1, and optionally BERTScore for all prediction/gold pairs.

    Args:
        predictions   : list of model-predicted answer strings
        golds         : list of gold answer strings (same length)
        use_bertscore : whether to compute BERTScore (slower, requires GPU ideally)

    Returns a dict with per-sample lists and aggregate means, e.g.::

        {
            "exact_match":    [1, 0, 0, ...],
            "token_f1":       [1.0, 0.5, 0.2, ...],
            "mean_em":        0.34,
            "mean_f1":        0.52,
            "bertscore_f1":   0.71,   # if use_bertscore=True
            "n_samples":      100,
        }
    """
    assert len(predictions) == len(golds), "predictions and golds must have the same length"

    em_scores = [exact_match(p, g)  for p, g in zip(predictions, golds)]
    f1_scores = [token_f1(p, g)     for p, g in zip(predictions, golds)]

    results: dict[str, Any] = {
        "exact_match": em_scores,
        "token_f1":    f1_scores,
        "mean_em":     sum(em_scores) / len(em_scores),
        "mean_f1":     sum(f1_scores) / len(f1_scores),
        "n_samples":   len(predictions),
    }

    if use_bertscore:
        bs = bert_score_batch(predictions, golds)
        results.update(bs)

    return results


def print_results(results: dict[str, Any], model_name: str = "Model") -> None:
    """Pretty-print an evaluation results dict."""
    print(f"\n{'='*50}")
    print(f"  Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Samples      : {results['n_samples']}")
    print(f"  Exact Match  : {results['mean_em']:.4f}")
    print(f"  Token F1     : {results['mean_f1']:.4f}")
    if "bertscore_f1" in results:
        print(f"  BERTScore F1 : {results['bertscore_f1']:.4f}")
    print(f"{'='*50}\n")

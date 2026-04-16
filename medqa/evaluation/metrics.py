"""Evaluation metrics for the MedQA system.

Provides Exact Match, Token F1, BERTScore, ROUGE-L, Yes/No accuracy, and a
per-source breakdown.
"""

from collections import Counter, defaultdict
from typing import Any, Iterable

from medqa.data.preprocessor import normalise_answer, extract_yesno


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if the normalised strings are identical, else 0.0."""
    return float(normalise_answer(prediction) == normalise_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold after normalisation."""
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
    return 2 * precision * recall / (precision + recall)


def yesno_accuracy(predictions, golds):
    """Accuracy on the subset where the gold label is yes / no / maybe.

    Uses extract_yesno() so fluent outputs like 'Yes, because...' count
    correctly. Returns dict with n, correct, accuracy, and uncategorised.
    """
    n, correct, uncategorised = 0, 0, 0
    for pred, gold in zip(predictions, golds):
        gold_label = extract_yesno(gold)
        if gold_label is None:
            continue
        n += 1
        pred_label = extract_yesno(pred)
        if pred_label is None:
            uncategorised += 1
            continue
        if pred_label == gold_label:
            correct += 1
    return {
        "n":             n,
        "correct":       correct,
        "accuracy":      correct / n if n else 0.0,
        "uncategorised": uncategorised,
    }


def bert_score_batch(predictions, golds, model_type="roberta-base", batch_size=32):
    """Compute BERTScore (precision, recall, F1) for a batch.

    Returns None per field if the library is missing or the call fails.
    """
    try:
        from bert_score import score as bs_score
    except ImportError:
        print("[Metrics] bert_score not installed; BERTScore unavailable.")
        return {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}

    try:
        P, R, F = bs_score(
            predictions,
            golds,
            model_type=model_type,
            batch_size=batch_size,
            lang="en",
            verbose=False,
        )
    except Exception as e:
        print(f"[Metrics] BERTScore computation failed: {e}")
        return {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}

    return {
        "bertscore_p":  float(P.mean()),
        "bertscore_r":  float(R.mean()),
        "bertscore_f1": float(F.mean()),
    }


def rouge_l_mean(predictions, golds):
    """Mean ROUGE-L F1 across the batch. Returns 0.0 if rouge_score unavailable."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[Metrics] rouge_score not installed; skipping ROUGE-L.")
        return 0.0
    if not predictions:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(target=g, prediction=p)["rougeL"].fmeasure
        for p, g in zip(predictions, golds)
    ]
    return sum(scores) / len(scores)


def _chunk_contains_answer(chunk_text: str, gold_answer: str, min_overlap: float = 0.5) -> bool:
    """Does chunk_text contain gold_answer? Uses substring + token-overlap fallback."""
    if not chunk_text or not gold_answer:
        return False
    gold_norm = normalise_answer(gold_answer)
    chunk_norm = normalise_answer(chunk_text)
    if not gold_norm or not chunk_norm:
        return False
    if gold_norm in chunk_norm:
        return True
    gold_toks = set(gold_norm.split())
    chunk_toks = set(chunk_norm.split())
    if not gold_toks:
        return False
    overlap = len(gold_toks & chunk_toks) / len(gold_toks)
    return overlap >= min_overlap


def retrieval_metrics(retrieved_per_query, gold_answers, ks=(1, 3, 5, 10)) -> dict:
    """Recall@k and MRR for a batch of retrieval results.

    Args:
        retrieved_per_query : one entry per question; each entry is a list of
            retrieved chunks ordered by rank (str or dict with 'text' key).
        gold_answers : gold answer strings (same length).
        ks : which k values to report Recall@k at.

    A retrieval counts as a hit if any top-k chunk contains the gold answer.
    """
    assert len(retrieved_per_query) == len(gold_answers)
    n = len(gold_answers)
    if n == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "n": 0}

    def _text(chunk):
        return chunk["text"] if isinstance(chunk, dict) else str(chunk)

    recall_at_k = {k: 0 for k in ks}
    rr_sum = 0.0
    max_k = max(ks)

    for chunks, gold in zip(retrieved_per_query, gold_answers):
        first_hit_rank = None
        for rank, chunk in enumerate(chunks[:max_k], start=1):
            if _chunk_contains_answer(_text(chunk), gold):
                first_hit_rank = rank
                break
        if first_hit_rank is not None:
            rr_sum += 1.0 / first_hit_rank
            for k in ks:
                if first_hit_rank <= k:
                    recall_at_k[k] += 1

    out = {f"recall@{k}": recall_at_k[k] / n for k in ks}
    out["mrr"] = rr_sum / n
    out["n"] = n
    return out


def llm_as_judge(predictions, golds, questions, llm):
    """Ask an LLM whether each prediction is semantically correct.

    Compensates for EM/F1 under-scoring paraphrased-but-correct answers.
    A prediction counts as correct if the judge's reply contains 'correct'
    (or starts with 'yes') and does not contain 'incorrect'.
    """
    judge_prompt = (
        "Decide whether the candidate answer is semantically equivalent to "
        "the reference answer for the given question. "
        "Reply with exactly one word: CORRECT or INCORRECT."
    )
    correct = 0
    evaluated = 0
    for q, p, g in zip(questions, predictions, golds):
        context = (
            f"Question: {q}\n"
            f"Reference answer: {g}\n"
            f"Candidate answer: {p}\n"
        )
        try:
            verdict = llm.predict(judge_prompt, context).get("predicted_answer", "")
        except Exception as e:
            print(f"[LLM-as-judge] failed on one example: {e}")
            continue
        evaluated += 1
        v = verdict.strip().lower()
        if "incorrect" in v:
            continue
        if "correct" in v or v.startswith("yes"):
            correct += 1
    return {
        "n":        evaluated,
        "correct":  correct,
        "accuracy": correct / evaluated if evaluated else 0.0,
    }


def evaluate(predictions, golds, sources=None, use_bertscore=True, use_rouge=True):
    """Compute all metrics, optionally broken down by sources."""
    assert len(predictions) == len(golds), \
        "predictions and golds must have the same length"

    em_scores = [exact_match(p, g) for p, g in zip(predictions, golds)]
    f1_scores = [token_f1(p, g)    for p, g in zip(predictions, golds)]

    results: dict = {
        "exact_match": em_scores,
        "token_f1":    f1_scores,
        "mean_em":     sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "mean_f1":     sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "n_samples":   len(predictions),
    }

    results["yesno"] = yesno_accuracy(predictions, golds)

    if use_bertscore:
        results.update(bert_score_batch(predictions, golds))

    if use_rouge:
        results["rouge_l"] = rouge_l_mean(predictions, golds)

    if sources is not None:
        by_source: dict = defaultdict(lambda: {"em": [], "f1": []})
        for src, em, f1 in zip(sources, em_scores, f1_scores):
            by_source[src]["em"].append(em)
            by_source[src]["f1"].append(f1)

        results["per_source"] = {
            src: {
                "n":       len(vals["em"]),
                "mean_em": sum(vals["em"]) / len(vals["em"]) if vals["em"] else 0.0,
                "mean_f1": sum(vals["f1"]) / len(vals["f1"]) if vals["f1"] else 0.0,
            }
            for src, vals in by_source.items()
        }

    return results


def print_results(results: dict, model_name: str = "Model") -> None:
    """Pretty-print an evaluation results dict."""
    print(f"\n{'=' * 60}")
    print(f"  Results: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Samples      : {results['n_samples']}")
    print(f"  Exact Match  : {results['mean_em']:.4f}")
    print(f"  Token F1     : {results['mean_f1']:.4f}")
    bs = results.get("bertscore_f1")
    if bs is None and "bertscore_f1" in results:
        print(f"  BERTScore F1 : n/a (not computed)")
    elif bs is not None:
        print(f"  BERTScore F1 : {bs:.4f}")
    if "rouge_l" in results:
        print(f"  ROUGE-L      : {results['rouge_l']:.4f}")

    yn = results.get("yesno", {})
    if yn and yn.get("n"):
        print(f"  Yes/No acc   : {yn['accuracy']:.4f}  "
              f"({yn['correct']}/{yn['n']}, {yn['uncategorised']} uncategorised)")

    if "per_source" in results:
        print("  Per source:")
        for src, stats in results["per_source"].items():
            print(f"    {src:<20} n={stats['n']:>4}  "
                  f"EM={stats['mean_em']:.4f}  F1={stats['mean_f1']:.4f}")
    print(f"{'=' * 60}\n")

"""Unit tests for medqa.evaluation.metrics."""

import pytest

from medqa.evaluation.metrics import (
    bert_score_batch,
    evaluate,
    exact_match,
    token_f1,
    yesno_accuracy,
)


class TestExactMatch:
    @pytest.mark.parametrize("p, g, expected", [
        ("yes", "yes", 1.0),
        ("Yes.", "yes", 1.0),
        ("The answer is yes", "yes", 1.0),
        ("yes", "no", 0.0),
        ("the cat", "cat", 1.0),
    ])
    def test_cases(self, p, g, expected):
        assert exact_match(p, g) == expected


class TestTokenF1:
    def test_identical(self):
        assert token_f1("type 2 diabetes", "type 2 diabetes") == 1.0

    def test_partial_overlap(self):
        assert token_f1("type diabetes", "type 2 diabetes") == pytest.approx(0.8, abs=1e-4)

    def test_no_overlap(self):
        assert token_f1("aspirin", "insulin") == 0.0

    def test_empty(self):
        assert token_f1("", "yes") == 0.0
        assert token_f1("yes", "") == 0.0


class TestYesNoAccuracy:
    def test_counts_only_yesno_golds(self):
        preds = ["yes", "the answer is cat", "no"]
        golds = ["yes", "maybe",              "no"]
        res = yesno_accuracy(preds, golds)
        assert res["n"] == 3
        assert res["correct"] == 2
        assert res["uncategorised"] == 1
        assert res["accuracy"] == pytest.approx(2 / 3)

    def test_skips_non_yesno_golds(self):
        preds = ["aspirin", "yes"]
        golds = ["insulin", "yes"]
        res = yesno_accuracy(preds, golds)
        assert res["n"] == 1
        assert res["correct"] == 1


def test_bertscore_returns_none_when_unavailable(monkeypatch):
    """If bert_score isn't installed, return None rather than 0.0."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "bert_score":
            raise ImportError("simulated missing dep")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = bert_score_batch(["a"], ["a"])
    assert res == {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}


def test_evaluate_with_sources_breakdown():
    preds = ["yes", "no",  "aspirin"]
    golds = ["yes", "yes", "aspirin"]
    sources = ["pubmedqa", "pubmedqa", "bioasq"]
    res = evaluate(preds, golds, sources=sources, use_bertscore=False, use_rouge=False)

    assert res["n_samples"] == 3
    assert res["mean_em"] == pytest.approx(2 / 3)
    assert res["per_source"]["pubmedqa"]["n"] == 2
    assert res["per_source"]["bioasq"]["mean_em"] == 1.0

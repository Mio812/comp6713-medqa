"""Unit tests for medqa.data.preprocessor."""

import pytest

from medqa.data.preprocessor import (
    clean_text,
    clean_record,
    extract_yesno,
    normalise_answer,
    split_dataset,
)


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("a  b\n\tc") == "a b c"

    def test_strips_control_chars(self):
        assert clean_text("foo\x00bar\x1fbaz") == "foo bar baz"

    def test_preserves_unicode(self):
        assert clean_text("α-helix at 37°C, p ≤ 0.05") == "α-helix at 37°C, p ≤ 0.05"

    def test_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""


def test_clean_record_only_touches_strings():
    r = clean_record({"q": "a\n b", "n": 3, "ctx": None})
    assert r == {"q": "a b", "n": 3, "ctx": None}


class TestNormaliseAnswer:
    @pytest.mark.parametrize("raw, expected", [
        ("Yes.", "yes"),
        ("The answer is: yes", "yes"),
        ("A: Yes", "yes"),
        ("  the   cat  ", "cat"),
        ("an apple, a day", "apple day"),
        (None, ""),
    ])
    def test_cases(self, raw, expected):
        assert normalise_answer(raw) == expected


class TestExtractYesNo:
    def test_first_token_wins(self):
        assert extract_yesno("Yes, because...") == "yes"
        assert extract_yesno("No - the study shows otherwise") == "no"
        assert extract_yesno("Maybe. Further research needed.") == "maybe"

    def test_deterministic_order_on_fallback(self):
        assert extract_yesno("I think no, not yes") == "no"

    def test_prefix_stripped(self):
        assert extract_yesno("The answer is no") == "no"

    def test_unknown_returns_none(self):
        assert extract_yesno("The treatment is aspirin") is None
        assert extract_yesno("") is None
        assert extract_yesno(None) is None


class TestSplit:
    def _records(self, n, source):
        return [{"question": f"q{i}", "answer": "a", "source": source} for i in range(n)]

    def test_stratified_split_balances_sources(self):
        records = self._records(50, "pubmedqa") + self._records(50, "bioasq")
        train, test = split_dataset(records, test_size=0.2, random_state=0)

        def src_counts(rs):
            return {"pubmedqa": sum(r["source"] == "pubmedqa" for r in rs),
                    "bioasq":   sum(r["source"] == "bioasq"   for r in rs)}

        assert len(train) + len(test) == 100
        train_counts = src_counts(train)
        test_counts = src_counts(test)
        assert train_counts["pubmedqa"] > 0 and train_counts["bioasq"] > 0
        assert test_counts["pubmedqa"] > 0 and test_counts["bioasq"] > 0

    def test_falls_back_when_class_too_small(self):
        records = self._records(20, "pubmedqa") + [{"source": "rare", "question": "q", "answer": "a"}]
        train, test = split_dataset(records, test_size=0.2, random_state=0)
        assert len(train) + len(test) == 21

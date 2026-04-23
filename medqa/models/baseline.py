"""TF-IDF retrieval + most-relevant-sentence extraction (non-learned baseline)."""

import pickle
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from medqa._log import get_logger
from medqa.config import PROCESSED_DIR
from medqa.data.preprocessor import clean_text

log = get_logger("baseline")

_VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
_MATRIX_PATH     = PROCESSED_DIR / "tfidf_matrix.pkl"
_CORPUS_PATH     = PROCESSED_DIR / "tfidf_corpus.pkl"


class TFIDFBaseline:
    """TF-IDF + cosine similarity baseline for medical QA.

    Usage:
        model = TFIDFBaseline()
        model.fit(train_records)
        result = model.predict("What causes diabetes?")
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
        )
        self.tfidf_matrix = None
        self.corpus: list = []

    def fit(self, records: list) -> None:
        """Build TF-IDF index from training records.

        Args:
            records: list of dicts with at least 'context' and 'answer' keys.
        """
        self.corpus = records
        texts = [clean_text(r["context"]) for r in records]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        log.info("TF-IDF matrix: %s", self.tfidf_matrix.shape)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(_MATRIX_PATH, "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(_CORPUS_PATH, "wb") as f:
            pickle.dump(self.corpus, f)
        log.info("Index saved to disk.")

    def load(self) -> bool:
        """Load a previously saved index. Returns True if successful."""
        if not (_VECTORIZER_PATH.exists() and _MATRIX_PATH.exists()):
            return False
        with open(_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(_CORPUS_PATH, "rb") as f:
            self.corpus = pickle.load(f)
        log.info("Index loaded from disk.")
        return True

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Return the top_k most similar records to query, each with a 'score' key."""
        query_vec = self.vectorizer.transform([clean_text(query)])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results

    def predict(self, query: str) -> dict:
        """Retrieve the best matching context and extract the answer sentence.

        Returns a dict with:
            predicted_answer : the extracted sentence
            context          : the full matched context
            score            : retrieval similarity score
        """
        top_results = self.retrieve(query, top_k=1)
        if not top_results:
            return {"predicted_answer": "", "context": "", "score": 0.0}

        best = top_results[0]
        context = best["context"]
        answer_sentence = _extract_best_sentence(query, context, self.vectorizer)

        return {
            "predicted_answer": answer_sentence,
            "context":          context,
            "score":            best["score"],
            "gold_answer":      best.get("answer", ""),
        }

    def batch_predict(self, queries: list) -> list:
        """Run predict() over a list of queries."""
        return [self.predict(q) for q in queries]


_SENT_SPLIT_RE = None


def _split_sentences(text: str) -> list:
    """Lightweight sentence splitter, no NLTK / network dependency.

    Good enough for TF-IDF baseline: splits on '.', '!', '?' followed by
    whitespace, with a few common-abbreviation guards.
    """
    import re
    global _SENT_SPLIT_RE
    if _SENT_SPLIT_RE is None:
        # Split after .!? followed by whitespace; keep punctuation with the
        # preceding sentence. Not as accurate as Punkt but zero-dep and fast.
        _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _extract_best_sentence(query: str, context: str, vectorizer: TfidfVectorizer) -> str:
    """Split context into sentences and return the one most similar to query."""
    sentences = _split_sentences(context)
    if not sentences:
        return context

    query_vec = vectorizer.transform([clean_text(query)])
    sent_vecs = vectorizer.transform([clean_text(s) for s in sentences])
    scores = cosine_similarity(query_vec, sent_vecs).flatten()
    best_idx = int(np.argmax(scores))
    return sentences[best_idx]

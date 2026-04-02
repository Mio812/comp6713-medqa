"""
Baseline model: TF-IDF retrieval + most-relevant-sentence extraction.

This is intentionally simple — it serves as the lower-bound comparison for
the fine-tuned BERT and RAG models.

Pipeline:
  1. Build a TF-IDF matrix over all training contexts.
  2. Given a query, find the most similar context by cosine similarity.
  3. Extract the single sentence from that context with the highest
     cosine similarity to the query as the predicted answer.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from medqa.config import PROCESSED_DIR
from medqa.data.preprocessor import clean_text

_VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
_MATRIX_PATH = PROCESSED_DIR / "tfidf_matrix.pkl"
_CORPUS_PATH = PROCESSED_DIR / "tfidf_corpus.pkl"


class TFIDFBaseline:
    """
    TF-IDF + cosine similarity baseline for medical QA.

    Usage:
        model = TFIDFBaseline()
        model.fit(train_records)
        result = model.predict("What causes diabetes?")
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,   # unigrams + bigrams
            sublinear_tf=True,         # log-scale TF dampens frequent terms
            stop_words="english",
        )
        self.tfidf_matrix = None
        self.corpus: list[dict[str, Any]] = []  # stores original records

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, records: list[dict[str, Any]]) -> None:
        """
        Build TF-IDF index from training records.

        Args:
            records: list of dicts with at least 'context' and 'answer' keys.
        """
        self.corpus = records
        texts = [clean_text(r["context"]) for r in records]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[Baseline] TF-IDF matrix: {self.tfidf_matrix.shape}")

        # Persist to disk so we do not need to rebuild each run
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(_MATRIX_PATH, "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(_CORPUS_PATH, "wb") as f:
            pickle.dump(self.corpus, f)
        print("[Baseline] Index saved to disk.")

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
        print("[Baseline] Index loaded from disk.")
        return True

    # ── Inference ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return the *top_k* most similar records to *query*.
        Each returned dict includes an extra 'score' key.
        """
        query_vec = self.vectorizer.transform([clean_text(query)])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results

    def predict(self, query: str) -> dict[str, Any]:
        """
        Retrieve the best matching context and extract the answer sentence.

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

        # Extract the single sentence most similar to the query
        answer_sentence = _extract_best_sentence(query, context, self.vectorizer)

        return {
            "predicted_answer": answer_sentence,
            "context":          context,
            "score":            best["score"],
            "gold_answer":      best.get("answer", ""),
        }

    def batch_predict(self, queries: list[str]) -> list[dict[str, Any]]:
        """Run predict() over a list of queries."""
        return [self.predict(q) for q in queries]


# ── Sentence extraction helper ────────────────────────────────────────────────

def _extract_best_sentence(query: str, context: str, vectorizer: TfidfVectorizer) -> str:
    """
    Split *context* into sentences and return the one most similar to *query*.
    Falls back to the full context if splitting fails.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(context)
    if not sentences:
        return context

    query_vec = vectorizer.transform([clean_text(query)])
    sent_vecs = vectorizer.transform([clean_text(s) for s in sentences])
    scores = cosine_similarity(query_vec, sent_vecs).flatten()
    best_idx = int(np.argmax(scores))
    return sentences[best_idx]

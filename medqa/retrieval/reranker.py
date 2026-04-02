"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

After dense retrieval returns ~10 candidate chunks, the reranker scores
each (query, chunk) pair with a cross-encoder and keeps the top-k.
Cross-encoders are slower but significantly more accurate than bi-encoders
for final ranking.
"""

from typing import Any

from medqa.config import RERANKER_MODEL, RAG
from medqa.data.preprocessor import clean_text


class Reranker:
    """
    BGE cross-encoder reranker.

    Usage:
        reranker = Reranker()
        reranker.load()
        top3 = reranker.rerank(query, candidates, top_k=3)
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = None

    def load(self) -> None:
        """Load the cross-encoder model (downloaded automatically on first run)."""
        from FlagEmbedding import FlagReranker
        print(f"[Reranker] Loading {self.model_name} ...")
        self.model = FlagReranker(self.model_name, use_fp16=True)
        print("[Reranker] Ready.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = RAG["rerank_top_k"],
    ) -> list[dict[str, Any]]:
        """
        Score each candidate against *query* and return the top *top_k*.

        Args:
            query      : the user question (already cleaned)
            candidates : list of dicts from VectorStore.retrieve(), each
                         must have a 'text' key
            top_k      : how many results to return after reranking

        Returns:
            Sorted list (best first) with an added 'rerank_score' key.
        """
        if self.model is None:
            raise RuntimeError("Call load() before rerank().")
        if not candidates:
            return []

        q = clean_text(query)
        pairs = [[q, clean_text(c["text"])] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)

        # Attach scores and sort descending
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

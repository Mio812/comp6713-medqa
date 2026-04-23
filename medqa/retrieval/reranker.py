"""BGE cross-encoder reranker applied to the top-k bi-encoder candidates."""

from typing import Any

from medqa._log import get_logger
from medqa.config import RERANKER_MODEL, RAG
from medqa.data.preprocessor import clean_text

log = get_logger("reranker")


class Reranker:
    """
    BGE cross-encoder reranker via sentence-transformers.

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
        from sentence_transformers import CrossEncoder
        log.info("Loading %s ...", self.model_name)
        self.model = CrossEncoder(self.model_name, max_length=512)
        log.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = RAG["rerank_top_k"],
    ) -> list[dict[str, Any]]:
        """Score candidates against query; return top_k with a 'rerank_score' key."""
        if self.model is None:
            raise RuntimeError("Call load() before rerank().")
        if not candidates:
            return []

        q = clean_text(query)
        pairs = [[q, clean_text(c["text"])] for c in candidates]
        scores = self.model.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

"""End-to-end RAG pipeline: UMLS expansion -> dense retrieval -> rerank -> generation."""

from typing import Any

from medqa.retrieval.vectorstore import VectorStore
from medqa.retrieval.reranker import Reranker
from medqa.data.preprocessor import expand_query_with_umls, clean_text
from medqa.config import RAG


class RAGPipeline:
    """End-to-end RAG pipeline combining dense retrieval, reranking, and generation.

    Usage:
        from medqa.models.llm_qa import LocalLLM
        llm = LocalLLM()
        llm.load()

        pipeline = RAGPipeline(llm=llm)
        pipeline.build_index(records)
        result = pipeline.query("What is the first-line treatment for hypertension?")
    """

    def __init__(
        self,
        llm: Any,
        vector_store=None,
        reranker=None,
        retrieve_k: int = RAG["retrieve_top_k"],
        rerank_k: int = RAG["rerank_top_k"],
        use_query_expansion: bool = True,
        use_reranker: bool = True,
    ):
        """
        Args:
            use_query_expansion : if False, skip UMLS expansion (ablation).
            use_reranker        : if False, skip the cross-encoder reranker
                                  and pass the top-rerank_k dense hits
                                  straight to the LLM (ablation).
        """
        self.llm = llm
        self.vs = vector_store or VectorStore()
        self.reranker = reranker or Reranker()
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k
        self.use_query_expansion = use_query_expansion
        self.use_reranker = use_reranker
        self._reranker_loaded = False

    def build_index(self, records: list) -> None:
        """Populate the vector store with records."""
        self.vs.build(records)

    def query(self, question: str, answer_type: str = "auto") -> dict:
        """Run the full RAG pipeline for a single question."""
        question = clean_text(question)

        if self.use_query_expansion:
            expanded = expand_query_with_umls(question)
        else:
            expanded = question

        candidates = self.vs.retrieve(expanded, k=self.retrieve_k)

        if self.use_reranker:
            if not self._reranker_loaded:
                self.reranker.load()
                self._reranker_loaded = True
            top_chunks = self.reranker.rerank(question, candidates, top_k=self.rerank_k)
        else:
            top_chunks = candidates[: self.rerank_k]
            for c in top_chunks:
                c.setdefault("rerank_score", c.get("score", 0.0))

        context = _assemble_context(top_chunks)

        # Auto-load the LLM on first use if the caller forgot to call load().
        # This keeps the pipeline usable from notebooks whose kernel state
        # predates the explicit llm.load() cell.
        if getattr(self.llm, "pipeline", None) is None and hasattr(self.llm, "load"):
            self.llm.load()

        try:
            result = self.llm.predict(question, context, answer_type=answer_type)
        except TypeError:
            result = self.llm.predict(question, context)

        return {
            "question":          question,
            "expanded_query":    expanded,
            "retrieved_chunks":  candidates,
            "reranked_chunks":   top_chunks,
            "context":           context,
            "predicted_answer":  result["predicted_answer"],
        }

    def batch_query(
        self,
        questions: list,
        answer_types=None,
    ) -> list:
        """Run query() over a list of questions (with optional answer_types override)."""
        if answer_types is None:
            answer_types = ["auto"] * len(questions)
        return [self.query(q, at) for q, at in zip(questions, answer_types)]


def _assemble_context(chunks: list, max_chars: int = 2000) -> str:
    """Concatenate reranked chunks into a single context string, capped at max_chars."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        header = f"[Source {i}]"
        entry = f"{header} {text}"
        
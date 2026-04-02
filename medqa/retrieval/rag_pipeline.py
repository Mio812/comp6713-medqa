"""
Full RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  user question
      │
      ▼
  query expansion (UMLS via scispaCy)
      │
      ▼
  dense retrieval (ChromaDB + BGE-M3, top-10)
      │
      ▼
  reranking (BGE cross-encoder, top-3)
      │
      ▼
  context assembly
      │
      ▼
  LLM generation (Qwen / DeepSeek)
      │
      ▼
  answer + provenance
"""

from typing import Any

from medqa.retrieval.vectorstore import VectorStore
from medqa.retrieval.reranker import Reranker
from medqa.data.preprocessor import expand_query_with_umls, clean_text
from medqa.config import RAG


class RAGPipeline:
    """
    End-to-end RAG pipeline combining dense retrieval, reranking, and generation.

    Accepts either a LocalLLM or APILLM instance as the generator so the
    pipeline is backend-agnostic.

    Usage:
        from medqa.models.llm_qa import APILLM
        llm = APILLM(model="deepseek-chat")

        pipeline = RAGPipeline(llm=llm)
        pipeline.build_index(records)
        result = pipeline.query("What is the first-line treatment for hypertension?")
    """

    def __init__(
        self,
        llm: Any,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        retrieve_k: int = RAG["retrieve_top_k"],
        rerank_k: int = RAG["rerank_top_k"],
        use_query_expansion: bool = True,
    ):
        self.llm = llm
        self.vs = vector_store or VectorStore()
        self.reranker = reranker or Reranker()
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k
        self.use_query_expansion = use_query_expansion
        self._reranker_loaded = False

    # ── Index management ───────────────────────────────────────────────────────

    def build_index(self, records: list[dict[str, Any]]) -> None:
        """Populate the vector store with *records*."""
        self.vs.build(records)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Run the full RAG pipeline for a single *question*.

        Returns a dict with:
            question         : original question
            expanded_query   : question after UMLS expansion
            retrieved_chunks : raw retrieval results (pre-rerank)
            reranked_chunks  : top-k after reranking
            context          : assembled context string passed to LLM
            predicted_answer : LLM-generated answer
        """
        question = clean_text(question)

        # 1. Query expansion
        if self.use_query_expansion:
            expanded = expand_query_with_umls(question)
        else:
            expanded = question

        # 2. Dense retrieval
        candidates = self.vs.retrieve(expanded, k=self.retrieve_k)

        # 3. Reranking
        if not self._reranker_loaded:
            self.reranker.load()
            self._reranker_loaded = True
        top_chunks = self.reranker.rerank(question, candidates, top_k=self.rerank_k)

        # 4. Assemble context from top-k chunks
        context = _assemble_context(top_chunks)

        # 5. Generate answer
        result = self.llm.predict(question, context)

        return {
            "question":          question,
            "expanded_query":    expanded,
            "retrieved_chunks":  candidates,
            "reranked_chunks":   top_chunks,
            "context":           context,
            "predicted_answer":  result["predicted_answer"],
        }

    def batch_query(self, questions: list[str]) -> list[dict[str, Any]]:
        """Run query() over a list of questions."""
        return [self.query(q) for q in questions]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assemble_context(chunks: list[dict[str, Any]], max_chars: int = 2000) -> str:
    """
    Concatenate reranked chunks into a single context string.
    Truncates to *max_chars* to stay within LLM context limits.
    """
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        header = f"[Source {i}]"
        entry = f"{header} {text}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > len(header) + 20:
                entry = entry[:remaining] + "..."
                parts.append(entry)
            break
        parts.append(entry)
        total += len(entry) + 1   # +1 for newline

    return "\n\n".join(parts)

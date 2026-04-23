"""ChromaDB vector store backed by BGE-M3 embeddings for dense retrieval."""

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from medqa._log import get_logger
from medqa.config import CHROMA_DIR, EMBEDDING_MODEL, RAG
from medqa.data.preprocessor import clean_text

log = get_logger("vectorstore")


class VectorStore:
    """
    Persistent ChromaDB vector store backed by BGE-M3 embeddings.

    Usage:
        vs = VectorStore()
        vs.build(records)                        # index documents
        results = vs.retrieve("heart attack", k=10)
    """

    def __init__(
        self,
        collection_name: str = RAG["collection_name"],
        embedding_model: str = EMBEDDING_MODEL,
        persist_dir: str = str(CHROMA_DIR),
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        log.info("Loading embedding model: %s", embedding_model)
        self.embedder = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._cached_count: int | None = None
        log.info("Collection '%s' ready (%d docs).", collection_name, self.count())

    def build(self, records: list[dict[str, Any]], batch_size: int = 64) -> None:
        """Chunk and embed records, then upsert into ChromaDB."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG["chunk_size"],
            chunk_overlap=RAG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " "],
        )

        documents, metadatas, ids = [], [], []

        for record in records:
            context = clean_text(record.get("context", ""))
            if not context:
                continue

            doc_id = record.get("id", "")
            chunks = splitter.split_text(context)
            for chunk_idx, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source":    record.get("source", "unknown"),
                    "doc_id":    str(doc_id) if doc_id else "",
                    "chunk_idx": chunk_idx,
                })
                # UUIDs keep ids unique across re-runs of build().
                ids.append(uuid.uuid4().hex)

        if not documents:
            log.warning("No documents to index.")
            return

        log.info("Embedding %d chunks ...", len(documents))
        for i in range(0, len(documents), batch_size):
            batch_docs  = documents[i : i + batch_size]
            batch_meta  = metadatas[i : i + batch_size]
            batch_ids   = ids[i : i + batch_size]
            embeddings  = self.embedder.encode(
                batch_docs, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
            self.collection.upsert(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            log.info("  Upserted %d/%d", min(i + batch_size, len(documents)), len(documents))

        self._cached_count = None
        log.info("Index now contains %d chunks.", self.count())

    def retrieve(self, query: str, k: int = RAG["retrieve_top_k"]) -> list[dict[str, Any]]:
        """Embed query and return the k most similar chunks."""
        query_embedding = self.embedder.encode(
            [clean_text(query)], normalize_embeddings=True
        ).tolist()

        total = self.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":   doc,
                "score":  1 - dist,
                "source": meta.get("source", ""),
                "doc_id": meta.get("doc_id", ""),
            })
        return hits

    def count(self) -> int:
        """Return the number of indexed chunks (memoised)."""
        if self._cached_count is None:
            self._cached_count = self.collection.count()
        return self._cached_count

    def reset(self) -> None:
        """Delete and recreate the collection (clears the index)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._cached_count = 0
        log.info("Collection '%s' reset.", self.collection_name)

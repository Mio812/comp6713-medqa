"""
ChromaDB vector store for dense retrieval.

Embeds medical document chunks using BAAI/bge-m3 and stores them in a
persistent ChromaDB collection. Supports both building the index from
scratch and loading an existing one.
"""

from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from medqa.config import CHROMA_DIR, EMBEDDING_MODEL, RAG
from medqa.data.preprocessor import clean_text


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

        # BGE-M3: multilingual, strong on biomedical text
        print(f"[VectorStore] Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        print(f"[VectorStore] Collection '{collection_name}' ready "
              f"({self.collection.count()} docs).")

    # ── Index building ─────────────────────────────────────────────────────────

    def build(self, records: list[dict[str, Any]], batch_size: int = 64) -> None:
        """
        Chunk and embed *records*, then upsert into ChromaDB.

        Each record's context is split into overlapping chunks so long
        abstracts are indexed at a finer granularity.

        Args:
            records    : list of dicts with at least 'context' and 'question' keys
            batch_size : number of chunks to embed at once (tune for VRAM)
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG["chunk_size"],
            chunk_overlap=RAG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " "],
        )

        documents, metadatas, ids = [], [], []
        chunk_id = self.collection.count()   # continue numbering from existing docs

        for record in records:
            context = clean_text(record.get("context", ""))
            if not context:
                continue

            chunks = splitter.split_text(context)
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({
                    "question": record.get("question", "")[:200],
                    "answer":   record.get("answer",   "")[:200],
                    "source":   record.get("source",   "unknown"),
                })
                ids.append(f"doc_{chunk_id}")
                chunk_id += 1

        if not documents:
            print("[VectorStore] No documents to index.")
            return

        # Embed and upsert in batches
        print(f"[VectorStore] Embedding {len(documents)} chunks ...")
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
            print(f"  Upserted {min(i + batch_size, len(documents))}/{len(documents)}")

        print(f"[VectorStore] Index now contains {self.collection.count()} chunks.")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = RAG["retrieve_top_k"]) -> list[dict[str, Any]]:
        """
        Embed *query* and return the *k* most similar chunks.

        Returns a list of dicts with keys:
            text, score, question, answer, source
        """
        query_embedding = self.embedder.encode(
            [clean_text(query)], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":     doc,
                "score":    1 - dist,   # cosine distance → similarity
                "question": meta.get("question", ""),
                "answer":   meta.get("answer", ""),
                "source":   meta.get("source", ""),
            })
        return hits

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (clears the index)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Collection '{self.collection_name}' reset.")

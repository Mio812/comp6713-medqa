"""Build the RAG vector index (BGE-M3 embeddings, ChromaDB).

Embeds PubMedQA labelled (1k) + unlabelled (~61k) abstracts. ~30 min on GPU.
Output is persisted to chroma_db/ and loaded on subsequent runs.
"""
from medqa.data.loader import load_pubmedqa, load_pubmedqa_unlabeled
from medqa.models.llm_qa import LocalLLM
from medqa.retrieval.rag_pipeline import RAGPipeline

# build_index() only uses the embedder/vector store, so the LLM is not loaded here.
pipeline = RAGPipeline(llm=LocalLLM())

print("[index] gathering records ...")
records = load_pubmedqa() + load_pubmedqa_unlabeled()
print(f"[index] {len(records)} records to embed")

print("[index] building vector index (this takes ~30 min on GPU) ...")
pipeline.build_index(records)

print(f"[index] done. vector store contains {pipeline.vs.count()} chunks")

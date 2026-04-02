"""
Global configuration for the MedQA system.
All paths and hyperparameters are defined here.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# ── Datasets ──────────────────────────────────────────────────
PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
BIOASQ_DATASET = "kroshan/BioASQ"

# ── Models ────────────────────────────────────────────────────
# Encoder model for fine-tuning (BERT-based, extractive QA)
BERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Embedding model for RAG vector store
EMBEDDING_MODEL = "BAAI/bge-m3"

# Reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Decoder LLM for generative QA (set API key in .env or use local)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # or point to local path

# ── Fine-tuning hyperparameters ───────────────────────────────
BERT_FINETUNE = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "output_dir": str(ROOT_DIR / "checkpoints" / "bert_qa"),
}

# ── RAG settings ──────────────────────────────────────────────
RAG = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieve_top_k": 10,   # candidates before reranking
    "rerank_top_k": 3,      # final context passed to LLM
    "collection_name": "pubmed_abstracts",
}

# ── Evaluation ────────────────────────────────────────────────
EVAL = {
    "test_split": 0.2,
    "metrics": ["exact_match", "f1", "bertscore"],
}

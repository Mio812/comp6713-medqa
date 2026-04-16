"""Global configuration for the MedQA system."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = ROOT_DIR / "chroma_db"

PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
BIOASQ_DATASET = "kroshan/BioASQ"

BERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"

SEED = 42

BERT_FINETUNE = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "loss_type": "focal_ls",
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
    "output_dir": str(ROOT_DIR / "checkpoints" / "bert_qa"),
}

RAG = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieve_top_k": 10,
    "rerank_top_k": 3,
    "collection_name": "pubmed_abstracts",
}

EVAL = {
    "test_split": 0.2,
    "metrics": ["exact_match", "f1", "bertscore"],
}

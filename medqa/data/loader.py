"""
Data loader for PubMedQA and BioASQ datasets.

Both datasets are loaded directly from HuggingFace — no registration required.
Each loader returns a standardised dict with keys:
    question (str), context (str), answer (str), answer_type (str)
"""

from datasets import load_dataset
from typing import Any
from medqa.config import PUBMEDQA_DATASET, PUBMEDQA_CONFIG, BIOASQ_DATASET


# ── PubMedQA ─────────────────────────────────────────────────────────────────

def load_pubmedqa(split: str = "train") -> list[dict[str, Any]]:
    """
    Load the PubMedQA labelled subset (1 000 expert-annotated QA pairs).

    Each item contains:
      - question : the research question
      - context  : concatenated abstract sentences (the evidence)
      - answer   : long-form answer string
      - answer_type: 'yes' | 'no' | 'maybe'

    Args:
        split: 'train' (PubMedQA has no official train/test split, so we
               return all 1 000 items and split later in preprocessing).
    """
    raw = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG)

    # PubMedQA uses the key "train" for the full labelled set
    records = []
    for item in raw["train"]:
        # context is a list of sentences; join into a single string
        context_sentences = item.get("context", {}).get("contexts", [])
        context = " ".join(context_sentences)

        records.append({
            "question":    item["question"],
            "context":     context,
            "answer":      item.get("long_answer", ""),
            "answer_type": item.get("final_decision", ""),   # yes / no / maybe
            "source":      "pubmedqa",
        })

    print(f"[PubMedQA] Loaded {len(records)} records.")
    return records


def load_pubmedqa_unlabeled() -> list[dict[str, Any]]:
    """
    Load the larger unlabelled PubMedQA subset (~61 k items).
    Useful for building the RAG vector store (no gold answers needed).
    """
    raw = load_dataset(PUBMEDQA_DATASET, "pqa_unlabeled")
    records = []
    for item in raw["train"]:
        context_sentences = item.get("context", {}).get("contexts", [])
        context = " ".join(context_sentences)
        records.append({
            "question": item["question"],
            "context":  context,
            "source":   "pubmedqa_unlabeled",
        })
    print(f"[PubMedQA-Unlabeled] Loaded {len(records)} records.")
    return records


# ── BioASQ ───────────────────────────────────────────────────────────────────

def load_bioasq() -> list[dict[str, Any]]:
    """
    Load BioASQ from HuggingFace (no registration required).

    Normalises the varied BioASQ answer formats into the same schema as
    PubMedQA so both datasets can be used interchangeably downstream.
    """
    raw = load_dataset(BIOASQ_DATASET, trust_remote_code=True)

    records = []
    for split_name in raw:
        for item in raw[split_name]:
            # BioASQ stores snippets as evidence context
            snippets = item.get("snippets", [])
            if isinstance(snippets, list):
                context = " ".join(
                    s.get("text", "") if isinstance(s, dict) else str(s)
                    for s in snippets
                )
            else:
                context = str(snippets)

            # ideal_answer can be a list; join into one string
            ideal = item.get("ideal_answer", "") or ""
            if isinstance(ideal, list):
                ideal = " ".join(ideal)

            records.append({
                "question":    item.get("body", ""),
                "context":     context,
                "answer":      ideal,
                "answer_type": item.get("type", ""),   # factoid / yesno / list / summary
                "source":      "bioasq",
            })

    print(f"[BioASQ] Loaded {len(records)} records.")
    return records


# ── Combined loader ───────────────────────────────────────────────────────────

def load_all() -> list[dict[str, Any]]:
    """Return PubMedQA + BioASQ merged into a single list."""
    return load_pubmedqa() + load_bioasq()

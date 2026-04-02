"""
Data loader for PubMedQA, BioASQ, PubMedQA-Unlabeled, and MedQA-USMLE datasets.

All datasets are loaded directly from HuggingFace — no registration required.
Each loader returns a standardised dict with keys:
    question (str), context (str), answer (str), answer_type (str)
"""

from datasets import load_dataset
from typing import Any
from medqa.config import PUBMEDQA_DATASET, PUBMEDQA_CONFIG, BIOASQ_DATASET

MEDQA_USMLE_DATASET = "GBaker/MedQA-USMLE-4-options"


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

def _parse_bioasq_text(text: str) -> tuple[str, str]:
    """
    Parse the BioASQ 'text' field which has format:
        <answer> ... <context> ...
    Returns (answer, context).
    """
    import re
    answer_match = re.search(r"<answer>\s*(.*?)\s*(?:<context>|$)", text, re.DOTALL)
    context_match = re.search(r"<context>\s*(.*)", text, re.DOTALL)
    answer  = answer_match.group(1).strip()  if answer_match  else ""
    context = context_match.group(1).strip() if context_match else text
    return answer, context


def load_bioasq() -> list[dict[str, Any]]:
    """
    Load BioASQ from HuggingFace (no registration required).

    The kroshan/BioASQ dataset has two fields per item:
        question : the biomedical question string
        text     : '<answer> ... <context> ...' combined field

    This function parses those into the same schema as PubMedQA so both
    datasets can be used interchangeably downstream.
    """
    raw = load_dataset(BIOASQ_DATASET)

    records = []
    for split_name in raw:
        for item in raw[split_name]:
            question = item.get("question", "").strip()
            raw_text = item.get("text", "")
            answer, context = _parse_bioasq_text(raw_text)

            if not question or not context:
                continue

            records.append({
                "question":    question,
                "context":     context,
                "answer":      answer,
                "answer_type": "factoid",   # kroshan/BioASQ is factoid-style
                "source":      "bioasq",
            })

    print(f"[BioASQ] Loaded {len(records)} records.")
    return records


# ── MedQA-USMLE ──────────────────────────────────────────────────────────────

def load_medqa_usmle() -> list[dict[str, Any]]:
    """
    Load MedQA-USMLE 4-option multiple choice dataset (~12k items).

    Each item is a US medical licensing exam question with 4 options and
    a correct answer. We format the options into the context field so the
    downstream models can reason over them.

    Format normalised to match PubMedQA schema:
        question    : the exam question stem
        context     : the 4 answer options formatted as plain text
        answer      : the correct option text
        answer_type : 'mcq'
        source      : 'medqa_usmle'
    """
    raw = load_dataset(MEDQA_USMLE_DATASET)

    records = []
    for split_name in raw:
        for item in raw[split_name]:
            question = item.get("question", "").strip()
            options  = item.get("options", {})   # dict: {"A": "...", "B": "...", ...}
            answer_key = item.get("answer_idx", item.get("answer", ""))

            if not question:
                continue

            # Format options as readable context
            if isinstance(options, dict):
                context = " | ".join(f"{k}: {v}" for k, v in options.items())
                answer  = options.get(str(answer_key), str(answer_key))
            else:
                context = str(options)
                answer  = str(answer_key)

            records.append({
                "question":    question,
                "context":     context,
                "answer":      answer,
                "answer_type": "mcq",
                "source":      "medqa_usmle",
            })

    print(f"[MedQA-USMLE] Loaded {len(records)} records.")
    return records


# ── Combined loaders ──────────────────────────────────────────────────────────

def load_all() -> list[dict[str, Any]]:
    """
    Return PubMedQA + BioASQ + MedQA-USMLE merged into a single list.
    Total: ~21k labelled QA pairs.
    """
    return load_pubmedqa() + load_bioasq() + load_medqa_usmle()


def load_rag_corpus() -> list[dict[str, Any]]:
    """
    Return the full corpus for building the RAG vector store.
    Includes unlabelled PubMedQA (~61k) for richer retrieval coverage.
    """
    return load_pubmedqa() + load_pubmedqa_unlabeled() + load_bioasq() + load_medqa_usmle()

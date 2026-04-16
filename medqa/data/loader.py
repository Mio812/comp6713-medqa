"""Dataset loaders for PubMedQA, BioASQ, and MedQA-USMLE (all via HuggingFace)."""

from datasets import load_dataset
from typing import Any

from medqa._log import get_logger
from medqa.config import PUBMEDQA_DATASET, PUBMEDQA_CONFIG, BIOASQ_DATASET

log = get_logger("loader")

MEDQA_USMLE_DATASET = "GBaker/MedQA-USMLE-4-options"


def load_pubmedqa(split: str = "train") -> list[dict[str, Any]]:
    """Load the PubMedQA labelled subset (1 000 expert-annotated QA pairs)."""
    raw = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG)

    records = []
    for item in raw["train"]:
        context_sentences = item.get("context", {}).get("contexts", [])
        context = " ".join(context_sentences)

        records.append({
            "question":    item["question"],
            "context":     context,
            "answer":      item.get("long_answer", ""),
            "answer_type": item.get("final_decision", ""),
            "source":      "pubmedqa",
        })

    log.info("PubMedQA: loaded %d records.", len(records))
    return records


def load_pubmedqa_unlabeled() -> list[dict[str, Any]]:
    """Load the larger unlabelled PubMedQA subset (~61k items) for RAG indexing."""
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
    log.info("PubMedQA-Unlabeled: loaded %d records.", len(records))
    return records


def _parse_bioasq_text(text: str) -> tuple[str, str]:
    """Parse BioASQ 'text' field of form '<answer>...<context>...' into (answer, context)."""
    import re
    answer_match = re.search(r"<answer>\s*(.*?)\s*(?:<context>|$)", text, re.DOTALL)
    context_match = re.search(r"<context>\s*(.*)", text, re.DOTALL)
    answer  = answer_match.group(1).strip()  if answer_match  else ""
    context = context_match.group(1).strip() if context_match else text
    return answer, context


def load_bioasq() -> list[dict[str, Any]]:
    """Load BioASQ (kroshan/BioASQ) and normalise to the PubMedQA schema."""
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
                "answer_type": "factoid",
                "source":      "bioasq",
            })

    log.info("BioASQ: loaded %d records.", len(records))
    return records


def load_medqa_usmle() -> list[dict[str, Any]]:
    """Load MedQA-USMLE 4-option MCQ dataset, normalised to the PubMedQA schema."""
    raw = load_dataset(MEDQA_USMLE_DATASET)

    records = []
    for split_name in raw:
        for item in raw[split_name]:
            question = item.get("question", "").strip()
            options  = item.get("options", {})
            answer_key = item.get("answer_idx", item.get("answer", ""))

            if not question:
                continue

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

    log.info("MedQA-USMLE: loaded %d records.", len(records))
    return records


def load_all() -> list[dict[str, Any]]:
    """Return PubMedQA + BioASQ + MedQA-USMLE merged into a single list (~21k QA pairs)."""
    return load_pubmedqa() + load_bioasq() + load_medqa_usmle()


def load_rag_corpus() -> list[dict[str, Any]]:
    """Return the full corpus for building the RAG vector store (includes unlabelled PubMedQA)."""
    return load_pubmedqa() + load_pubmedqa_unlabeled() + load_bioasq() + load_medqa_usmle()

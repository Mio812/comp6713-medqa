"""
Data preprocessing and medical term normalisation.

Uses scispaCy (en_core_sci_lg) as a drop-in alternative to UMLS — no
registration or API key required. scispaCy can recognise biomedical named
entities and link them to UMLS concept IDs when the linker is enabled.

Pipeline:
  1. Basic text cleaning  (whitespace, lowercasing for retrieval)
  2. Medical NER          (identify entities with scispaCy)
  3. Query expansion      (append canonical UMLS names to user query)
  4. Train/test split
"""

import re
import string
from typing import Any

from sklearn.model_selection import train_test_split as sk_split
from medqa.config import EVAL


# ── scispaCy setup (lazy-loaded to avoid slow import at module level) ─────────

_nlp = None

def _get_nlp():
    """Lazy-load the scispaCy model the first time it is needed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_sci_lg")
            # Add the UMLS entity linker for canonical concept lookup
            _nlp.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            print("[Preprocessor] scispaCy + UMLS linker loaded.")
        except Exception as e:
            # Graceful fallback: NER unavailable, still do basic cleaning
            print(f"[Preprocessor] scispaCy not available ({e}). Skipping NER.")
            _nlp = None
    return _nlp


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Collapse multiple whitespace characters
      - Strip leading/trailing whitespace
      - Remove non-printable characters
    Does NOT lowercase (case is preserved for BERT tokenisation).
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)          # normalise whitespace
    text = re.sub(r"[^\x20-\x7E]", " ", text) # remove non-ASCII control chars
    return text.strip()


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply clean_text to all string fields of a record."""
    return {k: (clean_text(v) if isinstance(v, str) else v) for k, v in record.items()}


# ── Medical NER & query expansion ─────────────────────────────────────────────

def extract_medical_entities(text: str) -> list[str]:
    """
    Return a list of recognised biomedical entity strings from *text*.
    Falls back to an empty list if scispaCy is unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def expand_query_with_umls(query: str) -> str:
    """
    Append canonical UMLS names for recognised entities to the query.

    Example:
        Input : "heart attack treatment"
        Output: "heart attack treatment myocardial infarction"

    This improves retrieval recall when the user writes informal terms.
    """
    nlp = _get_nlp()
    if nlp is None:
        return query   # no expansion if scispaCy unavailable

    doc = nlp(query)
    expansions = []
    for ent in doc.ents:
        # Each entity may link to multiple UMLS concepts; take the top one
        if ent._.kb_ents:
            top_cui, _ = ent._.kb_ents[0]
            linker = nlp.get_pipe("scispacy_linker")
            canonical_name = linker.kb.cui_to_entity[top_cui].canonical_name
            if canonical_name.lower() not in query.lower():
                expansions.append(canonical_name)

    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ── Train / test split ────────────────────────────────────────────────────────

def split_dataset(
    records: list[dict[str, Any]],
    test_size: float = EVAL["test_split"],
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split *records* into train and test sets.

    Returns:
        (train_records, test_records)
    """
    train, test = sk_split(records, test_size=test_size, random_state=random_state)
    print(f"[Preprocessor] Split → train: {len(train)}, test: {len(test)}")
    return list(train), list(test)


# ── Tokenisation helper for BERT ──────────────────────────────────────────────

def truncate_context(context: str, question: str, tokenizer, max_length: int = 512) -> str:
    """
    Truncate *context* so that question + context fits within *max_length*
    tokens. Removes tokens from the end of the context.
    """
    question_tokens = tokenizer.tokenize(question)
    # Reserve space for [CLS], [SEP], question tokens, [SEP]
    available = max_length - len(question_tokens) - 3
    context_tokens = tokenizer.tokenize(context)
    if len(context_tokens) > available:
        context_tokens = context_tokens[:available]
    return tokenizer.convert_tokens_to_string(context_tokens)


# ── Normalise answer for evaluation ──────────────────────────────────────────

def normalise_answer(answer: str) -> str:
    """
    Lower-case, remove punctuation and extra whitespace.
    Used when computing Exact Match so minor formatting differences are ignored.
    """
    answer = answer.lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer

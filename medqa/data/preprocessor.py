"""Text cleaning, BERT span annotation helpers, and UMLS query expansion."""

import re
import string
from typing import Any

from sklearn.model_selection import train_test_split as sk_split

from medqa._log import get_logger
from medqa.config import EVAL, SEED

log = get_logger("preprocessor")


_nlp = None

def _get_nlp():
    """Lazy-load the scispaCy model the first time it is needed.

    The scispacy.linking and scispacy.abbreviation imports look unused but
    are load-bearing: they register the factories used by add_pipe below.
    """
    global _nlp
    if _nlp is None:
        try:
            import spacy
            from scispacy.linking import EntityLinker          # noqa: F401
            from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
            _nlp = spacy.load("en_core_sci_lg")
            _nlp.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            log.info("scispaCy + UMLS linker loaded.")
        except Exception as e:
            log.warning("scispaCy not available (%s). Skipping NER.", e)
            _nlp = None
    return _nlp


def clean_text(text: str) -> str:
    """Normalise whitespace and strip ASCII control chars; preserves Unicode and case."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text.strip()


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply clean_text to all string fields of a record."""
    return {k: (clean_text(v) if isinstance(v, str) else v) for k, v in record.items()}


def expand_query_with_umls(query: str) -> str:
    """Append canonical UMLS names for recognised entities to the query."""
    nlp = _get_nlp()
    if nlp is None:
        return query

    doc = nlp(query)
    expansions = []
    for ent in doc.ents:
        if ent._.kb_ents:
            top_cui, _ = ent._.kb_ents[0]
            linker = nlp.get_pipe("scispacy_linker")
            canonical_name = linker.kb.cui_to_entity[top_cui].canonical_name
            if canonical_name.lower() not in query.lower():
                expansions.append(canonical_name)

    if expansions:
        return query + " " + " ".join(expansions)
    return query


def split_dataset(
    records: list[dict[str, Any]],
    test_size: float = EVAL["test_split"],
    random_state: int = SEED,
    stratify_by: str = "source",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train and test sets, stratified by source."""
    strata = None
    if stratify_by:
        strata = [r.get(stratify_by, "") for r in records]
        if any(strata.count(s) < 2 for s in set(strata)):
            strata = None
    train, test = sk_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )
    log.info(
        "Split -> train: %d, test: %d (stratified=%s)",
        len(train), len(test), strata is not None,
    )
    return list(train), list(test)


def truncate_context(context: str, question: str, tokenizer, max_length: int = 512) -> str:
    """Truncate context so question + context fits within max_length tokens."""
    question_tokens = tokenizer.tokenize(question)
    # Reserve slots for [CLS], [SEP], question tokens, [SEP]
    available = max_length - len(question_tokens) - 3
    context_tokens = tokenizer.tokenize(context)
    if len(context_tokens) > available:
        context_tokens = context_tokens[:available]
    return tokenizer.convert_tokens_to_string(context_tokens)


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)

_ANSWER_PREFIXES = (
    "answer:", "a:", "the answer is", "final answer:", "based on the context",
)


def normalise_answer(answer: str) -> str:
    """SQuAD-style answer normalisation: lowercase, strip prefix/punctuation/articles."""
    if answer is None:
        return ""
    answer = str(answer).lower().strip()

    for prefix in _ANSWER_PREFIXES:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break

    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = _ARTICLES_RE.sub(" ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


_YESNO_TOKENS = ("yes", "no", "maybe")


def extract_yesno(answer: str):
    """Return 'yes' / 'no' / 'maybe' if the answer clearly expresses one, else None.

    Tolerant to 'Yes.', 'Yes, because...', 'The answer is no', 'No - ...'.
    """
    if not answer:
        return None
    norm = normalise_answer(answer)
    if not norm:
        return None
    tokens = norm.split()
    first = tokens[0]
    if first in _YESNO_TOKENS:
        return first
    for t in tokens:
        if t in _YESNO_TOKENS:
            return t
    return None


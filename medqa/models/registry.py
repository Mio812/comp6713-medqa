"""Central factory for QA backends.

Each loader returns (backend, predict_fn). Loaders are lazy so importing the
LLM module (which pulls in torch + transformers) is deferred until requested.
"""

from typing import Callable


def _load_baseline():
    from medqa.models.baseline import TFIDFBaseline
    from medqa.data.loader import load_all
    from medqa.data.preprocessor import split_dataset

    m = TFIDFBaseline()
    if not m.load():
        records, _ = split_dataset(load_all())
        m.fit(records)
    return m, lambda mdl, q, **kw: mdl.predict(q)


def _load_bert():
    from medqa.models.bert_qa import PubMedBERTQA
    from medqa.models.baseline import TFIDFBaseline
    from medqa.data.loader import load_all
    from medqa.data.preprocessor import split_dataset

    m = PubMedBERTQA()
    if not m.load():
        raise RuntimeError("No fine-tuned BERT checkpoint found. Run fine-tuning first.")

    # BERT needs a context at inference time; retrieve one with the baseline.
    retriever = TFIDFBaseline()
    if not retriever.load():
        records, _ = split_dataset(load_all())
        retriever.fit(records)

    def predict(mdl, q, **kw):
        top = retriever.retrieve(q, top_k=1)
        context = top[0]["context"] if top else ""
        return mdl.predict(q, context)

    return m, predict


def _load_rag():
    from medqa.models.llm_qa import LocalLLM
    from medqa.retrieval.rag_pipeline import RAGPipeline
    from medqa.data.loader import load_pubmedqa, load_pubmedqa_unlabeled

    llm = LocalLLM()
    llm.load()
    pipeline = RAGPipeline(llm=llm)
    if pipeline.vs.count() == 0:
        pipeline.build_index(load_pubmedqa() + load_pubmedqa_unlabeled())
    return pipeline, lambda mdl, q, **kw: mdl.query(q, answer_type=kw.get("answer_type", "auto"))


_REGISTRY: dict[str, Callable] = {
    "baseline": _load_baseline,
    "bert":     _load_bert,
    "rag":      _load_rag,
}


def available_backends() -> list[str]:
    return list(_REGISTRY.keys())


def get_backend(name: str):
    """Return (model, predict_fn) for name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Available: {available_backends()}")
    return _REGISTRY[name]()

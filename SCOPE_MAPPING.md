# Scope → Code Mapping

This file maps each item claimed in the project scope document to the exact
file and function(s) that implement it. Assessors can use this to verify that
every claimed credit item is reflected in the code-base.

## Part A — Problem Definition (10 credits claimed)

| Scope item          | Credits | Where in the code                                                  |
| ------------------- | ------- | ------------------------------------------------------------------ |
| NLP problem: QA     | 5       | `medqa/models/{baseline,bert_qa,llm_qa}.py` — three QA backends    |
| Domain: Medical     | 5       | `medqa/data/loader.py::load_pubmedqa`, `load_bioasq`               |

## Part B — Dataset Selection (20 credits claimed)

| Scope item                | Credits | Where in the code                                |
| ------------------------- | ------- | ------------------------------------------------ |
| Two existing datasets     | 10      | `medqa/data/loader.py::load_pubmedqa`, `load_bioasq`, `load_medqa_usmle` |
| Existing lexicon (UMLS)   | 10      | `medqa/data/preprocessor.py::expand_query_with_umls` (scispaCy + UMLS linker) |

## Part C — Modelling (40+ credits claimed)

| Scope item                                | Credits | Where in the code                              |
| ----------------------------------------- | ------- | ---------------------------------------------- |
| Baseline (rule-based / statistical)       | —       | `medqa/models/baseline.py::TFIDFBaseline`      |
| Fine-tune PubMedBERT                      | 20      | `medqa/models/bert_qa.py::PubMedBERTQA.fine_tune` |
| LLM + external tools (LangChain-style RAG)| 20      | `medqa/retrieval/rag_pipeline.py::RAGPipeline` + `retrieval/vectorstore.py` (LangChain text-splitter) + `retrieval/reranker.py` |
| *Method extension* (prompting beyond 0/few-shot) | 30 | `medqa/models/llm_qa.py::_build_prompt` — answer-type-conditioned prompting + `medqa/retrieval/rag_pipeline.py::query` — UMLS query expansion + BGE cross-encoder re-ranking (a non-trivial prompting + tool chain) |

## Part D — Evaluation (30 credits claimed)

| Scope item                 | Credits | Where in the code                                       |
| -------------------------- | ------- | ------------------------------------------------------- |
| Quantitative evaluation    | 10      | `medqa/evaluation/metrics.py::evaluate` (EM, F1, BERTScore, ROUGE-L, Yes/No accuracy, per-source breakdown) |
| Qualitative evaluation     | 5       | `medqa/evaluation/qualitative.py::analyse_errors` (error categorisation: WRONG_RETRIEVAL, HALLUCINATION, …) |
| Command-line testing       | 5       | `medqa/cli.py` (entry point `uv run medqa --question ...`) |
| Demo                       | 10      | `main.py` (Gradio web UI on port 7860)                  |

## Total: **100 credits** (minimum: 80)

---

## How to reproduce each claim

```bash
# Install
uv sync

# 1. Train the baseline (Part C)
uv run python -c "from medqa.models.baseline import TFIDFBaseline; \
from medqa.data.loader import load_all; from medqa.data.preprocessor import split_dataset; \
m=TFIDFBaseline(); tr,_=split_dataset(load_all()); m.fit(tr)"

# 2. Fine-tune PubMedBERT (Part C)
uv run python -c "from medqa.models.bert_qa import PubMedBERTQA; \
from medqa.data.loader import load_all; from medqa.data.preprocessor import split_dataset; \
m=PubMedBERTQA(); tr,te=split_dataset(load_all()); m.fine_tune(tr, te)"

# 3. Full evaluation with per-dataset breakdown (Part D, Quantitative + LLM-as-judge)
uv run python run_eval.py --n 100 --llm-judge

# 4. Qualitative error analysis (Part D, Qualitative)
uv run python -c "import json; from medqa.evaluation.qualitative import analyse_errors; \
preds=json.load(open('evaluation_predictions.json')); \
rows=[{'question':r['question'],'predicted_answer':r['pred'],'gold_answer':r['gold'],'context':r.get('context','')} for r in preds['Fine-tuned PubMedBERT']]; \
analyse_errors(rows, output_path='error_report.json')"

# 5. CLI testing (Part D)
uv run medqa --question "What is the first-line treatment for hypertension?" --mode rag

# 6. Gradio demo (Part D)
uv run python main.py  # open http://localhost:7860
```

# MedQA: Medical Literature Question Answering System

A system that answers medical questions by retrieving evidence from PubMed literature and generating grounded responses. Three backends are implemented and compared: a TF-IDF baseline, a fine-tuned PubMedBERT model, and a full RAG (Retrieval-Augmented Generation) pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Datasets](#4-datasets)
5. [Model Architectures & Loss Functions](#5-model-architectures--loss-functions)
6. [Method Extensions](#6-method-extensions)
7. [Hyperparameters](#7-hyperparameters)
8. [Running the System](#8-running-the-system)
9. [Evaluation Results](#9-evaluation-results)
10. [Engineering Decisions](#10-engineering-decisions)
11. [References](#11-references)

---

## 1. Overview

Given a medical question such as "What is the first-line treatment for hypertension?", the system finds relevant paragraphs from biomedical literature (PubMed) and extracts or generates an answer.

Medical language is specialised. The same concept appears under many names ("heart attack", "MI", "myocardial infarction", "acute coronary event"), and answers are often spread across multiple sentences, so plain keyword search fails in these cases.

Three approaches are compared in this project:

1. **TF-IDF Baseline.** A classical, statistics-based approach. Represents every word as a number based on how often it appears. Fast and interpretable, but cannot handle synonyms or paraphrase.

2. **Fine-tuned PubMedBERT.** A neural language model pre-trained on PubMed abstracts, further trained on our QA datasets so it learns to highlight the exact answer span within a given passage.

3. **RAG (Retrieval-Augmented Generation).** Combines a search engine (to find relevant passages) with a large language model (to write a fluent answer). Uses the UMLS medical ontology to expand the search query with medical synonyms before retrieval.

---

## 2. Project Structure

The package is split by pipeline stage: `data/` loads and normalises the datasets, `models/` holds the three backends, `retrieval/` contains the vector store and reranker, and `evaluation/` contains the metrics. All hyperparameters live in `config.py`. Each backend exposes the same `predict(question, context) -> dict` so the evaluation loop is backend-agnostic.

```
comp6713-medqa/
│
├── medqa/                        # Main Python package
│   ├── config.py                 # Paths, model names, and hyperparameters
│   ├── cli.py                    # Command-line entry point (uv run medqa ...)
│   │
│   ├── data/
│   │   ├── loader.py             # Dataset loaders: PubMedQA, BioASQ, MedQA-USMLE
│   │   └── preprocessor.py       # Text cleaning, UMLS query expansion, train/test split,
│   │                             #   answer span localisation for BERT annotation
│   │
│   ├── models/
│   │   ├── baseline.py           # TF-IDF + cosine similarity
│   │   ├── bert_qa.py            # Fine-tuned PubMedBERT extractive QA
│   │   ├── llm_qa.py             # Generative backend: LocalLLM (Qwen2.5-14B-Instruct, 4-bit NF4)
│   │   └── registry.py           # Backend registry used by CLI / demo
│   │
│   ├── retrieval/
│   │   ├── vectorstore.py        # ChromaDB vector store, BGE-M3 embeddings, LangChain chunking
│   │   ├── reranker.py           # BGE cross-encoder reranker
│   │   └── rag_pipeline.py       # End-to-end RAG: expansion, retrieval, rerank, generate
│   │
│   ├── training/
│   │   └── custom_loss.py        # Focal loss and label smoothing
│   │
│   └── evaluation/
│       ├── metrics.py            # EM, Token F1, BERTScore, ROUGE-L, Yes/No accuracy,
│       │                         #   LLM-as-judge, retrieval Recall@k / MRR, per-source breakdown
│       └── qualitative.py        # Heuristic error categorisation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset statistics and sample inspection
│   ├── 02_baseline.ipynb             # TF-IDF training and qualitative examples
│   ├── 03_bert_finetuning.ipynb      # BERT fine-tuning with loss curves
│   ├── 04_rag_pipeline.ipynb         # RAG index building and end-to-end demo
│   └── 05_evaluation.ipynb           # Metric comparison across all backends
│
├── data/
│   └── processed/                    # Serialised TF-IDF index (auto-generated on first run)
│
├── checkpoints/
│   ├── bert_qa/                      # Default fine-tuned PubMedBERT checkpoint
│   ├── bert_qa_hp/                   # HP-optimised fine-tuned PubMedBERT checkpoint
│   └── hp_search/                    # Per-trial Optuna checkpoints
│
├── chroma_db/                        # Persistent ChromaDB vector index
│
├── reports/                          # Evaluation artefacts
│   ├── evaluation_results_default.json     # Scalar metrics with default BERT
│   ├── evaluation_results_hp.json          # Scalar metrics with HP-optimised BERT
│   ├── evaluation_predictions_default.json # Per-sample predictions, default
│   ├── evaluation_predictions_hp.json      # Per-sample predictions, HP-optimised
│   ├── ablation_results.json               # RAG ablation sweep scalar metrics
│   ├── ablation_table.md                   # Ablation comparison table
│   ├── hp_search_results.json              # Optuna trial summary + best params
│   ├── _checkpoint_<backend>.jsonl         # Per-backend resume checkpoints
│   └── _ablation_ckpt_<key>.jsonl          # Per-config resume checkpoints
│
├── scripts/                          # Reproducibility entry points (also wired into .vscode/tasks.json)
│   ├── setup_scispacy.py             # Install the scispaCy UMLS linker model
│   ├── build_rag_index.py            # Embed PubMedQA (1k + 61k) into chroma_db/
│   ├── hp_search.py                  # Optuna hyperparameter search for PubMedBERT
│   ├── finetune_bert.py              # Fine-tune PubMedBERT on the train split
│   └── ablation.py                   # RAG ablation sweep
│
├── tests/                            # Unit tests (preprocessor, metrics)
│
├── .vscode/
│   ├── tasks.json                    # One-click tasks and pipeline composites (see §8)
│   ├── launch.json                   # Debug configurations
│   └── settings.json                 # Python interpreter and folder settings
│
├── main.py                           # Gradio web demo (http://localhost:7860)
├── run_eval.py                       # Full evaluation script
├── SCOPE_MAPPING.md                  # Maps marking-guide credit claims to code locations
└── pyproject.toml                    # Dependencies and project metadata (managed by uv)
```

All three model backends use the same data format (`{question, context, answer, answer_type, source}`) defined in `loader.py`. Switching backends requires no upstream code changes.

---

## 3. Environment Setup

This section assumes a fresh machine with nothing installed.

### 3.0 System Requirements

| Component | Version / Minimum |
|-----------|-------------------|
| OS        | Windows 10/11, macOS 12+, or Linux (Ubuntu 22.04+) |
| Python    | 3.11 to 3.13 |
| CUDA      | 11.8 / 12.x / 13.x (only needed for GPU) |
| GPU VRAM  | >= 10 GB for RAG/Qwen-14B (4-bit); 16 GB recommended |
| Disk      | ~30 GB free (models + vector index) |

Without a GPU, the TF-IDF baseline still runs. RAG/BERT tasks require a CUDA device.

### 3.1 Install prerequisites

Python 3.11 from https://www.python.org/downloads/. Check "Add Python to PATH" during install.

VS Code from https://code.visualstudio.com/. After install, open VS Code and install these extensions:

- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance (Microsoft)

On Windows, install Visual C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and select "Desktop development with C++". This is required to compile scispaCy.

For GPU users, install the NVIDIA driver and CUDA toolkit matching your GPU:

| GPU generation | CUDA toolkit |
|---|---|
| RTX 30-series / 40-series | CUDA 12.1 |
| RTX 50-series (Blackwell) | CUDA 12.8+ or CUDA 13.x |

Verify with:

```powershell
nvidia-smi
```

### 3.2 Install `uv` (Python package manager)

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal, then verify:

```bash
uv --version
```

`uv` is used instead of plain `pip` because it resolves dependencies faster and generates a `uv.lock` that pins exact versions.

### 3.3 Clone the repository and open it in VS Code

```bash
git clone <repo-url>
cd comp6713-medqa
code .
```

### 3.4 Install project dependencies

Inside VS Code, open the integrated terminal (`` Ctrl+` ``) and run:

```bash
uv sync
```

First install takes about 5 minutes. `uv sync` reads `pyproject.toml`, creates `.venv/`, and pins exact versions from `uv.lock`.

All models (Qwen2.5-14B-Instruct, BGE-M3, BGE-reranker, PubMedBERT) are downloaded from HuggingFace on first use and cached locally. No external API keys are required.

`uv sync` installs the `scispacy` Python package, but the model weights (`en_core_sci_lg`) live on allenai's S3 bucket and are not on PyPI, so they need a one-off install. Either run the VS Code task `0'. Install scispaCy model` (after §3.6), or from the terminal:

```bash
uv run python scripts/setup_scispacy.py
```

This step is idempotent and safe to re-run. The Pipeline composite tasks chain it in automatically. If the install fails (no network, proxy, etc.), the rest of the pipeline still runs; `preprocessor.py` skips UMLS query expansion and the log prints `Skipping NER`. The ablation study (§9) will then show `use_query_expansion=True` and `False` producing identical numbers, so fix this step if you want the expansion ablation to be meaningful.

### 3.5 Match PyTorch to your CUDA version

The bundled `pyproject.toml` defaults to the CUDA 12.8 PyTorch index (works for Blackwell GPUs and most recent drivers). If your CUDA version is different, edit the two blocks below in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url  = "https://download.pytorch.org/whl/cu128"   # change to match your CUDA
explicit = true

[tool.uv.sources]
torch       = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Common wheel URL suffixes:

| CUDA | Suffix |
|---|---|
| CUDA 11.8 | `cu118` |
| CUDA 12.1 | `cu121` |
| CUDA 12.4 | `cu124` |
| CUDA 12.8 | `cu128` |
| CUDA 13.0 | `cu130` |

After editing, re-install PyTorch:

```bash
uv sync --reinstall-package torch --reinstall-package torchvision
```

### 3.6 Select the Python interpreter in VS Code

`Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose `./.venv/Scripts/python.exe` (Windows) or `./.venv/bin/python` (macOS/Linux).

One-time setup. The bundled `.vscode/settings.json` keeps it selected across reopens.

### 3.7 Verify GPU is wired up

In VS Code: `Ctrl+Shift+P` -> `Tasks: Run Task` -> `0. GPU check`.

Expected output:

```
cuda = True
device = NVIDIA GeForce RTX ...
torch = 2.x.x+cu128
```

If `cuda = False`, recheck the CUDA toolkit install and the `pyproject.toml` index URL from step 3.5.

`fp16` mixed-precision training is enabled automatically when a CUDA device is detected.

---

## 4. Datasets

### 4.1 Dataset Overview

| Dataset | Size | Task type | Answer format | Source |
|---------|------|-----------|---------------|--------|
| PubMedQA (labelled) | 1,000 | Literature QA | yes / no / maybe + long answer | `qiaojin/PubMedQA` |
| BioASQ | ~3,000 | Biomedical QA | factoid sentence | `kroshan/BioASQ` |
| MedQA-USMLE | ~12,000 | Multiple-choice | one of A/B/C/D | HuggingFace |
| PubMedQA (unlabelled) | ~61,000 | RAG corpus only | n/a | `qiaojin/PubMedQA` |

All datasets are loaded from HuggingFace and normalised to a unified schema in `data/loader.py`. No manual download or registration is required.

### 4.2 Dataset Annotation Quality

All labelled datasets were created by domain experts and have documented annotation quality.

PubMedQA questions are derived from the title of PubMed papers; answers come from the conclusion sections written by the paper authors themselves. The 1,000 expert-labelled samples were independently annotated by two biomedical researchers, with a reported inter-annotator agreement of kappa = 0.78 (Cohen's kappa, substantial agreement). The yes/no/maybe classification and long-form answer were both checked for consistency.

BioASQ was constructed as a shared challenge task. Questions were formulated by biomedical experts and answers were validated through a structured annotation protocol. Multiple annotators independently verified factoid answers against referenced PubMed abstracts. The challenge organisers report high agreement (>0.85 accuracy when cross-validated against an expert gold standard).

PubMedQA's `long_answer` field is a free-form expert summary rather than an exact quote from the context. To obtain span labels for BERT training we apply a deterministic annotation algorithm (see §4.3). Since the algorithm is rule-based, there is no inter-annotator disagreement by construction.

### 4.3 Custom BERT Span Annotation

BERT's extractive QA head learns to highlight a contiguous text span (for example, the phrase "ACE inhibitors or calcium channel blockers") within a longer passage as the answer. To train BERT, the exact character position where the answer starts and ends is required.

PubMedQA answers are expert summaries, not direct quotes. The gold answer "ACE inhibitors are recommended as first-line therapy" may never appear verbatim in the source context, so a direct substring search often fails.

Annotation strategy (implemented in `preprocessor.py -> _prepare_dataset`):

```
For each training record:
  1. Clean both context and answer text
     (normalise whitespace, remove non-printable control characters)
  2. Truncate context to fit within 512 tokens alongside the question
  3. Search for answer[:50] as a substring in context (case-insensitive)
  4. If found    -> record char-level start position as the span label
  5. If not found -> set start = 0  (CLS token fallback)
  6. end position = start + len(answer)
```

The CLS fallback in step 5 avoids discarding samples. Setting the label to the `[CLS]` position follows the SQuAD v2 convention for unanswerable questions. The BERT loss still computes a valid gradient from this label; the model learns to predict CLS when no span is found, which also acts as a confidence signal at inference time.

Sliding-window tokenisation handles long contexts. BERT can only process 512 tokens at a time; longer abstracts are split into overlapping windows (each 512 tokens, overlapping by 128 tokens) so the answer is visible in at least one window:

- `max_length = 512` tokens total (question + context + 3 special tokens)
- `stride = 128` tokens: consecutive windows overlap by 128 tokens
- `return_overflowing_tokens = True`: one record may produce multiple training windows

---

## 5. Model Architectures & Loss Functions

### 5.1 TF-IDF Baseline

TF-IDF weights each term by how frequently it appears in a document relative to the whole corpus, then finds the most query-relevant document via cosine similarity over those weighted vectors.

Architecture: `TfidfVectorizer(max_features=50,000, ngram_range=(1,2), sublinear_tf=True)` with cosine similarity and top-sentence extraction.

No training objective. The baseline serves as a lower bound to measure how much the neural models improve over lexical matching.

The TF-IDF representation of "heart attack" and "myocardial infarction" are completely different vectors (the two phrases share no words). A patient asking about "heart attack" would not retrieve a paper that only uses the clinical term. This is the core limitation motivating both BERT and RAG.

### 5.2 Fine-tuned PubMedBERT

BERT reads a question and a passage jointly and predicts the contiguous span of text within the passage that answers the question.

Architecture: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`.

PubMedBERT is a BERT-base model (12 transformer layers, 768 hidden dimensions, 12 attention heads, 110M parameters) pre-trained from scratch on the entire PubMed corpus. Unlike BioBERT, which adapts general BERT, PubMedBERT's vocabulary and weights are entirely biomedical, giving it a stronger understanding of medical terminology from the start.

Fine-tuning task: extractive span prediction (SQuAD-style). Two linear classification heads are placed on top of BERT's output:

```
H in R^(seq_len x 768)        # BERT output: one 768-dim vector per token
start_logits = H . W_start    # W_start in R^(768 x 1)
end_logits   = H . W_end      # W_end   in R^(768 x 1)
```

Loss function: cross-entropy loss applied independently to both span endpoints, then summed:

```
L = CrossEntropy(start_logits, start_label)
  + CrossEntropy(end_logits,   end_label)
```

The model is penalised independently for incorrect start and end positions; the two losses are summed and minimised jointly. This is the SQuAD default and gives per-token probabilities the inference code uses as a confidence score.

When the gold answer cannot be located in the context (see §4.3), both labels are set to position 0 (the `[CLS]` token). A high CLS probability at inference time acts as a "not answerable from this passage" signal, which `qualitative.py` uses for error categorisation.

Training procedure:

1. Tokenise with `truncation="longest_first"` and sliding window (`stride=128`).
2. Align char-level labels to token-level labels using `offset_mapping`.
3. Train with `Trainer` (HuggingFace) using `DefaultDataCollator`.
4. Save best checkpoint by validation loss; reload for inference.

### 5.3 RAG Pipeline (Qwen-14B)

RAG (Retrieval-Augmented Generation) first retrieves the most relevant PubMed abstract chunks for a given query, then passes those chunks to a large language model to generate a grounded answer, rather than relying solely on the model's parametric knowledge.

A five-stage pipeline:

| Stage | Component | Function |
|-------|-----------|----------|
| Query expansion | scispaCy + UMLS linker | Adds medical synonyms to the query (e.g. "heart attack" becomes "heart attack myocardial infarction") |
| Dense retrieval | ChromaDB + BGE-M3 | Converts query to a dense vector and finds the 10 most similar document chunks |
| Reranking | BGE-reranker-v2-m3 | Re-scores the top 10 chunks jointly with the query and keeps the 3 most relevant |
| Context assembly | `_assemble_context()` | Concatenates the 3 best chunks with labelled headers, up to 2,000 characters |
| Generation | Qwen2.5-14B-Instruct | Reads the retrieved context and writes a fluent answer |

Qwen is used inference-only and was instruction-tuned by the Alibaba team. Our contribution is the retrieval pipeline, UMLS query expansion, two-stage retrieve-then-rerank, and answer-type-conditioned prompting (§6.3).

Full float16 inference of Qwen2.5-14B requires about 28 GB VRAM. 4-bit NF4 quantisation via BitsAndBytes reduces VRAM usage to about 9 GB with less than 2% accuracy loss in our evaluations. This makes the model runnable on a single consumer GPU with at least 10 GB VRAM (e.g. RTX 3090, 4090).

---

## 6. Method Extensions

### 6.1 UMLS Ontology Incorporation

The Unified Medical Language System (UMLS) is a database maintained by the US National Library of Medicine. It maps over 3.5 million biomedical concepts to their canonical names, synonyms, and abbreviations. For example, it knows that "heart attack", "MI", "myocardial infarction", and "AMI" all refer to the same underlying concept (CUI C0027051).

The standard RAG retrieval pipeline uses the user's query verbatim. We add a query expansion step that uses the scispaCy NLP library to:

1. Detect medical entity mentions in the query (e.g. "heart attack").
2. Look up each entity's canonical UMLS concept (e.g. `C0027051 -> "Myocardial Infarction"`).
3. Append the canonical name to the query if it is not already present.

This injects structured ontological knowledge into the neural retrieval process, bridging informal patient language and formal clinical terminology without requiring any labelled training examples.

Implementation: `preprocessor.py -> expand_query_with_umls()` using `scispacy_linker` with `linker_name="umls"`. Degrades gracefully (returns original query unchanged) if scispaCy is unavailable.

Example:

```
Input query:    "treatment for heart attack in elderly patients"
Expanded query: "treatment for heart attack in elderly patients Myocardial Infarction"
```

The expanded query now matches both patient-language documents ("heart attack") and clinical-language documents ("myocardial infarction").

### 6.2 Custom Span Annotation for Unannotatable Samples

As described in §4.3, we convert PubMedQA's free-form expert summaries into span-level labels before fine-tuning. About 40% of samples (mostly yes/no/maybe questions whose gold answer doesn't appear verbatim in the context) have no direct span match; for these we label the `[CLS]` token instead of dropping the sample.

### 6.3 Answer-Type-Conditioned Prompting

PubMedQA gold answers are one word (`yes` / `no` / `maybe`), BioASQ-factoid gold answers are short noun phrases from the abstract, and only the free-form subset expects a full sentence. An instruction-tuned LLM defaults to writing full sentences, so EM collapses to near zero even when the answer is semantically right.

`llm_qa.py` picks one of three system prompts based on `answer_type`:

- `yesno`: reply must be a single word from {yes, no, maybe}; `max_new_tokens=4`.
- `factoid`: reply must be a 1-8 word noun phrase copied from the context; `max_new_tokens=32`.
- `free`: at most two sentences; default token budget.

`answer_type` is passed in by `run_eval.py` (derived from the record's `source`) or inferred by `_detect_answer_type()` using the first word of the question. `RAGPipeline.query()` forwards `answer_type` to the LLM, falling back to the two-argument call via `TypeError` for older backends.

### 6.4 Answer Normalisation and Yes/No Extraction

Two helpers in `preprocessor.py` support the metrics:

- `normalise_answer()`: SQuAD-style lowercase, strip `a` / `an` / `the`, strip leading `Answer:` / `The answer is:` / `Final answer:` prefixes, drop punctuation, collapse whitespace. Used by EM and F1 so that "Answer: Yes." matches a gold `yes`.
- `extract_yesno()`: maps fluent replies like "Yes, because..." or "No, the evidence is weak" back to `yes` / `no` / `maybe`. Drives `yesno_accuracy()` in `metrics.py`, which is a more useful number than EM on PubMedQA.

---

## 7. Hyperparameters

All hyperparameters are centralised in `medqa/config.py`.

### BERT Fine-tuning

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `learning_rate` | 2e-5 | Standard range for BERT fine-tuning. Lower (1e-5) converges too slowly; higher (5e-5) risks catastrophic forgetting on small datasets. |
| `num_train_epochs` | 3 | Sufficient for convergence on ~16k combined samples; more epochs risk overfitting the 1k PubMedQA labelled set. |
| `per_device_train_batch_size` | 16 | Maximum that fits in 16 GB VRAM with fp16 and 512-token sequences. |
| `warmup_steps` | 500 | Around 3% of total training steps. Prevents large gradient updates before model weights stabilise. |
| `weight_decay` | 0.01 | L2 regularisation on non-bias parameters. |
| `max_length` | 512 | BERT's hard token limit. |
| `stride` | 128 | 25% of max_length. Trades more training windows for better answer coverage. |
| `fp16` | auto (GPU) | Mixed-precision: about 2x memory saving, 1.5x speed-up, with negligible accuracy change. |

### RAG / Vector Store

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `chunk_size` | 512 chars | Aligns with BERT's token budget; keeps each chunk on a single sub-topic. |
| `chunk_overlap` | 64 chars | About 12% overlap. Prevents information loss when a sentence is split across chunks. |
| `retrieve_top_k` | 10 | Wide enough for the cross-encoder reranker to recover relevant chunks. |
| `rerank_top_k` | 3 | About 1,500 chars, fits in the LLM context window without diluting the signal. |
| `max_new_tokens` | 4 / 32 / 512 | Dispatched by `answer_type`: 4 for yes/no, 32 for factoid, 512 for free-form (§6.3). |
| `temperature` | 0.0 (greedy) | Deterministic outputs for reproducible evaluation runs. |

---

## 8. Running the System

Two execution paths are supported: VS Code tasks (recommended, one-click) and terminal commands (for CI or headless machines).

### 8.1 Running via VS Code tasks

After completing §3 setup, open the project in VS Code. Twelve tasks are pre-defined in `.vscode/tasks.json`:

| # | Task | Time | GPU VRAM | Produces |
|---|------|------|----------|----------|
| 0 | GPU check | 10 s | - | stdout only |
| 0' | Install scispaCy UMLS model | ~5 min | - | `en_core_sci_lg` weights in the uv env |
| 1 | Cache datasets | ~5 min | - | HuggingFace cache |
| 2 | Build RAG index | ~30 min | ~2 GB | `chroma_db/` |
| 3 | Fit TF-IDF | ~1 min | - | `checkpoints/tfidf.pkl` |
| 4a | BERT HP search (Optuna, optional) | ~1-2 h | ~5 GB | `reports/hp_search_results.json` + `checkpoints/hp_search/` |
| 4b | Fine-tune PubMedBERT (default params) | ~30 min | ~5 GB | `checkpoints/bert_qa/` |
| 4b' | Fine-tune PubMedBERT with HP-search best params | ~30 min | ~5 GB | `checkpoints/bert_qa_hp/` |
| 5a | RAG ablation sweep | ~1 h | ~12 GB | `reports/ablation_results.json` + `reports/ablation_table.md` |
| 5b | Evaluation with default BERT (n=200) | ~1-2 h | ~12 GB peak | `reports/evaluation_results_default.json` + predictions |
| 5b' | Evaluation with HP-optimised BERT (n=200) | ~1-2 h | ~12 GB peak | `reports/evaluation_results_hp.json` + predictions |
| 6 | Launch Gradio demo | continuous | ~12 GB | Web UI at http://localhost:7860 |

Run any task: `Ctrl+Shift+P` -> `Tasks: Run Task` -> select.

Composite pipelines (for unattended overnight runs):

- `Pipeline: Prepare only (0 -> 1 -> 2b)`: builds all artefacts, no evaluation.
- `Pipeline: Full run with default params (0 -> ... -> 3b)`: artefacts + ablation + evaluation.
- `Pipeline: HP search + fine-tune (best params) + eval (2a -> 2b' -> 3b')`: optional second track that fine-tunes with Optuna-best hyperparameters and re-runs evaluation so 3b vs 3b' is directly comparable.

Pipelines run their child tasks in sequence; if any step fails, the remaining steps are skipped.

The RAG index (task 1) is the longest GPU step, so it runs first after the scispaCy model install. The ablation sweep (3a) is placed before 3b/3b' because it reuses the shared LLM and vector store, so running it first primes warm caches for the longer 200-sample end-to-end runs.

Dataset downloads (PubMedQA, BioASQ, MedQA-USMLE) are cached by `datasets` on first load; the TF-IDF baseline is fit inline from `notebooks/02_baseline.ipynb`. Neither needs a separate wrapper script.

### 8.2 Running via terminal

If you prefer the terminal or are on a headless machine, call the scripts directly:

```bash
# 0. Install scispaCy UMLS linker model
uv run python scripts/setup_scispacy.py

# 1. Build RAG index
uv run python scripts/build_rag_index.py

# 2a. (optional) BERT HP search with Optuna
uv run python scripts/hp_search.py --trials 15 --train 1000 --resume

# 2b. Fine-tune PubMedBERT with default params
uv run python scripts/finetune_bert.py

# 2b'. (optional) Fine-tune PubMedBERT with HP-search best params
uv run python scripts/finetune_bert.py --use-best-params

# 3a. RAG ablation sweep
uv run python scripts/ablation.py --n 60 --resume

# 3b. End-to-end evaluation with default BERT
uv run python run_eval.py --n 200 --tag default \
    --bert-checkpoint checkpoints/bert_qa --output-dir reports --resume

# 3b'. End-to-end evaluation with HP-optimised BERT
uv run python run_eval.py --n 200 --tag hp \
    --bert-checkpoint checkpoints/bert_qa_hp --output-dir reports --resume
```

Each script imports from `medqa/` exactly like the VS Code tasks.

### 8.3 Inspecting results

All evaluation artefacts live under `reports/`:

- `reports/evaluation_results_<tag>.json`: scalar metrics (EM / F1 / BERTScore / ROUGE-L / yes-no accuracy, with per-source breakdown and Retrieval Recall@k/MRR for Full RAG).
- `reports/evaluation_predictions_<tag>.json`: one row per test sample with the gold answer and each backend's prediction. Used by `evaluation/qualitative.py` for the error-category breakdown in §9.3.
- `reports/ablation_results.json` and `reports/ablation_table.md`: RAG ablation sweep (§9.5).
- `reports/hp_search_results.json`: Optuna trial summaries and best hyperparameters (§9.4).

Open any file directly in VS Code (`Ctrl+K V` for a side-by-side preview).

### 8.4 Interactive Web demo

Run task `6. Launch Gradio demo` (or `uv run python main.py` from the terminal). Open http://localhost:7860 in your browser to ask questions against any of the three backends.

### 8.5 Single-question CLI

```bash
# Ask one question
uv run medqa --question "What is the first-line treatment for hypertension?" --mode rag --show-context

# Batch-answer a file (one question per line)
uv run medqa --input questions.txt --mode rag --output answers.json
```

Supported `--mode` values: `baseline`, `bert`, `rag`.

### 8.6 Evaluation flags

```bash
uv run python run_eval.py --n 200                       # default is 100 test examples
uv run python run_eval.py --skip-rag                    # skip every LLM backend (CPU-only friendly)
uv run python run_eval.py --skip-full-rag               # keep LLM+gold-context diagnostic, skip retrieval
uv run python run_eval.py --llm-judge                   # additionally run LLM-as-judge semantic check
uv run python run_eval.py --tag default                 # suffix output files (e.g. *_default.json)
uv run python run_eval.py --bert-checkpoint checkpoints/bert_qa_hp  # swap BERT checkpoint
uv run python run_eval.py --output-dir reports --resume # default; --resume reuses per-backend checkpoints
```

### 8.7 Debugging

`.vscode/launch.json` defines five debug configurations (F5 to launch):

- `Run eval (200 samples)`: step through `run_eval.py`
- `Run eval (no RAG, CPU-friendly)`: fast iteration without loading Qwen
- `Gradio demo (main.py)`: debug the web UI
- `CLI - single question (RAG)`: step through the CLI path
- `Current file`: debug whatever file is focused in the editor

Set breakpoints by clicking left of a line number, then press F5 and pick a configuration.

### 8.8 Resuming / skipping steps

All build artefacts are self-checkpointing:

- `chroma_db/` present and non-empty: task 2 can be skipped.
- `checkpoints/tfidf.pkl` present: task 3 can be skipped.
- `checkpoints/bert_qa/` or `checkpoints/bert_qa_hp/` present: task 4b / 4b' can be skipped.
- `reports/_checkpoint_<backend>.jsonl` present + `--resume`: task 5b / 5b' picks up mid-loop.
- `reports/_ablation_ckpt_<key>.jsonl` present + `--resume`: task 5a picks up mid-sweep per configuration.

`run_eval.py` detects missing artefacts and skips the corresponding backend with a warning rather than crashing, so partial runs still produce results for whatever is available. A Full RAG failure prints the full traceback so you can diagnose mid-run crashes without losing the TF-IDF / BERT / LLM-gold rows that already completed.

---

## 9. Evaluation Results

### 9.1 Evaluation Protocol

`run_eval.py` runs four conditions on 200 held-out test samples (20% split, `random_state=42`; stratified across MedQA-USMLE / BioASQ / PubMedQA):

1. TF-IDF Baseline: TF-IDF top-1 with best-sentence extraction.
2. Fine-tuned PubMedBERT: extractive span prediction on the gold context.
3. LLM + gold context (diagnostic): Qwen on the gold context, used to measure the LLM's reasoning without retrieval error.
4. Full RAG: UMLS query expansion, dense retrieval, cross-encoder rerank, Qwen on the top-3 reranked chunks.

Reporting (3) and (4) separately makes the gap between them interpretable as the cost of retrieval.

`loader.py` stores PubMedQA's free-form `long_answer` in the `answer` field, with the authoritative `final_decision` label (`yes` / `no` / `maybe`) kept separately in `answer_type`. All backends prompt PubMedQA questions with the yes/no template and emit a single token, so scoring that token against a paragraph-length `long_answer` would structurally force token F1 close to 0. `run_eval.py -> _gold_for_metrics()` therefore uses `final_decision` as the gold for PubMedQA yes/no items, matching the accuracy definition in the original PubMedQA paper. Retrieval metrics (`recall@k` / MRR) still use the long-form `answer` as the target, because `_chunk_contains_answer` does a substring check and a one-word `yes` / `no` would match almost every chunk trivially.

### 9.2 Quantitative Results (Task 5b, default PubMedBERT)

Evaluated with SQuAD-style normalisation (§6.4) and, for the LLM conditions, answer-type-conditioned prompting (§6.3). Numbers are from `reports/evaluation_results_default.json` (n=200).

| Model | n | EM | Token F1 | BERTScore F1 | ROUGE-L | Yes/No Acc |
|-------|:-:|:---:|:---:|:---:|:---:|:---:|
| TF-IDF Baseline | 200 | 0.000 | 0.022 | 0.779 | 0.021 | 0.000 |
| Fine-tuned PubMedBERT | 200 | **0.350** | 0.387 | 0.848 | 0.389 | 0.375 |
| LLM + gold context (diagnostic) | 200 | 0.190 | **0.410** | **0.886** | **0.410** | 0.375 |
| Full RAG (retrieval + LLM) | 200 | 0.090 | 0.115 | 0.822 | 0.113 | 0.250 |

Per-source breakdown (mean F1):

| Model | MedQA-USMLE (n=103) | BioASQ (n=89) | PubMedQA (n=8) |
|-------|:---:|:---:|:---:|
| TF-IDF Baseline | 0.008 | 0.040 | 0.000 |
| Fine-tuned PubMedBERT | 0.136 | **0.697** | 0.188 |
| LLM + gold context | 0.226 | 0.626 | **0.375** |
| Full RAG (retrieval + LLM) | 0.044 | 0.184 | 0.250 |

Retrieval quality (Full RAG, n=200): `Recall@1=0.155`, `Recall@3=0.210`, `Recall@5=0.210`, `Recall@10=0.210`, `MRR=0.178`.

`LLM + gold context` is the generation upper bound (F1 = 0.410). Fine-tuned PubMedBERT matches it on exact-string matching (EM = 0.350 vs 0.190) because its extractive head copies answers verbatim, but falls short on F1 / BERTScore on free-form items. BioASQ's factoid format favours the extractive BERT (F1 = 0.697). Full RAG sits well below both (F1 = 0.115), and the gap is structural: Recall@1 = 0.155 means the right chunk is only retrieved 15.5% of the time, so the LLM often cannot generate a correct answer even if it could in principle. Recall@3 is approximately Recall@10 (0.21), so adding more candidates past the reranker's top-3 does not help because the bi-encoder rarely puts the gold chunk in the top-10 at all.

### 9.3 Qualitative Error Analysis

Implemented in `evaluation/qualitative.py`. Each incorrect prediction is categorised into one of the following error types:

| Error Type | Definition | Example |
|------------|-----------|---------|
| `WRONG_RETRIEVAL` | Retrieved passage does not contain the answer | Query about hypertension retrieves a diabetes paper |
| `HALLUCINATION` | Model generates a confident answer not supported by context | RAG invents a drug dosage not present in retrieved text |
| `PARTIAL` | Answer partially correct but incomplete or boundary error | BERT finds "ACE inhibitors" but misses "or ARBs" |
| `FORMAT_MISMATCH` | Correct information, wrong format | BERT outputs a full sentence where gold label is a single word |
| `UNANSWERABLE` | Question not answerable from the provided context | No relevant passage in the PubMed index |

Consistent with the quantitative numbers: RAG's primary failure mode is `WRONG_RETRIEVAL` (driven by the Recall@1 = 0.155 ceiling above) with `HALLUCINATION` as a secondary mode; BERT's failures are mostly `PARTIAL` (boundary errors on MedQA-USMLE) and `FORMAT_MISMATCH` on PubMedQA.

### 9.4 HP-Optimised vs Default (Task 5b vs 5b')

Task 4a runs a 15-trial Optuna search on a 1,000-sample training slice (val F1 objective). Task 4b' re-fine-tunes PubMedBERT with the best-trial hyperparameters, and task 5b' re-evaluates. Full results in `reports/evaluation_results_hp.json`.

| Backend | Default (5b) F1 | HP-optimised (5b') F1 | Delta |
|---|:---:|:---:|:---:|
| Fine-tuned PubMedBERT | 0.387 | 0.365 | -0.022 |
| &nbsp;&nbsp;&nbsp;&nbsp;on MedQA-USMLE | 0.136 | 0.070 | -0.066 |
| &nbsp;&nbsp;&nbsp;&nbsp;on BioASQ | 0.697 | 0.724 | +0.027 |
| &nbsp;&nbsp;&nbsp;&nbsp;on PubMedQA | 0.188 | 0.188 | 0.000 |

The HP search improves BioASQ slightly but regresses MedQA-USMLE more, so total F1 drops by 0.022. This is a typical overfit-to-the-val-slice outcome: Optuna optimised a single aggregate number on a 1,000-sample train / held-out val split, but the test set's source mix (USMLE-heavy) differs from that slice. Both numbers are reported rather than silently dropping 5b'. The LLM+gold and Full RAG columns are identical between 5b and 5b' (same LLM, same retrieval) and serve as a consistency check.

### 9.5 RAG Ablation Study (Task 5a)

`scripts/ablation.py` toggles each RAG component over 60 examples drawn from the same test split. Full results in `reports/ablation_results.json` and `reports/ablation_table.md`; top-line numbers below.

| Configuration | EM | F1 | BERTScore | Recall@1 | Recall@3 | MRR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw-query + dense@5 | 0.083 | 0.101 | 0.818 | 0.083 | 0.083 | 0.083 |
| raw-query + rerank@5 | **0.083** | **0.103** | **0.821** | **0.117** | **0.167** | **0.137** |
| query-exp + dense@5 | 0.033 | 0.051 | 0.814 | 0.083 | 0.083 | 0.083 |
| query-exp + rerank@5 | 0.067 | 0.087 | 0.817 | 0.117 | 0.167 | 0.136 |

Takeaways:

1. The BGE cross-encoder reranker is worth keeping. Adding it on top of dense retrieval raises Recall@1 by 41% (0.083 to 0.117), doubles Recall@3 (0.083 to 0.167), and lifts MRR by 64% (0.083 to 0.137), with a 3-4x latency cost per query.
2. UMLS query expansion is a negative finding on this dataset. In dense-only mode it halves EM (0.083 to 0.033); even with the reranker it underperforms raw queries (EM 0.067 vs 0.083). The expanded query dilutes the BGE-M3 embedding with canonical medical terms that shift the chunk distribution away from the topical match. The pipeline still supports the feature (it may help on patient-language queries the test set does not cover), but it is disabled by default in `run_eval.py` task 5b. A useful extension would be expansion conditional on entity-match confidence.
3. Larger `rerank_top_k` helps retrieval but not generation. Moving from top-1 to top-5 after reranking lifts Recall@5 to 0.183 (raw-query) but leaves EM / F1 essentially flat, because the LLM already copes with context volume and extra chunks mostly add noise. We keep `rerank_top_k = 3` as a latency / recall compromise.

Metric notes:

- EM / F1: computed after SQuAD-style normalisation (lowercase, strip articles, strip `Answer:` / `Final answer:` prefixes).
- BERTScore F1: semantic similarity under `roberta-base` embeddings. We use `roberta-base` instead of DeBERTa because DeBERTa triggers an int32 overflow on positional embeddings under Python 3.11.
- Yes/No Acc: PubMedQA only. `yesno_accuracy()` uses `extract_yesno()` so fluent outputs like "Yes, because..." are still counted correctly; answers that cannot be mapped to yes/no/maybe are reported as uncategorised and count as wrong.
- LLM-as-judge (optional, `--llm-judge`): a second LLM pass judging whether the candidate is semantically equivalent to the gold. Useful because EM / F1 under-score paraphrased-but-correct answers.

---

## 10. Engineering Decisions

### Unified data schema

All three loaders return the same `{question, context, answer, answer_type, source}` dictionary, so the evaluation loop, CLI, and Gradio demo are backend-agnostic. Adding a fourth model requires only implementing `predict(question, context) -> dict`.

### No end-to-end RAG fine-tuning

End-to-end RAG training requires differentiable retrieval and large batches of `(question, supporting documents, answer)` triples. Our combined dataset of ~21k samples is insufficient for stable joint training. The current design (fixed retriever, fixed LLM) remains reproducible and interpretable.

### ChromaDB over FAISS

ChromaDB persists the index to disk automatically. FAISS requires manual serialisation and re-loading logic. At our scale (~62k chunks), ChromaDB's HNSW index provides millisecond retrieval with no perceptible latency difference from FAISS, while reducing boilerplate.

### 4-bit quantisation (NF4)

Full float16 inference of Qwen2.5-14B requires about 28 GB VRAM. NF4 quantisation with double quantisation (BitsAndBytes) reduces this to about 9 GB with an empirical accuracy drop below 2%, making the model runnable on a single consumer GPU.

### Greedy decoding (`do_sample=False`)

Greedy decoding is deterministic: the same input always produces the same output. This is required for reproducible evaluation. Sampling-based decoding (nucleus sampling, beam search) would introduce variance across runs and make metric comparisons unreliable.

### Answer-type-conditioned prompting

Without it the LLM writes full sentences and scores EM near 0.02 on PubMedQA (gold label is the single token `yes`). Fixing the prompt keeps EM directly comparable across all three backends. The three templates (yesno / factoid / free) map onto the three gold-answer formats in the datasets.

### Two-stage retrieval (bi-encoder + cross-encoder)

- Bi-encoder (BGE-M3): encodes query and document independently. Retrieval is a vector dot product, scalable to millions of documents in milliseconds.
- Cross-encoder (BGE-reranker): encodes the `(query, document)` pair jointly, allowing attention to flow between both inputs. More accurate but slower (O(n) forward passes).

Using both stages in sequence gives near-cross-encoder accuracy at near-bi-encoder speed for the top-10 candidates.

### scispaCy + UMLS over general NLP tools

General-purpose NER tools (spaCy `en_core_web_lg`) are not trained on biomedical text and frequently miss or misclassify medical entities. `en_core_sci_lg` is trained on biomedical literature and directly supports UMLS linking, providing canonical concept IDs without any external API or registration.

---

## 11. References

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Gu et al. (2021). Domain-specific language model pretraining for biomedical NLP. (PubMedBERT)
- Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP 2019.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- Nogueira & Cho (2019). Passage Re-ranking with BERT.
- Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.
- Rajpurkar et al. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. ACL 2018.
- Tsatsaronis et al. (2015). An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition. BMC Bioinformatics.
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. (NF4 quantisation)

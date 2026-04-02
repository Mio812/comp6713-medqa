"""
Command-line interface for the MedQA system.

Supports two modes:
  1. Single question  : pass --question "..."
  2. Batch from file  : pass --input questions.txt (one question per line)

Backend is selected via --mode:
  baseline  : TF-IDF retrieval
  bert      : fine-tuned PubMedBERT (requires trained checkpoint)
  rag       : full RAG pipeline with Qwen/DeepSeek LLM

Usage examples:
  uv run medqa --question "What is the treatment for Type 2 diabetes?" --mode rag
  uv run medqa --input questions.txt --mode baseline --output answers.json
"""

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medqa",
        description="Medical Literature QA System — COMP6713",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--question", "-q",
        type=str,
        help="Single question to answer.",
    )
    group.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to a .txt file with one question per line.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["baseline", "bert", "rag"],
        default="rag",
        help="Which model backend to use (default: rag).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to show (RAG mode only, default: 3).",
    )
    return parser


def _load_model(mode: str):
    """Lazy-import and initialise the chosen backend."""
    if mode == "baseline":
        from medqa.models.baseline import TFIDFBaseline
        from medqa.data.loader import load_all
        model = TFIDFBaseline()
        if not model.load():
            print("[CLI] No saved index found — building from dataset ...")
            from medqa.data.preprocessor import split_dataset
            records, _ = split_dataset(load_all())
            model.fit(records)
        return model

    elif mode == "bert":
        from medqa.models.bert_qa import PubMedBERTQA
        model = PubMedBERTQA()
        if not model.load():
            print("[CLI] No fine-tuned checkpoint found. Run fine-tuning first.")
            sys.exit(1)
        return model

    elif mode == "rag":
        from medqa.models.llm_qa import APILLM
        from medqa.retrieval.rag_pipeline import RAGPipeline
        llm = APILLM()   # reads DEEPSEEK_API_KEY from .env
        pipeline = RAGPipeline(llm=llm)
        if pipeline.vs.count() == 0:
            print("[CLI] Vector store empty — indexing dataset ...")
            from medqa.data.loader import load_pubmedqa, load_pubmedqa_unlabeled
            records = load_pubmedqa() + load_pubmedqa_unlabeled()
            pipeline.build_index(records)
        return pipeline


def _predict(model, question: str, mode: str) -> dict:
    """Dispatch prediction to the correct backend."""
    if mode == "baseline":
        return model.predict(question)
    elif mode == "bert":
        # BERT needs a context; use retrieval result from baseline as fallback
        from medqa.models.baseline import TFIDFBaseline
        baseline = TFIDFBaseline()
        baseline.load()
        top = baseline.retrieve(question, top_k=1)
        context = top[0]["context"] if top else ""
        return model.predict(question, context)
    elif mode == "rag":
        return model.query(question)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load .env for API keys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print(f"[CLI] Loading backend: {args.mode}")
    model = _load_model(args.mode)

    # Collect questions
    if args.question:
        questions = [args.question]
    else:
        questions = args.input.read_text().strip().splitlines()
        questions = [q.strip() for q in questions if q.strip()]

    # Run predictions
    results = []
    for q in questions:
        print(f"\nQ: {q}")
        result = _predict(model, q, args.mode)
        answer = result.get("predicted_answer", "")
        print(f"A: {answer}")
        results.append({"question": q, **result})

    # Save output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[CLI] Results saved to {args.output}")


if __name__ == "__main__":
    main()

"""Command-line interface for the MedQA system."""

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    from medqa.models.registry import available_backends

    parser = argparse.ArgumentParser(
        prog="medqa",
        description="Medical Literature QA System",
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
        choices=available_backends(),
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
    parser.add_argument(
        "--answer-type",
        choices=["auto", "yesno", "factoid", "free"],
        default="auto",
        help="Force the LLM prompt template (rag / llm backends).",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="For rag mode: print the top reranked chunks used for each question.",
    )
    return parser


def _load_backend(mode: str):
    """Load a backend through the registry."""
    from medqa.models.registry import get_backend
    try:
        return get_backend(mode)
    except RuntimeError as e:
        print(f"[CLI] {e}")
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"[CLI] Loading backend: {args.mode}")
    model, predict_fn = _load_backend(args.mode)

    if args.question:
        questions = [args.question]
    else:
        questions = args.input.read_text().strip().splitlines()
        questions = [q.strip() for q in questions if q.strip()]

    results = []
    for q in questions:
        print(f"\nQ: {q}")
        result = predict_fn(model, q, answer_type=args.answer_type)
        answer = result.get("predicted_answer", "")
        print(f"A: {answer}")
        if args.show_context and "reranked_chunks" in result:
            for i, chunk in enumerate(result["reranked_chunks"], 1):
                text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                print(f"  [chunk {i}] {text[:200]}...")
        results.append({"question": q, **result})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[CLI] Results saved to {args.output}")


if __name__ == "__main__":
    main()

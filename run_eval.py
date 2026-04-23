"""End-to-end evaluation driver.

Runs four backends on the held-out test split and saves a results JSON,
per-example predictions, and per-backend resume checkpoints under reports/.

Backends:
    Baseline (TF-IDF)          TF-IDF retrieval + best-sentence extraction.
    Fine-tuned PubMedBERT      PubMedBERT span prediction on the gold context.
    LLM + gold context         Diagnostic: LLM with gold passage (upper bound).
    Full RAG (retrieval + LLM) Retrieval + reranking + LLM (realistic setting).

Usage:
    uv run python run_eval.py
    uv run python run_eval.py --n 200 --tag hp --bert-checkpoint checkpoints/bert_qa_hp
    uv run python run_eval.py --skip-rag
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from medqa._seed import set_seed
from medqa.config import SEED
from medqa.data.loader import load_all
from medqa.data.preprocessor import split_dataset
from medqa.evaluation.metrics import evaluate, print_results, llm_as_judge, retrieval_metrics
from medqa.models.baseline import TFIDFBaseline
from medqa.models.bert_qa import PubMedBERTQA


DEFAULT_OUTPUT_DIR = Path("reports")
CHECKPOINT_EVERY = 10


def _answer_type_for(record: dict) -> str:
    """Pick the best LLM prompt template for a test record."""
    source = record.get("source", "")
    atype  = record.get("answer_type", "")
    if source == "pubmedqa" or atype in {"yes", "no", "maybe"}:
        return "yesno"
    if source == "bioasq" or atype == "factoid":
        return "factoid"
    return "free"


def _gold_for_metrics(record: dict) -> str:
    """Return the gold answer used for EM / F1 / BERTScore / ROUGE.

    For PubMedQA yes/no/maybe items we compare the model's one-token reply
    against the final_decision label (yes / no / maybe) rather than the
    paragraph-long long_answer. Other sources keep their original gold.
    """
    source = record.get("source", "")
    atype  = record.get("answer_type", "")
    if source == "pubmedqa" and atype in {"yes", "no", "maybe"}:
        return atype
    return record.get("answer", "")


def _summarise(res: dict) -> dict:
    """Keep only scalar metrics for the saved JSON summary."""
    summary = {
        "n":         res.get("n_samples", 0),
        "EM":        res.get("mean_em", 0.0),
        "F1":        res.get("mean_f1", 0.0),
        "BERTScore": res.get("bertscore_f1"),
        "ROUGE-L":   res.get("rouge_l", 0.0),
    }
    yn = res.get("yesno", {})
    if yn.get("n"):
        summary["YesNo_Acc"]           = yn["accuracy"]
        summary["YesNo_N"]             = yn["n"]
        summary["YesNo_Uncategorised"] = yn["uncategorised"]
    if "per_source" in res:
        summary["per_source"] = res["per_source"]
    return summary


def _save(output_dir: Path, all_summaries: dict, all_predictions: dict, tag: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    results_path     = output_dir / f"evaluation_results{suffix}.json"
    predictions_path = output_dir / f"evaluation_predictions{suffix}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    trimmed = {
        name: [
            {**row, "context": (row.get("context") or "")[:500]}
            for row in rows
        ]
        for name, rows in all_predictions.items()
    }
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2)
    print(f"\n[Saved] Scalar results  -> {results_path}")
    print(f"[Saved] Per-example pred -> {predictions_path}")


def _checkpoint_path(output_dir: Path, backend: str) -> Path:
    safe = backend.replace(" ", "_").replace("/", "_").replace("+", "plus")
    return output_dir / f"_checkpoint_{safe}.jsonl"


def _load_checkpoint(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _run_with_checkpoint(
    backend_name: str,
    output_dir: Path,
    resume: bool,
    n_total: int,
    one_shot,
):
    """Run one_shot(i) over range(n_total) and append each row to a JSONL checkpoint."""
    ckpt_path = _checkpoint_path(output_dir, backend_name)
    existing = _load_checkpoint(ckpt_path) if resume else []
    done = {row["i"]: row for row in existing}
    if done:
        print(f"    [resume] {len(done)}/{n_total} already cached at {ckpt_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    f = open(ckpt_path, "a", encoding="utf-8")
    try:
        for i in range(n_total):
            if i in done:
                continue
            try:
                row = one_shot(i)
            except Exception as e:
                print(f"    [warn] sample {i} failed: {e}")
                row = {"pred": "", "error": str(e)}
            row = {"i": i, **row}
            done[i] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % CHECKPOINT_EVERY == 0:
                f.flush()
                print(f"    ... {i + 1}/{n_total}")
    finally:
        f.close()
    return [done[i] for i in range(n_total)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full MedQA evaluation.")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of test examples to evaluate (default: 100).")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip all LLM-based backends (useful without GPU).")
    parser.add_argument("--skip-full-rag", action="store_true",
                        help="Skip the full RAG pipeline but still run LLM+gold-ctx.")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Also run an LLM-as-judge correctness check.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write results + per-backend checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse per-backend JSONL checkpoints from --output-dir.")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for output filenames, e.g. --tag hp -> evaluation_results_hp.json.")
    parser.add_argument("--bert-checkpoint", type=Path, default=None,
                        help="Override BERT checkpoint path (e.g. checkpoints/bert_qa_hp).")
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    set_seed()

    print("[Step 1] Loading and splitting data ...")
    _, test = split_dataset(load_all())
    test_subset = test[:args.n]
    questions    = [r["question"]              for r in test_subset]
    contexts     = [r["context"]               for r in test_subset]
    golds        = [_gold_for_metrics(r)       for r in test_subset]
    # Retrieval substring/overlap check uses the paragraph-level answer; a
    # one-word yes/no gold would match almost every chunk.
    retrieval_golds = [r.get("answer", "")     for r in test_subset]
    sources      = [r.get("source", "unknown") for r in test_subset]
    answer_types = [_answer_type_for(r)        for r in test_subset]

    print(f"  -> Evaluating on {len(test_subset)} examples")
    print(f"  -> Source distribution: "
          f"{dict((s, sources.count(s)) for s in set(sources))}")

    summaries:   dict = {}
    predictions: dict = {}

    # Baseline (TF-IDF)
    print("\n[Step 2] Evaluating Baseline (TF-IDF) ...")
    bl = TFIDFBaseline()
    if bl.load():
        bl_preds = [bl.predict(q)["predicted_answer"] for q in questions]
        bl_res = evaluate(bl_preds, golds, sources=sources, use_bertscore=True)
        print_results(bl_res, model_name="Baseline (TF-IDF)")
        summaries["Baseline (TF-IDF)"] = _summarise(bl_res)
        predictions["Baseline (TF-IDF)"] = [
            {"question": q, "gold": g, "pred": p, "source": s}
            for q, g, p, s in zip(questions, golds, bl_preds, sources)
        ]
    else:
        print("  -> No saved TF-IDF index found. Run fit() first. Skipping.")

    # Fine-tuned BERT
    print("\n[Step 3] Evaluating Fine-tuned PubMedBERT ...")
    bert = PubMedBERTQA()
    if args.bert_checkpoint:
        from pathlib import Path as _P
        bert.output_dir = _P(args.bert_checkpoint)
        print(f"  -> Using custom checkpoint: {bert.output_dir}")
    if bert.load():
        bert_out  = bert.batch_predict(questions, contexts)
        bert_preds = [p["predicted_answer"] for p in bert_out]
        bert_res   = evaluate(bert_preds, golds, sources=sources, use_bertscore=True)
        print_results(bert_res, model_name="Fine-tuned PubMedBERT")
        summaries["Fine-tuned PubMedBERT"] = _summarise(bert_res)
        predictions["Fine-tuned PubMedBERT"] = [
            {"question": q, "gold": g, "pred": p, "source": s, "context": c}
            for q, g, p, s, c in zip(questions, golds, bert_preds, sources, contexts)
        ]
    else:
        print("  -> No BERT checkpoint found. Run fine_tune() first. Skipping.")

    # LLM backends
    llm = None
    if not args.skip_rag:
        print("\n[Step 4] Loading LLM backend ...")
        try:
            from medqa.models.llm_qa import LocalLLM
            llm = LocalLLM()
            llm.load()
        except Exception as e:
            print(f"  -> LLM load failed ({e}). Skipping all LLM-based conditions.")
            llm = None

    if llm is not None:
        # LLM with gold context (diagnostic upper bound).
        print("\n[Step 4a] Evaluating LLM + gold context (diagnostic) ...")

        def llm_gold_shot(i):
            out = llm.predict(questions[i], contexts[i], answer_type=answer_types[i])
            return {"pred": out.get("predicted_answer", "")}

        rows = _run_with_checkpoint(
            "LLM_gold_context", output_dir, args.resume, len(questions), llm_gold_shot
        )
        llm_preds = [r.get("pred", "") for r in rows]

        llm_res = evaluate(llm_preds, golds, sources=sources, use_bertscore=True)
        print_results(llm_res, model_name="LLM + gold context")
        summaries["LLM + gold context"] = _summarise(llm_res)
        predictions["LLM + gold context"] = [
            {"question": q, "gold": g, "pred": p, "source": s}
            for q, g, p, s in zip(questions, golds, llm_preds, sources)
        ]

        if args.llm_judge:
            print("\n[Step 4b] LLM-as-judge on the LLM+gold-context predictions ...")
            judge_res = llm_as_judge(llm_preds, golds, questions, llm)
            summaries["LLM + gold context"]["LLM_Judge"] = judge_res
            print(f"  LLM-as-judge accuracy: {judge_res['accuracy']:.4f} "
                  f"({judge_res['correct']}/{judge_res['n']})")

    # Full RAG pipeline.
    if llm is not None and not args.skip_full_rag:
        print("\n[Step 5] Evaluating full RAG pipeline (retrieval + rerank + LLM) ...")
        try:
            from medqa.retrieval.rag_pipeline import RAGPipeline
            from medqa.data.loader import load_pubmedqa, load_pubmedqa_unlabeled

            pipeline = RAGPipeline(llm=llm)
            if pipeline.vs.count() == 0:
                print("  -> Building vector index (one-off, can take a while) ...")
                pipeline.build_index(load_pubmedqa() + load_pubmedqa_unlabeled())

            def rag_shot(i):
                out = pipeline.query(questions[i], answer_type=answer_types[i])
                chunks = [
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in (out.get("reranked_chunks") or [])
                ]
                return {"pred": out.get("predicted_answer", ""), "chunks": chunks}

            rag_rows = _run_with_checkpoint(
                "Full_RAG", output_dir, args.resume, len(questions), rag_shot
            )
            rag_preds = [r.get("pred", "") for r in rag_rows]
            rag_chunks = [r.get("chunks", []) for r in rag_rows]

            rag_res = evaluate(rag_preds, golds, sources=sources, use_bertscore=True)

            retr = retrieval_metrics(rag_chunks, retrieval_golds, ks=(1, 3, 5, 10))
            print(f"  Retrieval: Recall@1={retr['recall@1']:.3f}  "
                  f"Recall@3={retr['recall@3']:.3f}  Recall@10={retr['recall@10']:.3f}  "
                  f"MRR={retr['mrr']:.3f}")
            print_results(rag_res, model_name="Full RAG (retrieval + LLM)")
            summaries["Full RAG (retrieval + LLM)"] = _summarise(rag_res)
            summaries["Full RAG (retrieval + LLM)"]["Retrieval"] = retr
            predictions["Full RAG (retrieval + LLM)"] = [
                {"question": q, "gold": g, "pred": p, "source": s}
                for q, g, p, s in zip(questions, golds, rag_preds, sources)
            ]
        except Exception as e:
            import traceback
            print(f"  -> Full RAG eval failed: {e}")
            print("  -> Traceback:")
            traceback.print_exc()

    # Final table + save.
    print("\n" + "=" * 84)
    print(" FINAL COMPARISON ".center(84, "="))
    print("=" * 84)
    header = f"{'Model':<30} | {'n':>4} | {'EM':>6} | {'F1':>6} | {'BERT':>6} | {'ROUGE':>6} | {'Y/N':>6}"
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        yn = s.get("YesNo_Acc", 0.0)
        bs = s.get("BERTScore")
        bs_str = f"{bs:.4f}" if isinstance(bs, float) else "  n/a "
        print(f"{name:<30} | {s['n']:>4} | {s['EM']:.4f} | {s['F1']:.4f} | "
              f"{bs_str} | {s['ROUGE-L']:.4f} | {yn:.4f}")
    print("=" * 84)

    _save(output_dir, summaries, predictions, tag=args.tag)


if __name__ == "__main__":
    main()

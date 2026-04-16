"""RAG ablation sweep.

Toggles UMLS query expansion and the cross-encoder reranker (and varies
rerank_top_k) and re-runs end-to-end eval on the same test subset. Reports
EM, F1, BERTScore, ROUGE-L, Recall@k, MRR per configuration.

Outputs (under reports/ by default):
    ablation_results.json         : scalar metrics per configuration
    ablation_table.md             : markdown comparison table
    _ablation_ckpt_<config>.jsonl : per-config resume checkpoint

Usage:
    uv run python scripts/ablation.py             # default: 60 examples
    uv run python scripts/ablation.py --n 100
    uv run python scripts/ablation.py --resume
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from medqa._seed import set_seed
from medqa.config import SEED
from medqa.data.loader import load_all, load_pubmedqa, load_pubmedqa_unlabeled
from medqa.data.preprocessor import split_dataset
from medqa.evaluation.metrics import evaluate, print_results, retrieval_metrics


REPORTS_DIR = Path("reports")
CHECKPOINT_EVERY = 5


def _answer_type_for(record: dict) -> str:
    source = record.get("source", "")
    atype  = record.get("answer_type", "")
    if source == "pubmedqa" or atype in {"yes", "no", "maybe"}:
        return "yesno"
    if source == "bioasq" or atype == "factoid":
        return "factoid"
    return "free"


def _gold_for_metrics(record: dict) -> str:
    """PubMedQA yes/no items compare against final_decision; other sources unchanged."""
    source = record.get("source", "")
    atype  = record.get("answer_type", "")
    if source == "pubmedqa" and atype in {"yes", "no", "maybe"}:
        return atype
    return record.get("answer", "")


def _config_key(use_qe: bool, use_rr: bool, top_k: int) -> str:
    qe = "QE" if use_qe else "noQE"
    rr = f"RR{top_k}" if use_rr else "noRR"
    return f"{qe}_{rr}"


def _config_label(use_qe: bool, use_rr: bool, top_k: int) -> str:
    parts = []
    parts.append("query-exp" if use_qe else "raw-query")
    parts.append(f"rerank@{top_k}" if use_rr else f"dense@{top_k}")
    return " + ".join(parts)


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[row["i"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _run_config(
    pipeline,
    ckpt_path: Path,
    resume: bool,
    questions: list,
    answer_types: list,
) -> list:
    """Run pipeline.query over all questions, checkpointing after every call."""
    n = len(questions)
    done = _load_checkpoint(ckpt_path) if resume else {}
    if done:
        print(f"    [resume] {len(done)}/{n} cached at {ckpt_path.name}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(ckpt_path, "a", encoding="utf-8")
    try:
        for i in range(n):
            if i in done:
                continue
            try:
                out = pipeline.query(questions[i], answer_type=answer_types[i])
                chunks = [
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in (out.get("reranked_chunks") or [])
                ]
                row = {"i": i, "pred": out.get("predicted_answer", ""), "chunks": chunks}
            except Exception as e:
                print(f"    [warn] sample {i} failed: {e}")
                row = {"i": i, "pred": "", "chunks": [], "error": str(e)}
            done[i] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % CHECKPOINT_EVERY == 0:
                f.flush()
                print(f"    ... {i + 1}/{n}")
    finally:
        f.close()
    return [done[i] for i in range(n)]


def _write_markdown_table(results: dict, out_path: Path) -> None:
    lines = [
        "# RAG Ablation Results",
        "",
        "`QE` = UMLS query expansion, `RR@k` = cross-encoder reranker keeping "
        "top-k, `dense@k` = dense-retrieval only (no reranker).",
        "",
        "| Configuration | n | EM | F1 | BERTScore | ROUGE-L | Recall@1 | Recall@3 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in results.items():
        bs = s.get("BERTScore")
        bs_str = f"{bs:.4f}" if isinstance(bs, (int, float)) else "n/a"
        retr = s.get("Retrieval", {})
        lines.append(
            f"| {label} | {s.get('n', 0)} | "
            f"{s.get('EM', 0.0):.4f} | {s.get('F1', 0.0):.4f} | "
            f"{bs_str} | {s.get('ROUGE-L', 0.0):.4f} | "
            f"{retr.get('recall@1', 0.0):.3f} | {retr.get('recall@3', 0.0):.3f} | "
            f"{retr.get('mrr', 0.0):.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarise(res: dict) -> dict:
    return {
        "n":         res.get("n_samples", 0),
        "EM":        res.get("mean_em", 0.0),
        "F1":        res.get("mean_f1", 0.0),
        "BERTScore": res.get("bertscore_f1"),
        "ROUGE-L":   res.get("rouge_l", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ablation sweep.")
    parser.add_argument("--n", type=int, default=60,
                        help="Number of test examples per configuration (default: 60).")
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 3, 5],
                        help="rerank_top_k values to sweep (default: 1 3 5).")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore (faster; drops one column).")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse per-configuration JSONL checkpoints.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR,
                        help="Where to write ablation_results.json + checkpoints.")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed()

    # Data
    print("[Step 1] Loading and splitting test data ...")
    _, test = split_dataset(load_all())
    test_subset  = test[:args.n]
    questions    = [r["question"]              for r in test_subset]
    golds        = [_gold_for_metrics(r)       for r in test_subset]
    retrieval_golds = [r.get("answer", "")     for r in test_subset]
    sources      = [r.get("source", "unknown") for r in test_subset]
    answer_types = [_answer_type_for(r)        for r in test_subset]
    print(f"  -> {len(test_subset)} examples, sources={dict((s, sources.count(s)) for s in set(sources))}")

    # Shared LLM + vector store
    print("\n[Step 2] Loading LLM backend (shared across all configs) ...")
    from medqa.models.llm_qa import LocalLLM
    from medqa.retrieval.rag_pipeline import RAGPipeline

    llm = LocalLLM()
    llm.load()

    base = RAGPipeline(llm=llm)
    if base.vs.count() == 0:
        print("  -> Vector index empty; building (one-off) ...")
        base.build_index(load_pubmedqa() + load_pubmedqa_unlabeled())
    shared_vs = base.vs
    shared_reranker = base.reranker

    # Sweep
    configs = list(itertools.product([False, True], [False, True], args.top_ks))

    results: dict = {}
    print(f"\n[Step 3] Running {len(configs)} configurations over {len(questions)} examples each ...")

    for idx, (use_qe, use_rr, top_k) in enumerate(configs, start=1):
        key = _config_key(use_qe, use_rr, top_k)
        label = _config_label(use_qe, use_rr, top_k)
        print(f"\n  [{idx}/{len(configs)}] {label}  (key={key})")

        pipeline = RAGPipeline(
            llm=llm,
            vector_store=shared_vs,
            reranker=shared_reranker,
            rerank_k=top_k,
            use_query_expansion=use_qe,
            use_reranker=use_rr,
        )

        ckpt = output_dir / f"_ablation_ckpt_{key}.jsonl"
        rows = _run_config(pipeline, ckpt, args.resume, questions, answer_types)

        preds  = [r.get("pred", "")    for r in rows]
        chunks = [r.get("chunks", []) for r in rows]

        res = evaluate(preds, golds, sources=sources,
                       use_bertscore=not args.no_bertscore)
        print_results(res, model_name=f"RAG [{label}]")

        ks = tuple(k for k in (1, 3, 5, 10) if k <= top_k)
        retr = retrieval_metrics(chunks, retrieval_golds, ks=ks) if ks else {}
        if retr:
            print(f"    Retrieval: " + "  ".join(
                f"R@{k}={retr[f'recall@{k}']:.3f}" for k in ks
            ) + f"  MRR={retr['mrr']:.3f}")

        summary = _summarise(res)
        summary["Retrieval"] = retr
        summary["config"] = {
            "use_query_expansion": use_qe,
            "use_reranker":        use_rr,
            "rerank_top_k":        top_k,
        }
        results[label] = summary

        with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        _write_markdown_table(results, output_dir / "ablation_table.md")

    # Summary
    print("\n" + "=" * 96)
    print(" ABLATION SUMMARY ".center(96, "="))
    print("=" * 96)
    header = f"{'Configuration':<36} | {'n':>4} | {'EM':>6} | {'F1':>6} | {'BERT':>6} | {'ROUGE':>6} | {'R@1':>5}"
    print(header)
    print("-" * len(header))
    for label, s in results.items():
        bs = s.get("BERTScore")
        bs_str = f"{bs:.4f}" if isinstance(bs, (int, float)) else " n/a "
        r1 = s.get("Retrieval", {}).get("recall@1", 0.0)
        print(f"{label:<36} | {s['n']:>4} | {s['EM']:.4f} | {s['F1']:.4f} | "
              f"{bs_str} | {s['ROUGE-L']:.4f} | {r1:.3f}")
    print("=" * 96)
    print(f"\n[Saved] {output_dir / 'ablation_results.json'}")
    print(f"[Saved] {output_dir / 'ablation_table.md'}")


if __name__ == "__main__":
    main()

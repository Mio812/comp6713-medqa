"""Fine-tune PubMedBERT on the combined training split.

GPU, ~30 min on a 12-16 GB card. Output: checkpoints/bert_qa/.

Usage:
    uv run python scripts/finetune_bert.py
    uv run python scripts/finetune_bert.py --use-best-params
    uv run python scripts/finetune_bert.py --params-file path/to/file.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medqa.config import BERT_FINETUNE, ROOT_DIR
from medqa.data.loader import load_all
from medqa.data.preprocessor import split_dataset
from medqa.models.bert_qa import PubMedBERTQA


DEFAULT_HP_RESULTS = ROOT_DIR / "reports" / "hp_search_results.json"


def _load_best_params(path: Path) -> dict:
    """Extract best_params from the HP-search results JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run Task 4a (BERT HP search) first, or drop "
            f"--use-best-params to fine-tune with the config defaults."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best_params")
    if not isinstance(best, dict) or not best:
        raise ValueError(
            f"{path} exists but has no usable best_params."
        )
    return best


def _format_diff(best: dict, defaults: dict) -> str:
    """Pretty-print the diff between best_params and the config defaults."""
    lines = []
    lines.append(f"  {'param':<32} {'best':<18} {'default':<18}")
    lines.append(f"  {'-' * 32} {'-' * 18} {'-' * 18}")
    for k in sorted(best.keys()):
        b = best[k]
        d = defaults.get(k, "-")
        marker = "" if b == d else "  *"
        b_str = f"{b:.6g}" if isinstance(b, float) else str(b)
        d_str = f"{d:.6g}" if isinstance(d, float) else str(d)
        lines.append(f"  {k:<32} {b_str:<18} {d_str:<18}{marker}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PubMedBERT on the combined training set.")
    parser.add_argument(
        "--use-best-params", action="store_true",
        help=f"Load best_params from {DEFAULT_HP_RESULTS.relative_to(ROOT_DIR)} "
             f"and apply them as overrides on top of BERT_FINETUNE.",
    )
    parser.add_argument(
        "--params-file", type=Path, default=None,
        help="Read best_params from this JSON file instead of the default HP-search output.",
    )
    args = parser.parse_args()

    print("[bert] loading data and splitting ...")
    train, test = split_dataset(load_all())
    print(f"[bert] train={len(train)}  test={len(test)}")

    overrides: dict | None = None
    output_dir: str | None = None
    if args.use_best_params or args.params_file:
        path = args.params_file or DEFAULT_HP_RESULTS
        best = _load_best_params(path)
        print(f"[bert] applying best_params from {path}:")
        print(_format_diff(best, BERT_FINETUNE))
        overrides = best
        output_dir = str(ROOT_DIR / "checkpoints" / "bert_qa_hp")
        print(f"[bert] output_dir = {output_dir}  (separate from default)")
    else:
        output_dir = str(ROOT_DIR / "checkpoints" / "bert_qa")
        print(f"[bert] output_dir = {output_dir}  (default params)")

    print("[bert] fine-tuning (~30 min) ...")
    model = PubMedBERTQA()
    metrics = model.fine_tune(train, test, overrides=overrides, output_dir=output_dir)

    print("[bert] done -- checkpoint saved.")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"[bert]   {k} = {v:.4f}")


if __name__ == "__main__":
    main()

"""Optuna hyperparameter search for PubMedBERT fine-tuning.

Runs many short training jobs over a sub-sampled training set, letting
Optuna's TPE sampler propose hyperparameters and minimise eval_loss on the
validation split. MedianPruner kills unpromising trials early.

Usage:
    uv run python scripts/hp_search.py
    uv run python scripts/hp_search.py --trials 30 --train 3000
    uv run python scripts/hp_search.py --resume
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from medqa.config import BERT_FINETUNE, ROOT_DIR, SEED
from medqa.data.loader import load_all
from medqa.data.preprocessor import split_dataset
from medqa.models.bert_qa import PubMedBERTQA


REPORTS_DIR = ROOT_DIR / "reports"
TRIALS_DIR  = ROOT_DIR / "checkpoints" / "hp_search"


def _build_search_space(trial) -> dict:
    """Suggest hyperparameters for this trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16]
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "loss_type":       "focal_ls",
        "focal_gamma":     trial.suggest_float("focal_gamma", 0.0, 3.0),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
    }


def _objective_factory(train_records: list, eval_records: list):
    """Return an Optuna objective closure over the given data splits."""
    def objective(trial) -> float:
        hp = _build_search_space(trial)
        trial_dir = TRIALS_DIR / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = PubMedBERTQA()
        try:
            metrics = model.fine_tune(
                train_records=train_records,
                eval_records=eval_records,
                overrides=hp,
                output_dir=str(trial_dir),
            )
        except Exception as e:
            import optuna
            print(f"  [trial {trial.number}] failed: {e}")
            raise optuna.TrialPruned() from e

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            import optuna
            raise optuna.TrialPruned("No eval_loss reported")

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("train_loss", metrics.get("train_loss"))
        return float(eval_loss)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HP search for PubMedBERT.")
    parser.add_argument("--trials", type=int, default=15,
                        help="Number of Optuna trials (default: 15).")
    parser.add_argument("--train", type=int, default=1000,
                        help="Sub-sample size of the training set per trial (default: 1000).")
    parser.add_argument("--eval", type=int, default=200,
                        help="Sub-sample size of the eval set per trial (default: 200).")
    parser.add_argument("--study-name", type=str, default="pubmedbert_hp",
                        help="Optuna study name (also used for the SQLite storage).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study from SQLite rather than restarting.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Optional wall-clock timeout in seconds.")
    args = parser.parse_args()

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    print("[Step 1] Loading and splitting data ...")
    train, test = split_dataset(load_all())
    random.Random(SEED).shuffle(train)
    random.Random(SEED).shuffle(test)
    train_sub = train[: args.train]
    eval_sub  = test[: args.eval]
    print(f"  -> train subset: {len(train_sub)}   eval subset: {len(eval_sub)}")

    storage_url = f"sqlite:///{(TRIALS_DIR / 'optuna.sqlite3').as_posix()}"
    print(f"\n[Step 2] Study storage: {storage_url}")

    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    print(f"\n[Step 3] Running up to {args.trials} trials ...")
    objective = _objective_factory(train_sub, eval_sub)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   show_progress_bar=False, gc_after_trial=True)

    best = study.best_trial
    print("\n" + "=" * 70)
    print(" HP SEARCH RESULTS ".center(70, "="))
    print("=" * 70)
    print(f"  Completed trials : {len(study.trials)}")
    print(f"  Best trial       : #{best.number}")
    print(f"  Best eval_loss   : {best.value:.4f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k:<32} = {v:.6g}")
        else:
            print(f"    {k:<32} = {v}")

    print("\n  vs. current defaults in BERT_FINETUNE:")
    for k, v in best.params.items():
        default = BERT_FINETUNE.get(k, "-")
        print(f"    {k:<32}  best={v!r:<12}  default={default!r}")

    results_path = REPORTS_DIR / "hp_search_results.json"
    trials_dump = [
        {
            "number":    t.number,
            "state":     t.state.name,
            "value":     t.value,
            "params":    t.params,
            "user_attrs": t.user_attrs,
        }
        for t in study.trials
    ]
    payload = {
        "study_name":   args.study_name,
        "storage":      storage_url,
        "n_trials":     len(study.trials),
        "best_trial":   best.number,
        "best_value":   best.value,
        "best_params":  best.params,
        "defaults":     {k: BERT_FINETUNE.get(k) for k in best.params},
        "trials":       trials_dump,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[Saved] Full trial history -> {results_path}")


if __name__ == "__main__":
    main()

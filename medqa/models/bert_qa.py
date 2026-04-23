"""Fine-tuned PubMedBERT for extractive QA (SQuAD-style span prediction)."""

from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import Dataset

from medqa._log import get_logger
from medqa._seed import set_seed
from medqa.config import BERT_MODEL, BERT_FINETUNE, SEED
from medqa.data.preprocessor import clean_text, truncate_context

log = get_logger("bert")


class PubMedBERTQA:
    """Wrapper around a fine-tuned PubMedBERT extractive QA model.

    Usage:
        model = PubMedBERTQA()
        model.fine_tune(train_records)
        answer = model.predict("What is the treatment for Type 2 diabetes?", context)
    """

    def __init__(self, model_name: str = BERT_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.qa_pipeline = None
        self.output_dir = Path(BERT_FINETUNE["output_dir"])

    def _prepare_dataset(self, records: list) -> Dataset:
        """Convert raw records into token-level features for BERT QA.

        MedQA-USMLE MCQ records are filtered out because their context is a
        pipe-separated option string with no span to extract.
        """
        mcq_skipped = sum(1 for r in records if r.get("answer_type") == "mcq")
        if mcq_skipped:
            log.info("Skipping %d MCQ records (incompatible with span prediction)", mcq_skipped)
        records = [r for r in records if r.get("answer_type") != "mcq"]

        processed = []
        missing_span = 0
        for r in records:
            question = clean_text(str(r.get("question", ""))).strip()
            context  = clean_text(str(r.get("context", ""))).strip()
            answer   = clean_text(str(r.get("answer", ""))).strip()

            if len(question) < 2: question = "Medical question placeholder."
            if len(context) < 2:  context  = "Medical context placeholder."

            context = truncate_context(context, question, self.tokenizer)

            answer_start = context.lower().find(answer.lower()[:50]) if answer else -1
            if answer_start == -1:
                missing_span += 1

            processed.append({
                "question": question,
                "context": context,
                "answers": {"text": [answer], "answer_start": [answer_start]},
            })

        if missing_span:
            pct = missing_span / len(records) * 100 if records else 0.0
            log.info(
                "%d/%d (%.1f%%) records have no extractable span; these train as unanswerable (CLS).",
                missing_span, len(records), pct,
            )

        dataset = Dataset.from_list(processed)

        def tokenize_fn(examples):
            qs   = [str(q) if len(str(q)) > 1 else "Question placeholder" for q in examples["question"]]
            ctxs = [str(c) if len(str(c)) > 1 else "Context placeholder"  for c in examples["context"]]

            tokenized_examples = self.tokenizer(
                qs,
                ctxs,
                truncation="longest_first",
                max_length=512,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized_examples.pop("offset_mapping")
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
                sequence_ids = tokenized_examples.sequence_ids(i)
                sample_index = sample_mapping[i]
                answers = examples["answers"][sample_index]

                if len(answers["answer_start"]) == 0 or answers["answer_start"][0] == -1:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    token_start_index = 0
                    while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                        token_start_index += 1
                    token_end_index = len(input_ids) - 1
                    while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    # Boundary safety check: fall back to CLS if span is out of window
                    if (token_start_index >= len(offsets) or token_end_index < 0 or
                            not (offsets[token_start_index][0] <= start_char and
                                 offsets[token_end_index][1] >= end_char)):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        curr_start = token_start_index
                        while curr_start < len(offsets) and offsets[curr_start][0] <= start_char:
                            curr_start += 1
                        tokenized_examples["start_positions"].append(curr_start - 1)

                        curr_end = token_end_index
                        while curr_end >= 0 and offsets[curr_end][1] >= end_char:
                            curr_end -= 1
                        tokenized_examples["end_positions"].append(curr_end + 1)

            return tokenized_examples

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing and aligning answers",
        )
        return tokenized

    def fine_tune(
        self,
        train_records: list,
        eval_records=None,
        overrides: dict | None = None,
        output_dir: str | None = None,
    ) -> dict:
        """Fine-tune PubMedBERT on train_records.

        Args:
            train_records : training split from loader.py
            eval_records  : optional validation split
            overrides     : optional per-run overrides for BERT_FINETUNE keys.
            output_dir    : optional override for checkpoint path.

        Returns:
            Metrics dict with train_loss and (if eval_records given) eval_loss.
        """
        set_seed(SEED)

        # Resolve hyperparameters: BERT_FINETUNE defaults + per-call overrides
        hp = {**BERT_FINETUNE, **(overrides or {})}
        out_dir = Path(output_dir) if output_dir else Path(hp["output_dir"])

        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        train_dataset = self._prepare_dataset(train_records)
        eval_dataset  = self._prepare_dataset(eval_records) if eval_records else None

        args = TrainingArguments(
            output_dir=str(out_dir),
            learning_rate=hp["learning_rate"],
            num_train_epochs=hp["num_train_epochs"],
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            per_device_eval_batch_size=hp["per_device_eval_batch_size"],
            warmup_ratio=hp.get("warmup_ratio", 0.06),
            weight_decay=hp["weight_decay"],
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            logging_dir=str(out_dir / "logs"),
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=SEED,
            data_seed=SEED,
        )

        loss_type = hp.get("loss_type", "ce")
        if loss_type == "focal_ls":
            from medqa.training.custom_loss import (
                FocalLabelSmoothingLoss, QASpanTrainer,
            )
            loss_fn = FocalLabelSmoothingLoss(
                gamma=hp.get("focal_gamma", 2.0),
                label_smoothing=hp.get("label_smoothing", 0.1),
            )
            log.info(
                "Using focal_ls loss (gamma=%.2f, label_smoothing=%.2f)",
                hp.get("focal_gamma", 2.0),
                hp.get("label_smoothing", 0.1),
            )
            trainer = QASpanTrainer(
                model=self.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DefaultDataCollator(),
                loss_fn=loss_fn,
            )
        elif loss_type == "ce":
            log.info("Using plain cross-entropy loss (HF Trainer default)")
            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DefaultDataCollator(),
            )
        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type!r}. Expected 'ce' or 'focal_ls'."
            )

        log.info("Fine-tuning on %d examples (out_dir=%s) ...", len(train_records), out_dir)
        train_result = trainer.train()
        trainer.save_model(str(out_dir))
        self.tokenizer.save_pretrained(str(out_dir))
        log.info("Model saved to %s", out_dir)

        self.output_dir = out_dir
        self._build_pipeline()

        metrics = {"train_loss": float(train_result.training_loss)}
        if eval_dataset:
            eval_metrics = trainer.evaluate()
            metrics.update({k: float(v) for k, v in eval_metrics.items()
                            if isinstance(v, (int, float))})
        return metrics

    def _build_pipeline(self) -> None:
        """Load the (fine-tuned) model+tokenizer from disk and set eval mode."""
        model_path = str(self.output_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def load(self) -> bool:
        """Load fine-tuned model from disk. Returns True if checkpoint exists."""
        if not self.output_dir.exists():
            return False
        self._build_pipeline()
        log.info("Loaded fine-tuned model from %s", self.output_dir)
        return True

    def predict(self, question: str, context: str) -> dict:
        """Extract an answer span via a manual forward pass."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call fine_tune() or load() first.")

        question = clean_text(question)
        context  = truncate_context(clean_text(context), question, self.tokenizer)

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end   = torch.argmax(outputs.end_logits)

        start_prob = torch.max(torch.softmax(outputs.start_logits, dim=-1)).item()
        end_prob   = torch.max(torch.softmax(outputs.end_logits,   dim=-1)).item()
        confidence_score = (start_prob + end_prob) / 2

        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        answer = self.tokenizer.convert_tokens_to_string(
            all_tokens[answer_start : answer_end + 1]
        )
        answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()

        return {
            "predicted_answer": answer if answer else "No answer found",
            "score": confidence_score,
        }

    def batch_predict(self, questions: list, contexts: list) -> list:
        """Run predict() over paired lists of questions and contexts."""
        return [self.predict(q, c) for q, c in zip(questions, contexts)]

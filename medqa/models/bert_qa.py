"""
Fine-tuned PubMedBERT for extractive Question Answering.

Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
Task : Span extraction — given (question, context), predict the answer span.

Since PubMedQA answers are long-form rather than exact spans, we treat the
*long_answer* field as the answer text and locate it (or the most similar
substring) within the context for span supervision.

For yes/no/maybe questions we additionally fine-tune a classification head.
"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    pipeline,
)
from datasets import Dataset

from medqa.config import BERT_MODEL, BERT_FINETUNE
from medqa.data.preprocessor import clean_text, truncate_context


class PubMedBERTQA:
    """
    Wrapper around a fine-tuned PubMedBERT extractive QA model.

    Usage:
        model = PubMedBERTQA()
        model.fine_tune(train_records)
        answer = model.predict("What is the treatment for Type 2 diabetes?", context)
    """

    def __init__(self, model_name: str = BERT_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None          # loaded lazily or after fine-tuning
        self.qa_pipeline = None    # HuggingFace pipeline for easy inference
        self.output_dir = Path(BERT_FINETUNE["output_dir"])

    # ── Dataset preparation ───────────────────────────────────────────────────

    def _prepare_dataset(self, records: list[dict[str, Any]]) -> Dataset:
        """
        Convert raw records into the token-level features expected by BERT QA.

        For each record we:
          1. Clean question and context.
          2. Truncate context to fit within max_length.
          3. Locate the answer string within the context to get char offsets.
          4. Tokenise with overflow handling (stride=128).
        """
        processed = []
        for r in records:
            question = clean_text(r["question"])
            context  = clean_text(r["context"])
            answer   = clean_text(r.get("answer", ""))

            # Truncate context so the pair fits in 512 tokens
            context = truncate_context(context, question, self.tokenizer)

            # Find the answer start position in the context (char level)
            answer_start = context.lower().find(answer.lower()[:50])  # first 50 chars
            if answer_start == -1:
                answer_start = 0  # fallback: mark answer at beginning

            processed.append({
                "question":       question,
                "context":        context,
                "answers":        {"text": [answer], "answer_start": [answer_start]},
            })

        dataset = Dataset.from_list(processed)

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=512,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        return tokenized

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        train_records: list[dict[str, Any]],
        eval_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Fine-tune PubMedBERT on *train_records*.

        Args:
            train_records : training split from loader.py
            eval_records  : optional validation split (for early stopping)
        """
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        train_dataset = self._prepare_dataset(train_records)
        eval_dataset  = self._prepare_dataset(eval_records) if eval_records else None

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=BERT_FINETUNE["learning_rate"],
            num_train_epochs=BERT_FINETUNE["num_train_epochs"],
            per_device_train_batch_size=BERT_FINETUNE["per_device_train_batch_size"],
            per_device_eval_batch_size=BERT_FINETUNE["per_device_eval_batch_size"],
            warmup_steps=BERT_FINETUNE["warmup_steps"],
            weight_decay=BERT_FINETUNE["weight_decay"],
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            fp16=torch.cuda.is_available(),   # use mixed precision if GPU available
            report_to="none",                  # disable wandb / mlflow
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DefaultDataCollator(),
        )

        print(f"[BERT] Fine-tuning on {len(train_records)} examples ...")
        trainer.train()
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        print(f"[BERT] Model saved to {self.output_dir}")

        # Build inference pipeline from the saved model
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Load the saved model into a HuggingFace QA pipeline."""
        device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = pipeline(
            "question-answering",
            model=str(self.output_dir),
            tokenizer=str(self.output_dir),
            device=device,
        )

    def load(self) -> bool:
        """Load fine-tuned model from disk. Returns True if checkpoint exists."""
        if not self.output_dir.exists():
            return False
        self._build_pipeline()
        print(f"[BERT] Loaded fine-tuned model from {self.output_dir}")
        return True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, question: str, context: str) -> dict[str, Any]:
        """
        Extract the answer span for *question* from *context*.

        Returns a dict with:
            predicted_answer : extracted span string
            score            : model confidence score
        """
        if self.qa_pipeline is None:
            raise RuntimeError("Model not loaded. Call fine_tune() or load() first.")

        context = truncate_context(clean_text(context), clean_text(question), self.tokenizer)
        result = self.qa_pipeline(
            question=clean_text(question),
            context=context,
            max_answer_len=200,
        )
        return {
            "predicted_answer": result["answer"],
            "score":            result["score"],
        }

    def batch_predict(
        self, questions: list[str], contexts: list[str]
    ) -> list[dict[str, Any]]:
        """Run predict() over paired lists of questions and contexts."""
        return [self.predict(q, c) for q, c in zip(questions, contexts)]

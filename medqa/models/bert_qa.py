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

    # ── Dataset preparation ───────────────────────────────────────────────────

    def _prepare_dataset(self, records: list[dict[str, Any]]) -> Dataset:
        """
        Convert raw records into token-level features for BERT QA.
        Includes robust cleaning and boundary checks to prevent truncation errors.
        """
        processed = []
        for r in records:
            # 1. Ensure all fields are strings and clean them
            question = clean_text(str(r.get("question", ""))).strip()
            context  = clean_text(str(r.get("context", ""))).strip()
            answer   = clean_text(str(r.get("answer", ""))).strip()

            # 2. Strict minimum length check for the tokenizer's logic
            if len(question) < 2: question = "Medical question placeholder."
            if len(context) < 2:  context  = "Medical context placeholder."

            # 3. Truncate context to fit within the 512 token limit
            context = truncate_context(context, question, self.tokenizer)

            # Locate answer start position (char-level)
            answer_start = context.lower().find(answer.lower()[:50])
            if answer_start == -1:
                answer_start = 0

            processed.append({
                "question": question,
                "context": context,
                "answers": {"text": [answer], "answer_start": [answer_start]},
            })

        dataset = Dataset.from_list(processed)

        def tokenize_fn(examples):
            # 1. 依然保留安全清理逻辑
            qs = [str(q) if len(str(q)) > 1 else "Question placeholder" for q in examples["question"]]
            ctxs = [str(c) if len(str(c)) > 1 else "Context placeholder" for c in examples["context"]]

            # 2. 修改核心：将 truncation 策略改为 'longest_first'
            tokenized_examples = self.tokenizer(
                qs,
                ctxs,
                truncation="longest_first", # 改为这个，处理“巨型问题”
                max_length=512,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # --- 以下对齐逻辑保持不变 ---
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
                    
                    # 寻找 context 的起始和结束 token
                    token_start_index = 0
                    while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                        token_start_index += 1
                    token_end_index = len(input_ids) - 1
                    while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                        token_end_index -= 1
                    
                    # 边界安全检查
                    if (token_start_index >= len(offsets) or token_end_index < 0 or 
                        not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char)):
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

        # Map the function and drop raw text columns
        tokenized = dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Tokenizing and aligning answers"
        )
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
            eval_strategy="epoch" if eval_dataset else "no", 
            save_strategy="epoch",
            per_device_train_batch_size=BERT_FINETUNE["per_device_train_batch_size"],
            per_device_eval_batch_size=BERT_FINETUNE["per_device_eval_batch_size"],
            warmup_steps=BERT_FINETUNE["warmup_steps"],
            weight_decay=BERT_FINETUNE["weight_decay"],
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
        """Load the saved model using the explicit Pipeline class."""
        from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering
        
        # Explicitly load model and tokenizer
        model = AutoModelForQuestionAnswering.from_pretrained(str(self.output_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(self.output_dir))
        
        device = 0 if torch.cuda.is_available() else -1
        
        # Manually initialize the pipeline
        self.qa_pipeline = QuestionAnsweringPipeline(
            model=model, 
            tokenizer=tokenizer, 
            device=device
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

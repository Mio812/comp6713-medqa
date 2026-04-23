"""Generative QA using Qwen2.5-14B-Instruct, running locally with 4-bit quantisation.

Uses answer-type-conditioned prompting: the caller tells us whether the
expected answer is yes/no, factoid, or free-form, and the prompt is
specialised to produce the short format the benchmark expects.
"""

from typing import Any, Literal

from medqa._log import get_logger
from medqa.config import LLM_MODEL
from medqa.data.preprocessor import clean_text

log = get_logger("llm")

AnswerType = Literal["yesno", "factoid", "free", "auto"]

_SYS_BASE = (
    "You are a knowledgeable medical assistant. "
    "Answer the question using only the provided context. "
    "If the context does not contain enough information, say 'maybe'."
)

_SYS_YESNO = _SYS_BASE + (
    " Your entire reply must be exactly one word chosen from {yes, no, maybe}."
    " Do not write any other word, punctuation, or explanation."
)

_SYS_FACTOID = _SYS_BASE + (
    " Reply with the shortest possible noun phrase (1-8 words) copied verbatim "
    "from the context. Do not write a full sentence and do not include "
    "explanations, punctuation at the end, or prefixes like 'Answer:'."
)

_SYS_FREE = _SYS_BASE + " Be concise: at most two sentences."


def _detect_answer_type(question: str) -> str:
    """Cheap heuristic to guess the expected answer format from the question."""
    q = question.strip().lower()
    yesno_words = {
        "does", "do", "is", "are", "was", "were",
        "can", "could", "should", "has", "have", "did",
    }
    first = q.split()[0] if q else ""
    if first in yesno_words:
        return "yesno"
    factoid_words = {"what", "which", "who", "where", "when", "name"}
    if first in factoid_words:
        return "factoid"
    return "free"


def _build_prompt(
    question: str,
    context: str,
    answer_type: str = "auto",
) -> list:
    """Format question + context as a chat message list, specialised by answer type."""
    if answer_type == "auto":
        answer_type = _detect_answer_type(question)

    system = {
        "yesno":   _SYS_YESNO,
        "factoid": _SYS_FACTOID,
        "free":    _SYS_FREE,
    }.get(answer_type, _SYS_FREE)

    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


class LocalLLM:
    """Runs Qwen2.5-14B-Instruct locally with 4-bit NF4 quantisation (~9 GB VRAM).

    Requires a CUDA-capable GPU with at least 10 GB VRAM.
    """

    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.pipeline = None

    def load(self, load_in_4bit: bool = True) -> None:
        """Load the model into memory with optional 4-bit quantisation."""
        import gc
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig,
        )

        # Free any VRAM held by previous loads / other notebooks in this kernel.
        # Qwen-14B NF4 needs roughly 9 GB, and notebooks often have the reranker
        # and dense-retrieval embedder resident, so squeezing 14B in requires
        # a clean slate.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        log.info("Loading %s ...", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Reserve ~1 GiB of GPU headroom for activations and KV cache, then
            # let accelerate spill anything that doesn't fit to CPU RAM. This
            # keeps loading robust on 16 GB cards where the reranker and
            # embedder are already resident.
            total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_budget_gib = max(4, int(total_gib) - 1)  # leave ~1 GiB headroom

            load_kwargs = dict(
                quantization_config=quant_config,
                device_map="auto",
                max_memory={0: f"{gpu_budget_gib}GiB", "cpu": "64GiB"},
                low_cpu_mem_usage=True,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            except ValueError as e:
                # Fallback: accelerate refused to dispatch modules across
                # CPU/GPU for 4-bit. Force a fresh attempt with a tighter GPU
                # budget so more layers move to CPU (slower but will fit).
                if "dispatched on the CPU" not in str(e):
                    raise
                log.warning("GPU+CPU dispatch refused at %d GiB; retrying with smaller GPU budget.",
                            gpu_budget_gib)
                load_kwargs["max_memory"] = {0: f"{max(4, gpu_budget_gib - 2)}GiB", "cpu": "64GiB"}
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        else:
            log.warning("Running without 4-bit quantisation. May run out of VRAM.")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
        )
        # Clear the model's generation_config.max_length so it doesn't collide
        # with per-call max_new_tokens and emit a warning on every generation.
        gen_cfg = getattr(self.pipeline.model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.max_length = None
        log.info("Model loaded.")

    def unload(self) -> None:
        """Release the model from GPU/CPU memory."""
        import gc
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("Model unloaded.")

    def predict(
        self,
        question: str,
        context: str,
        answer_type: str = "auto",
    ) -> dict:
        """Generate an answer for *question* given *context*.

        Args:
            question    : the user question.
            context     : the passage(s) to condition on.
            answer_type : 'yesno' | 'factoid' | 'free' | 'auto' (default).
                          Controls which prompt template is used so the output
                          format matches what the benchmark expects.
        """
        if self.pipeline is None:
            raise RuntimeError("Call load() before predict().")

        q_clean = clean_text(question)
        c_clean = clean_text(context)
        messages = _build_prompt(q_clean, c_clean, answer_type=answer_type)

        # Cap generation length based on the effective answer type.
        effective = answer_type if answer_type != "auto" else _detect_answer_type(q_clean)
        gen_kwargs = {}
        if effective == "yesno":
            gen_kwargs["max_new_tokens"] = 4
        elif effective == "factoid":
            gen_kwargs["max_new_tokens"] = 32

        output = self.pipeline(messages, **gen_kwargs) if gen_kwargs else self.pipeline(messages)
        generated = output[0]["generated_text"]

        if isinstance(generated, list):
            answer = generated[-1].get("content", "")
        else:
            answer = str(generated).split("Answer:")[-1].strip()

        return {
            "predicted_answer": answer.strip(),
            "model":            self.model_name,
            "answer_type_used": effective,
        }

    def batch_predict(
        self,
        questions: list,
        contexts: list,
        answer_types=None,
        batch_size: int = 4,
    ) -> list:
        """Paired batch prediction, bucketed by answer_type.

new_tokens homogeneous within a
        batch (yesno=4, factoid=32, free=256).
        """
        if self.pipeline is None:
            raise RuntimeError("Call load() before batch_predict().")
        if answer_types is None:
            answer_types = ["auto"] * len(questions)
        assert len(questions) == len(contexts) == len(answer_types)

        resolved = [
            _detect_answer_type(q) if at == "auto" else at
            for q, at in zip(questions, answer_types)
        ]

        results: list[dict | None] = [None] * len(questions)
        buckets: dict[str, list[int]] = {}
        for idx, at in enumerate(resolved):
            buckets.setdefault(at, []).append(idx)

        max_new_by_type = {"yesno": 4, "factoid": 32, "free": 256}

        for at, idxs in buckets.items():
            prompts = [
                _build_prompt(clean_text(questions[i]), clean_text(contexts[i]), answer_type=at)
                for i in idxs
            ]
            gen_kwargs = {"max_new_tokens": max_new_by_type.get(at, 256)}
            outputs = self.pipeline(prompts, batch_size=batch_size, **gen_kwargs)

            for i, out in zip(idxs, outputs):
                generated = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
                if isinstance(generated, list):
                    answer = generated[-1].get("content", "")
                else:
                    answer = str(generated).split("Answer:")[-1].strip()
                results[i] = {
                    "predicted_answer": answer.strip(),
                    "model":            self.model_name,
                    "answer_type_used": at,
                }
        return results

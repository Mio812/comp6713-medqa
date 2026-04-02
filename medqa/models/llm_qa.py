"""
Decoder-based generative QA using Qwen2.5-7B-Instruct (or any OpenAI-compatible API).

This module provides two backends:
  - LocalLLM  : runs the model locally via HuggingFace transformers (GPU recommended)
  - APILLM    : calls any OpenAI-compatible endpoint (DeepSeek, Together, etc.)

Both expose the same predict() interface so the rest of the codebase is
backend-agnostic.
"""

import os
from typing import Any

from medqa.config import LLM_MODEL
from medqa.data.preprocessor import clean_text

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant. "
    "Answer the question based only on the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Be concise and precise."
)

def _build_prompt(question: str, context: str) -> list[dict[str, str]]:
    """Format question + context as a chat message list."""
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── Local HuggingFace backend ─────────────────────────────────────────────────

class LocalLLM:
    """
    Runs Qwen2.5-7B-Instruct (or any HF chat model) locally.
    Requires ~16 GB VRAM for full precision, or ~8 GB with 4-bit quantisation.

    Usage:
        llm = LocalLLM()
        llm.load()
        result = llm.predict("What causes type 2 diabetes?", context)
    """

    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.pipeline = None

    def load(self, load_in_4bit: bool = True) -> None:
        """
        Load the model into memory.

        Args:
            load_in_4bit: use bitsandbytes 4-bit quantisation to reduce VRAM usage.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

        print(f"[LLM] Loading {self.model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
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
            do_sample=False,   # greedy decoding for reproducibility
        )
        print("[LLM] Model loaded.")

    def predict(self, question: str, context: str) -> dict[str, Any]:
        """Generate an answer for *question* given *context*."""
        if self.pipeline is None:
            raise RuntimeError("Call load() before predict().")

        messages = _build_prompt(clean_text(question), clean_text(context))
        output = self.pipeline(messages)
        generated = output[0]["generated_text"]

        # Extract only the assistant's reply
        if isinstance(generated, list):
            answer = generated[-1].get("content", "")
        else:
            answer = str(generated).split("Answer:")[-1].strip()

        return {"predicted_answer": answer, "model": self.model_name}

    def batch_predict(
        self, questions: list[str], contexts: list[str]
    ) -> list[dict[str, Any]]:
        return [self.predict(q, c) for q, c in zip(questions, contexts)]


# ── API backend (OpenAI-compatible) ───────────────────────────────────────────

class APILLM:
    """
    Calls any OpenAI-compatible chat completion endpoint.

    Works with:
      - DeepSeek API  (base_url="https://api.deepseek.com/v1")
      - Together AI, Groq, local Ollama, etc.

    Set OPENAI_API_KEY (or DEEPSEEK_API_KEY) in your .env file.

    Usage:
        llm = APILLM(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
        result = llm.predict("What causes type 2 diabetes?", context)
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def predict(self, question: str, context: str) -> dict[str, Any]:
        """Call the API and return the generated answer."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: uv add openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = _build_prompt(clean_text(question), clean_text(context))

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        answer = response.choices[0].message.content.strip()
        return {"predicted_answer": answer, "model": self.model}

    def batch_predict(
        self, questions: list[str], contexts: list[str]
    ) -> list[dict[str, Any]]:
        return [self.predict(q, c) for q, c in zip(questions, contexts)]

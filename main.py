"""
Gradio demo for the MedQA system.

Launches a web UI where users can:
  - Type a medical question
  - Choose a model backend (Baseline / BERT / RAG)
  - View the answer and the retrieved source chunks

Run with:
    uv run python main.py
Then open http://localhost:7860 in your browser.
"""

import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ── Lazy model cache (loaded once per session) ────────────────────────────────

_models: dict = {}


def _get_baseline():
    if "baseline" not in _models:
        from medqa.models.baseline import TFIDFBaseline
        from medqa.data.loader import load_all
        from medqa.data.preprocessor import split_dataset

        m = TFIDFBaseline()
        if not m.load():
            records, _ = split_dataset(load_all())
            m.fit(records)
        _models["baseline"] = m
    return _models["baseline"]


def _get_bert():
    if "bert" not in _models:
        from medqa.models.bert_qa import PubMedBERTQA
        m = PubMedBERTQA()
        if not m.load():
            return None
        _models["bert"] = m
    return _models["bert"]


def _get_rag():
    if "rag" not in _models:
        from medqa.models.llm_qa import APILLM
        from medqa.retrieval.rag_pipeline import RAGPipeline
        from medqa.data.loader import load_pubmedqa, load_pubmedqa_unlabeled

        llm = APILLM()
        pipeline = RAGPipeline(llm=llm)
        if pipeline.vs.count() == 0:
            records = load_pubmedqa() + load_pubmedqa_unlabeled()
            pipeline.build_index(records)
        _models["rag"] = pipeline
    return _models["rag"]


# ── Core answer function ──────────────────────────────────────────────────────

def answer_question(question: str, mode: str) -> tuple[str, str]:
    """
    Called by Gradio on each submission.

    Returns:
        (answer_text, sources_text)
    """
    if not question.strip():
        return "Please enter a question.", ""

    try:
        if mode == "Baseline (TF-IDF)":
            model = _get_baseline()
            result = model.predict(question)
            answer = result.get("predicted_answer", "")
            sources = f"Retrieval score: {result.get('score', 0):.4f}\n\n{result.get('context', '')[:500]}..."
            return answer, sources

        elif mode == "BERT (Fine-tuned PubMedBERT)":
            bert = _get_bert()
            if bert is None:
                return "BERT model not found. Run fine-tuning first.", ""
            baseline = _get_baseline()
            top = baseline.retrieve(question, top_k=1)
            context = top[0]["context"] if top else ""
            result = bert.predict(question, context)
            return result.get("predicted_answer", ""), context[:500] + "..."

        elif mode == "RAG (Retrieval + Qwen/DeepSeek)":
            pipeline = _get_rag()
            result = pipeline.query(question)
            answer = result.get("predicted_answer", "")
            chunks = result.get("reranked_chunks", [])
            sources_parts = []
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get("rerank_score", chunk.get("score", 0))
                sources_parts.append(
                    f"**[Source {i}]** (score: {score:.4f})\n{chunk['text'][:300]}..."
                )
            sources = "\n\n".join(sources_parts)
            return answer, sources

    except Exception as e:
        return f"Error: {e}", ""

    return "Unknown mode.", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MedQA — Medical Literature QA", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏥 MedQA — Medical Literature Question Answering
            **COMP6713 Group Project** · University of New South Wales

            Ask any medical question. The system retrieves relevant PubMed abstracts
            and generates an evidence-based answer.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                question_box = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What is the first-line treatment for hypertension?",
                    lines=3,
                )
                mode_selector = gr.Radio(
                    choices=[
                        "Baseline (TF-IDF)",
                        "BERT (Fine-tuned PubMedBERT)",
                        "RAG (Retrieval + Qwen/DeepSeek)",
                    ],
                    value="RAG (Retrieval + Qwen/DeepSeek)",
                    label="Model Backend",
                )
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=4):
                answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
                sources_box = gr.Textbox(label="Retrieved Sources", lines=10, interactive=False)

        # Example questions
        gr.Examples(
            examples=[
                ["What are the symptoms of Type 2 diabetes?", "RAG (Retrieval + Qwen/DeepSeek)"],
                ["Is metformin effective for treating insulin resistance?", "Baseline (TF-IDF)"],
                ["What causes myocardial infarction?", "BERT (Fine-tuned PubMedBERT)"],
                ["What is the relationship between obesity and hypertension?", "RAG (Retrieval + Qwen/DeepSeek)"],
            ],
            inputs=[question_box, mode_selector],
        )

        submit_btn.click(
            fn=answer_question,
            inputs=[question_box, mode_selector],
            outputs=[answer_box, sources_box],
        )
        question_box.submit(
            fn=answer_question,
            inputs=[question_box, mode_selector],
            outputs=[answer_box, sources_box],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

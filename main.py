"""Gradio demo for the MedQA system."""

import gradio as gr

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
        from medqa.models.llm_qa import LocalLLM
        from medqa.retrieval.rag_pipeline import RAGPipeline
        from medqa.data.loader import load_pubmedqa, load_pubmedqa_unlabeled

        llm = LocalLLM()
        llm.load()
        pipeline = RAGPipeline(llm=llm)
        if pipeline.vs.count() == 0:
            records = load_pubmedqa() + load_pubmedqa_unlabeled()
            pipeline.build_index(records)
        _models["rag"] = pipeline
    return _models["rag"]


def answer_question(question: str, mode: str) -> tuple[str, str]:
    """Called by Gradio on each submission. Returns (answer, sources)."""
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


CSS = """
.gradio-container, .gradio-container * {
    font-family: "Times New Roman", Times, serif !important;
}
.gradio-container {
    background: #ffffff !important;
    max-width: 880px !important;
    margin: 0 auto !important;
}
/* Headings / labels */
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container label, .gradio-container .label-wrap {
    color: #1e3a8a !important;
}
/* Text inputs and dropdowns */
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: #ffffff !important;
    color: #1e3a8a !important;
    border: 1px solid #93c5fd !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
}
/* Primary button */
.gradio-container button.primary, .gradio-container .primary button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
}
.gradio-container button.primary:hover, .gradio-container .primary button:hover {
    background: #1d4ed8 !important;
}
/* Block containers */
.gradio-container .block, .gradio-container .form {
    background: #ffffff !important;
    border-color: #dbeafe !important;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MedQA") as demo:
        gr.Markdown("## MedQA - Medical Literature QA")

        question_box = gr.Textbox(
            label="Question",
            placeholder="e.g. What is the first-line treatment for hypertension?",
            lines=2,
        )
        mode_selector = gr.Dropdown(
            choices=[
                "RAG (Retrieval + Qwen/DeepSeek)",
                "BERT (Fine-tuned PubMedBERT)",
                "Baseline (TF-IDF)",
            ],
            value="RAG (Retrieval + Qwen/DeepSeek)",
            label="Model",
        )
        submit_btn = gr.Button("Ask", variant="primary")

        answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
        sources_box = gr.Textbox(label="Sources", lines=6, interactive=False)

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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="blue",
            neutral_hue="slate",
            font=["Times New Roman", "Times", "serif"],
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()

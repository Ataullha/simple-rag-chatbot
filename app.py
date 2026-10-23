"""Simple Gradio interface for the NLP RAG chatbot."""

import gradio as gr

from llm import generate_answer
from retrival import retrieve


def chat(message, history):
    documents = retrieve(message, top_k=3)
    answer = generate_answer(message, documents)
    if documents and not answer.startswith(("Please add", "Groq request failed")):
        topics = ", ".join(item["topic"] for item in documents)
        answer += f"\n\n---\n**Retrieved topics:** {topics}"
    return answer


if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="NLP Course Assistant | RAG + Groq",
        description="Ask questions based on the NLP topics in database.json.",
        examples=[
            "What is NLP?",
            "Explain TF-IDF and cosine similarity.",
            "What is the difference between BERT and BART?",
            "How does an LSTM work?",
            "Explain retrieval-augmented generation.",
        ],
    )
    demo.launch()

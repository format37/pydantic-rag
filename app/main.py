#!/usr/bin/env python3
"""Gradio app demonstrating Ollama embedding and generation for RAG."""

import os
import time

import gradio as gr
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")


def get_embedding(text: str) -> tuple[list[float], float]:
    """Get embedding for text using Ollama."""
    start = time.time()
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    embeddings = data.get("embeddings", [[]])[0]
    return embeddings, elapsed


def generate_response(prompt: str) -> tuple[str, float]:
    """Generate response using Ollama."""
    start = time.time()
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    return data.get("response", ""), elapsed


def embed_text(text: str) -> str:
    """Gradio handler for embedding."""
    if not text.strip():
        return "Please enter some text to embed."

    try:
        embedding, elapsed = get_embedding(text)
        preview = embedding[:5]
        return (
            f"Embedding generated in {elapsed:.2f}s\n"
            f"Dimension: {len(embedding)}\n"
            f"First 5 values: {preview}"
        )
    except Exception as e:
        return f"Error: {e}"


def chat(message: str, history: list) -> tuple[list, str]:
    """Gradio handler for chat."""
    if not message.strip():
        return history, ""

    # Add user message
    history.append({"role": "user", "content": message})

    try:
        response, elapsed = generate_response(message)
        history.append({"role": "assistant", "content": f"{response}\n\n_({elapsed:.2f}s)_"})
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {e}"})

    return history, ""


def check_ollama_status() -> str:
    """Check Ollama connection and available models."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return f"✓ Connected to Ollama\nModels: {', '.join(models)}"
    except Exception as e:
        return f"✗ Ollama unavailable: {e}"


# Build Gradio interface
with gr.Blocks(title="Pydantic RAG Demo") as demo:
    gr.Markdown("# Pydantic RAG Demo")
    gr.Markdown("Test Ollama embedding and generation capabilities.")

    with gr.Row():
        status_btn = gr.Button("Check Ollama Status")
        status_output = gr.Textbox(label="Status", interactive=False)
        status_btn.click(check_ollama_status, outputs=status_output)

    with gr.Tab("Embedding"):
        gr.Markdown(f"Generate embeddings using **{EMBED_MODEL}**")
        embed_input = gr.Textbox(
            label="Text to embed",
            placeholder="Enter text to generate embeddings...",
            lines=3,
        )
        embed_btn = gr.Button("Generate Embedding")
        embed_output = gr.Textbox(label="Result", lines=4)
        embed_btn.click(embed_text, inputs=embed_input, outputs=embed_output)

    with gr.Tab("Chat"):
        gr.Markdown(f"Chat with **{CHAT_MODEL}**")
        chatbot = gr.Chatbot(height=400)
        msg_input = gr.Textbox(
            label="Message",
            placeholder="Type your message...",
            show_label=False,
        )
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])


if __name__ == "__main__":
    print(f"Starting Gradio app...")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Embed model: {EMBED_MODEL}")
    print(f"Chat model: {CHAT_MODEL}")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())

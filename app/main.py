#!/usr/bin/env python3
"""Gradio app demonstrating Ollama embedding and generation for RAG."""

import os
import time

import gradio as gr
import httpx

from agent import RAGDeps, create_weaviate_client, get_agent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

# Global Weaviate client
_weaviate_client = None


def get_weaviate_client():
    """Get or create Weaviate client with lazy reconnection."""
    global _weaviate_client
    if _weaviate_client is None:
        try:
            _weaviate_client = create_weaviate_client()
        except Exception:
            _weaviate_client = None
    return _weaviate_client


def close_weaviate_client():
    """Close Weaviate client connection."""
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
        except Exception:
            pass
        _weaviate_client = None


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


async def rag_chat(message: str, history: list) -> tuple[list, str]:
    """Gradio handler for RAG-powered chat."""
    if not message.strip():
        return history, ""

    # Add user message
    history.append({"role": "user", "content": message})

    try:
        # Get Weaviate client
        client = get_weaviate_client()
        if client is None:
            history.append({
                "role": "assistant",
                "content": "Error: Could not connect to Weaviate. Please check the connection."
            })
            return history, ""

        # Check if collection exists
        try:
            collections = client.collections.list_all()
            if "Document" not in [c for c in collections]:
                history.append({
                    "role": "assistant",
                    "content": "Error: Document collection not found. Please ingest documents first."
                })
                return history, ""
        except Exception as e:
            history.append({
                "role": "assistant",
                "content": f"Error checking collections: {e}"
            })
            return history, ""

        # Create dependencies and run agent
        deps = RAGDeps(weaviate_client=client, collection_name="Document")
        agent = get_agent()

        start = time.time()
        result = await agent.run(message, deps=deps)
        elapsed = time.time() - start

        history.append({
            "role": "assistant",
            "content": f"{result.output}\n\n_({elapsed:.2f}s)_"
        })

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })

    return history, ""


def check_ollama_status() -> str:
    """Check Ollama connection and available models."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return f"Ollama: Connected\nModels: {', '.join(models)}"
    except Exception as e:
        return f"Ollama: Unavailable - {e}"


def check_weaviate_status() -> str:
    """Check Weaviate connection and collections."""
    try:
        client = get_weaviate_client()
        if client is None:
            return "Weaviate: Could not connect"

        # Check if ready
        if not client.is_ready():
            return "Weaviate: Not ready"

        # List collections
        collections = client.collections.list_all()
        collection_names = list(collections.keys()) if collections else []

        # Get document count if Document collection exists
        doc_count = 0
        if "Document" in collection_names:
            doc_collection = client.collections.get("Document")
            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count

        return (
            f"Weaviate: Connected\n"
            f"Collections: {', '.join(collection_names) if collection_names else 'None'}\n"
            f"Documents: {doc_count} chunks"
        )
    except Exception as e:
        return f"Weaviate: Error - {e}"


# Build Gradio interface
with gr.Blocks(title="Pydantic RAG Demo") as demo:
    gr.Markdown("# Pydantic RAG Demo")
    gr.Markdown("RAG-powered chat using Pydantic AI, Ollama, and Weaviate hybrid search.")

    with gr.Row():
        with gr.Column(scale=1):
            ollama_btn = gr.Button("Check Ollama")
            ollama_status = gr.Textbox(label="Ollama Status", interactive=False, lines=3)
            ollama_btn.click(check_ollama_status, outputs=ollama_status)

        with gr.Column(scale=1):
            weaviate_btn = gr.Button("Check Weaviate")
            weaviate_status = gr.Textbox(label="Weaviate Status", interactive=False, lines=3)
            weaviate_btn.click(check_weaviate_status, outputs=weaviate_status)

    with gr.Tab("RAG Chat"):
        gr.Markdown(f"Chat with documents using **{CHAT_MODEL}** and hybrid search")
        chatbot = gr.Chatbot(height=400)
        msg_input = gr.Textbox(
            label="Message",
            placeholder="Ask a question about your documents...",
            show_label=False,
        )
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")

        msg_input.submit(rag_chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

    with gr.Tab("Embedding Test"):
        gr.Markdown(f"Generate embeddings using **{EMBED_MODEL}**")
        embed_input = gr.Textbox(
            label="Text to embed",
            placeholder="Enter text to generate embeddings...",
            lines=3,
        )
        embed_btn = gr.Button("Generate Embedding")
        embed_output = gr.Textbox(label="Result", lines=4)
        embed_btn.click(embed_text, inputs=embed_input, outputs=embed_output)


if __name__ == "__main__":
    print(f"Starting Gradio app...")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Weaviate URL: {WEAVIATE_URL}")
    print(f"Embed model: {EMBED_MODEL}")
    print(f"Chat model: {CHAT_MODEL}")

    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    finally:
        close_weaviate_client()

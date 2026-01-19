#!/usr/bin/env python3
"""Gradio app demonstrating Ollama embedding and generation for RAG."""

import os
import time

import gradio as gr
import httpx

from agent import RAGDeps, create_weaviate_client, get_agent, get_multimodal_agent, get_available_names, run_multimodal_with_images

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
CHAT_MODEL_MULTIMODAL = os.getenv("CHAT_MODEL_MULTIMODAL", "llava:34b")
MULTIMODAL_MODE = os.getenv("MULTIMODAL_MODE", "false").lower() == "true"

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


def fetch_document_names() -> list[str]:
    """Fetch available document set names from Weaviate."""
    try:
        client = get_weaviate_client()
        if client and client.is_ready():
            return get_available_names(client, multimodal=MULTIMODAL_MODE)
    except Exception:
        pass
    return []


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


async def rag_chat(
    message: str,
    history: list,
    message_history: list,
    rag_mode: str,
    num_chunks: int,
    chunk_content_size: int,
    name_filter: list[str],
) -> tuple[list, str, list, str]:
    """Gradio handler for RAG-powered chat with conversation memory.

    Args:
        message: User's input message.
        history: Gradio chatbot display history.
        message_history: Pydantic AI message history for conversation memory.
        rag_mode: RAG mode selection ("Auto", "Force", "Disabled").
        num_chunks: Number of chunks to retrieve from vector DB.
        chunk_content_size: Max characters to show per chunk.
        name_filter: List of document set names to filter by (empty = all).

    Returns:
        Tuple of (updated history, cleared input, updated message_history, token info).
    """
    if not message.strip():
        return history, "", message_history, ""

    # Add user message to display history
    history.append({"role": "user", "content": message})

    try:
        # Get Weaviate client (still needed for Auto/Force modes)
        client = get_weaviate_client()
        rag_mode_lower = rag_mode.lower()

        # Determine collection name based on mode
        collection_name = "MultimodalDocument" if MULTIMODAL_MODE else "Document"

        # Only check Weaviate connection for RAG-enabled modes
        if rag_mode_lower != "disabled":
            if client is None:
                history.append({
                    "role": "assistant",
                    "content": "Error: Could not connect to Weaviate. Please check the connection."
                })
                return history, "", message_history, ""

            # Check if collection exists
            try:
                collections = client.collections.list_all()
                if collection_name not in [c for c in collections]:
                    history.append({
                        "role": "assistant",
                        "content": f"Error: {collection_name} collection not found. Please ingest documents first."
                    })
                    return history, "", message_history, ""
            except Exception as e:
                history.append({
                    "role": "assistant",
                    "content": f"Error checking collections: {e}"
                })
                return history, "", message_history, ""

        # Create dependencies with user-configured retrieval settings
        deps = RAGDeps(
            weaviate_client=client,
            collection_name=collection_name,
            multimodal=MULTIMODAL_MODE,
            num_chunks=int(num_chunks),
            chunk_content_size=int(chunk_content_size),
            name_filter=name_filter if name_filter else None,
        )

        # Use the appropriate agent based on mode
        if MULTIMODAL_MODE:
            agent = get_multimodal_agent(rag_mode_lower)

            start = time.time()
            # Use the image injection wrapper for multimodal mode
            output, new_message_history, usage = await run_multimodal_with_images(
                agent,
                message,
                deps=deps,
                message_history=message_history,
            )
            elapsed = time.time() - start
        else:
            agent = get_agent(rag_mode_lower)

            start = time.time()
            result = await agent.run(
                message,
                deps=deps,
                message_history=message_history,
            )
            elapsed = time.time() - start

            # Update message history for conversation memory
            new_message_history = result.all_messages()
            output = result.output
            usage = result.usage()

        # Track token usage
        total_tokens = usage.total_tokens if usage.total_tokens else (
            (usage.request_tokens or 0) + (usage.response_tokens or 0)
        )

        # Build response with timing info
        response_text = f"{output}\n\n_({elapsed:.2f}s)_"

        # Add token warning if approaching limit (8K context for llama3.2)
        token_info = f"Tokens: {total_tokens:,}"
        if total_tokens > 6000:
            response_text += "\n\n⚠️ _Approaching token limit. Consider resetting the conversation._"
            token_info += " ⚠️"

        history.append({
            "role": "assistant",
            "content": response_text
        })

        return history, "", new_message_history, token_info

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })
        return history, "", message_history, ""


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

        # Get item count from the active collection
        active_collection = "MultimodalDocument" if MULTIMODAL_MODE else "Document"
        item_count = 0
        if active_collection in collection_names:
            doc_collection = client.collections.get(active_collection)
            item_count = doc_collection.aggregate.over_all(total_count=True).total_count

        mode_str = "Multimodal" if MULTIMODAL_MODE else "Text-only"
        return (
            f"Weaviate: Connected ({mode_str} mode)\n"
            f"Collections: {', '.join(collection_names) if collection_names else 'None'}\n"
            f"Active: {active_collection} ({item_count} items)"
        )
    except Exception as e:
        return f"Weaviate: Error - {e}"


def refresh_names():
    """Refresh document set names from Weaviate for the CheckboxGroup."""
    names = fetch_document_names()
    return gr.CheckboxGroup(choices=names, value=[])


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
        if MULTIMODAL_MODE:
            gr.Markdown(f"**Mode: Multimodal** (CLIP embeddings + {CHAT_MODEL_MULTIMODAL} VLM) - Chat with documents and images")
        else:
            gr.Markdown(f"**Mode: Text-only** ({CHAT_MODEL} + nomic-embed-text) - Chat with documents")

        # Session state for Pydantic AI message history (per-user session isolation)
        message_history_state = gr.State(value=[])

        # RAG mode selector
        with gr.Row():
            rag_mode = gr.Radio(
                choices=["Auto", "Force", "Disabled"],
                value="Auto",
                label="RAG Mode",
                info="Auto: Agent decides | Force: Always search | Disabled: Plain chat",
            )
            token_display = gr.Textbox(
                label="Token Usage",
                value="",
                interactive=False,
                scale=1,
            )

        # Retrieval settings
        with gr.Accordion("Retrieval Settings", open=False):
            with gr.Row():
                num_chunks_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Chunks",
                    info="How many document chunks to retrieve per search",
                )
                chunk_size_slider = gr.Slider(
                    minimum=100,
                    maximum=4000,
                    value=500,
                    step=100,
                    label="Chunk Display Size (chars)",
                    info="Max characters to show per chunk (increase for larger context models)",
                )

            # Document set filter
            with gr.Row():
                name_filter = gr.CheckboxGroup(
                    choices=[],  # Populated dynamically
                    value=[],    # None selected = search all
                    label="Document Sets",
                    info="Select which document sets to search (empty = all)",
                )
                refresh_names_btn = gr.Button("Refresh List", size="sm", scale=0)

        chatbot = gr.Chatbot(height=400)
        msg_input = gr.Textbox(
            label="Message",
            placeholder="Ask a question about your documents...",
            show_label=False,
        )
        with gr.Row():
            reset_btn = gr.Button("Reset Chat", variant="secondary")

        # Reset clears chatbot display, input, message history, and token display
        def reset_chat():
            return [], "", [], ""

        msg_input.submit(
            rag_chat,
            inputs=[msg_input, chatbot, message_history_state, rag_mode, num_chunks_slider, chunk_size_slider, name_filter],
            outputs=[chatbot, msg_input, message_history_state, token_display],
        )
        reset_btn.click(
            reset_chat,
            outputs=[chatbot, msg_input, message_history_state, token_display],
        )
        refresh_names_btn.click(refresh_names, outputs=[name_filter])

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

    # Load document set names on app startup
    demo.load(refresh_names, outputs=[name_filter])


if __name__ == "__main__":
    print(f"Starting Gradio app...")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Weaviate URL: {WEAVIATE_URL}")
    print(f"Multimodal mode: {MULTIMODAL_MODE}")
    if MULTIMODAL_MODE:
        print(f"Chat model: {CHAT_MODEL_MULTIMODAL} (vision + tools)")
        print(f"Embed model: CLIP ViT-B-32 (multi2vec-clip)")
        print(f"Collection: MultimodalDocument")
        print(f"Image analysis: VLM at query time (no pre-generated captions)")
    else:
        print(f"Chat model: {CHAT_MODEL}")
        print(f"Embed model: {EMBED_MODEL}")
        print(f"Collection: Document")

    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    finally:
        close_weaviate_client()

#!/usr/bin/env python3
"""Gradio app demonstrating Ollama embedding and generation for RAG."""

import os
import time

import gradio as gr

from agent import RAGDeps, create_weaviate_client, get_agent, get_multimodal_agent, get_available_names, run_multimodal_with_images
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

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


async def rag_chat(
    message: str,
    query_image: str | None,
    history: list,
    message_history: list,
    rag_mode: str,
    name_filter: list[str],
) -> tuple[list, str, str | None, list, str]:
    """Gradio handler for RAG-powered chat with conversation memory.

    Args:
        message: User's input message.
        query_image: Optional path to query image for multimodal search.
        history: Gradio chatbot display history.
        message_history: Pydantic AI message history for conversation memory.
        rag_mode: RAG mode selection ("Auto", "Force", "Disabled").
        name_filter: List of document set names to filter by (empty = all).

    Returns:
        Tuple of (updated history, cleared input, cleared image, updated message_history, token info).
    """
    if not message.strip():
        return history, "", None, message_history, ""

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
                return history, "", None, message_history, ""

            # Check if collection exists
            try:
                collections = client.collections.list_all()
                if collection_name not in [c for c in collections]:
                    history.append({
                        "role": "assistant",
                        "content": f"Error: {collection_name} collection not found. Please ingest documents first."
                    })
                    return history, "", None, message_history, ""
            except Exception as e:
                history.append({
                    "role": "assistant",
                    "content": f"Error checking collections: {e}"
                })
                return history, "", None, message_history, ""

        # Create dependencies with default retrieval settings
        deps = RAGDeps(
            weaviate_client=client,
            collection_name=collection_name,
            multimodal=MULTIMODAL_MODE,
            num_chunks=5,
            chunk_content_size=500,
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
                query_image_path=query_image,
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

        return history, "", None, new_message_history, token_info

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })
        return history, "", None, message_history, ""


def refresh_names():
    """Refresh document set names from Weaviate for the CheckboxGroup."""
    names = fetch_document_names()
    return gr.CheckboxGroup(choices=names, value=[])


# Build Gradio interface
with gr.Blocks(title="Pydantic RAG") as demo:
    gr.Markdown("# Pydantic RAG")
    gr.Markdown("RAG-powered chat using Pydantic AI, Ollama, and Weaviate hybrid search.")
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
            value="Force",
            label="RAG Mode",
            info="Auto: Agent decides | Force: Always search | Disabled: Plain chat",
        )
        token_display = gr.Textbox(
            label="Token Usage",
            value="",
            interactive=False,
            scale=1,
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

    with gr.Row():
        msg_input = gr.Textbox(
            label="Message",
            placeholder="Ask a question about your documents...",
            show_label=False,
            scale=4,
        )
        query_image = gr.Image(
            label="Query Image (optional)",
            type="filepath",
            visible=MULTIMODAL_MODE,
            scale=1,
        )

    with gr.Row():
        reset_btn = gr.Button("Reset Chat", variant="secondary")

    # Reset clears chatbot display, input, query image, message history, and token display
    def reset_chat():
        return [], "", None, [], ""

    msg_input.submit(
        rag_chat,
        inputs=[msg_input, query_image, chatbot, message_history_state, rag_mode, name_filter],
        outputs=[chatbot, msg_input, query_image, message_history_state, token_display],
    )
    reset_btn.click(
        reset_chat,
        outputs=[chatbot, msg_input, query_image, message_history_state, token_display],
    )
    refresh_names_btn.click(refresh_names, outputs=[name_filter])

    # Load document set names on app startup
    demo.load(refresh_names, outputs=[name_filter])


if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
    logger.info(f"Weaviate URL: {WEAVIATE_URL}")
    logger.info(f"Multimodal mode: {MULTIMODAL_MODE}")
    if MULTIMODAL_MODE:
        logger.info(f"Chat model: {CHAT_MODEL_MULTIMODAL} (vision + tools)")
        logger.info("Embed model: CLIP ViT-B-32 (multi2vec-clip)")
        logger.info("Collection: MultimodalDocument")
        logger.info("Image analysis: VLM at query time (no pre-generated captions)")
    else:
        logger.info(f"Chat model: {CHAT_MODEL}")
        logger.info(f"Embed model: {EMBED_MODEL}")
        logger.info("Collection: Document")

    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    finally:
        close_weaviate_client()

#!/bin/bash
# Minimal Ollama entrypoint for MCP/embedding-only use.
# Only pulls the embedding model — no LLMs loaded.
ollama serve &

# Wait for ollama to be ready
sleep 5

# Pull only the embedding model used by Weaviate text2vec-ollama
ollama pull bge-m3

# Keep container running
wait

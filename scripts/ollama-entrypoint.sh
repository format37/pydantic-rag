#!/bin/bash
# Start ollama in background
ollama serve &

# Wait for ollama to be ready
sleep 5

# Pull models (idempotent - skips if exists)
ollama pull nomic-embed-text
ollama pull llama3.2

# Vision-language model for multimodal RAG
# mistral-small3.1 supports BOTH vision AND tool calling
# ~13GB quantized, needs ~24GB VRAM for good performance
ollama pull mistral-small3.1

# Keep container running
wait

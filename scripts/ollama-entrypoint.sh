#!/bin/bash
# Start ollama in background
ollama serve &

# Wait for ollama to be ready
sleep 5

# Pull models (idempotent - skips if exists)
ollama pull nomic-embed-text
ollama pull llama3.2

# Keep container running
wait

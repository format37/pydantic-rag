#!/usr/bin/env python3
"""Test script to verify Ollama is running and models are available."""

import logging
import time

import httpx

# Standalone logging setup for test script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_ollama")

OLLAMA_BASE_URL = "http://localhost:11434"


def wait_for_ollama(timeout: int = 300, poll_interval: int = 5) -> bool:
    """Wait for Ollama to be ready by polling /api/tags."""
    logger.info(f"Waiting for Ollama to be ready (timeout: {timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is ready!")
                return True
        except httpx.ConnectError:
            pass

        elapsed = int(time.time() - start)
        logger.debug(f"  Waiting... ({elapsed}s elapsed)")
        time.sleep(poll_interval)

    logger.error("Timeout waiting for Ollama")
    return False


def list_models() -> list[str]:
    """List available models."""
    response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags")
    response.raise_for_status()

    data = response.json()
    models = [m["name"] for m in data.get("models", [])]
    return models


def test_embedding(model: str = "nomic-embed-text") -> bool:
    """Test embedding generation."""
    logger.info(f"Testing embedding with {model}...")

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": "Hello, world!"},
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings", [])

    if embeddings and len(embeddings[0]) > 0:
        logger.info(f"  Embedding dimension: {len(embeddings[0])}")
        logger.info(f"  First 5 values: {embeddings[0][:5]}")
        return True

    logger.error("  Failed: No embeddings returned")
    return False


def test_generation(model: str = "llama3.2") -> bool:
    """Test text generation."""
    logger.info(f"Testing generation with {model}...")

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": "Say 'Hello from Ollama!' and nothing else.",
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    generated = data.get("response", "")

    if generated:
        logger.info(f"  Response: {generated.strip()[:100]}")
        return True

    logger.error("  Failed: No response generated")
    return False


def main():
    logger.info("=" * 50)
    logger.info("Ollama Test Script")
    logger.info("=" * 50)

    # Wait for Ollama to be ready
    if not wait_for_ollama():
        logger.error("Failed: Ollama is not available")
        return 1

    # List models
    logger.info("Available models:")
    models = list_models()
    if not models:
        logger.warning("  No models found. Models may still be downloading.")
        logger.info("  Check logs with: docker compose logs -f ollama")
        return 1

    for model in models:
        logger.info(f"  - {model}")

    # Test embedding
    embedding_ok = test_embedding()

    # Test generation
    generation_ok = test_generation()

    # Summary
    logger.info("=" * 50)
    logger.info("Results:")
    logger.info(f"  Embedding: {'PASS' if embedding_ok else 'FAIL'}")
    logger.info(f"  Generation: {'PASS' if generation_ok else 'FAIL'}")
    logger.info("=" * 50)

    return 0 if (embedding_ok and generation_ok) else 1


if __name__ == "__main__":
    exit(main())

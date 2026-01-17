#!/usr/bin/env python3
"""Test script to verify Ollama is running and models are available."""

import time
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"


def wait_for_ollama(timeout: int = 300, poll_interval: int = 5) -> bool:
    """Wait for Ollama to be ready by polling /api/tags."""
    print(f"Waiting for Ollama to be ready (timeout: {timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama is ready!")
                return True
        except httpx.ConnectError:
            pass

        elapsed = int(time.time() - start)
        print(f"  Waiting... ({elapsed}s elapsed)")
        time.sleep(poll_interval)

    print("Timeout waiting for Ollama")
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
    print(f"\nTesting embedding with {model}...")

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": "Hello, world!"},
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings", [])

    if embeddings and len(embeddings[0]) > 0:
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print(f"  First 5 values: {embeddings[0][:5]}")
        return True

    print("  Failed: No embeddings returned")
    return False


def test_generation(model: str = "llama3.2") -> bool:
    """Test text generation."""
    print(f"\nTesting generation with {model}...")

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
        print(f"  Response: {generated.strip()[:100]}")
        return True

    print("  Failed: No response generated")
    return False


def main():
    print("=" * 50)
    print("Ollama Test Script")
    print("=" * 50)

    # Wait for Ollama to be ready
    if not wait_for_ollama():
        print("\nFailed: Ollama is not available")
        return 1

    # List models
    print("\nAvailable models:")
    models = list_models()
    if not models:
        print("  No models found. Models may still be downloading.")
        print("  Check logs with: docker compose logs -f ollama")
        return 1

    for model in models:
        print(f"  - {model}")

    # Test embedding
    embedding_ok = test_embedding()

    # Test generation
    generation_ok = test_generation()

    # Summary
    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Embedding: {'PASS' if embedding_ok else 'FAIL'}")
    print(f"  Generation: {'PASS' if generation_ok else 'FAIL'}")
    print("=" * 50)

    return 0 if (embedding_ok and generation_ok) else 1


if __name__ == "__main__":
    exit(main())

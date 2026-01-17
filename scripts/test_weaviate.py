#!/usr/bin/env python3
"""
Test script for Weaviate connection and hybrid search.

Verifies:
- Weaviate connection
- Collection exists and has data
- Hybrid search (keyword + vector)

Usage:
    python scripts/test_weaviate.py [--weaviate-url URL] [--query "search query"]
"""

import argparse
import os
import sys
from urllib.parse import urlparse

import weaviate


def test_connection(client: weaviate.WeaviateClient) -> bool:
    """Test basic Weaviate connection."""
    print("Testing connection...")
    if client.is_ready():
        print("  - Weaviate is ready")
        return True
    else:
        print("  - Weaviate is NOT ready")
        return False


def test_collection(client: weaviate.WeaviateClient) -> bool:
    """Test that Document collection exists and has data."""
    print("\nTesting collection...")

    if not client.collections.exists("Document"):
        print("  - Collection 'Document' does not exist")
        print("  - Run 'python scripts/ingest.py' to create it")
        return False

    print("  - Collection 'Document' exists")

    collection = client.collections.get("Document")

    # Get count
    response = collection.aggregate.over_all(total_count=True)
    count = response.total_count

    print(f"  - Document count: {count}")

    if count == 0:
        print("  - Collection is empty. Run ingest.py with documents.")
        return False

    # Get sample
    sample = collection.query.fetch_objects(limit=1)
    if sample.objects:
        obj = sample.objects[0]
        print(f"  - Sample source: {obj.properties.get('source', 'N/A')}")

    return True


def test_hybrid_search(client: weaviate.WeaviateClient, query: str) -> bool:
    """Test hybrid search functionality."""
    print(f"\nTesting hybrid search with query: '{query}'")

    if not client.collections.exists("Document"):
        print("  - Collection 'Document' does not exist")
        return False

    collection = client.collections.get("Document")

    # Test different alpha values
    for alpha, name in [(0.0, "keyword-only"), (0.5, "balanced"), (1.0, "vector-only")]:
        print(f"\n  Alpha={alpha} ({name}):")

        try:
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=3
            )

            if not response.objects:
                print("    No results found")
            else:
                for i, obj in enumerate(response.objects, 1):
                    source = obj.properties.get("source", "unknown")
                    chunk_idx = obj.properties.get("chunk_index", "?")
                    content = obj.properties.get("content", "")[:100]
                    print(f"    {i}. [{source}:chunk {chunk_idx}]")
                    print(f"       {content}...")

        except Exception as e:
            print(f"    Error: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Weaviate setup and hybrid search")
    parser.add_argument(
        "--weaviate-url",
        default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"),
        help="Weaviate URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--query",
        default="What are the main topics covered?",
        help="Search query to test (default: 'What are the main topics covered?')"
    )
    args = parser.parse_args()

    # Parse URL
    parsed = urlparse(args.weaviate_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    print(f"Connecting to Weaviate at {args.weaviate_url}...")

    try:
        client = weaviate.connect_to_local(host=host, port=port)

        # Run tests
        results = []

        results.append(("Connection", test_connection(client)))
        results.append(("Collection", test_collection(client)))
        results.append(("Hybrid Search", test_hybrid_search(client, args.query)))

        # Summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        all_passed = True
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\nAll tests passed!")
            sys.exit(0)
        else:
            print("\nSome tests failed.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Weaviate is running:")
        print("  docker compose up -d weaviate")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()

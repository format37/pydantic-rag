#!/usr/bin/env python3
"""Wrapper script to ingest MRAG-Bench corpus using existing ingestion pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path


def ingest_mrag_bench(
    name: str = "mrag_bench",
    scripts_dir: str = "scripts",
) -> None:
    """Run ingestion for MRAG-Bench corpus.

    Args:
        name: Document set name for the benchmark corpus.
        scripts_dir: Directory containing ingest.py script.
    """
    ingest_script = Path(scripts_dir) / "ingest.py"

    if not ingest_script.exists():
        raise FileNotFoundError(
            f"Ingestion script not found: {ingest_script}\n"
            "Please run this from the project root directory."
        )

    # Verify corpus symlink exists
    corpus_link = Path("data/documents/mrag_bench_corpus")
    if not corpus_link.exists():
        raise FileNotFoundError(
            f"Corpus symlink not found: {corpus_link}\n"
            "Please run prepare_corpus.py first."
        )

    print(f"Running ingestion for MRAG-Bench corpus...")
    print(f"  Name: {name}")
    print(f"  Mode: multimodal (CLIP embeddings)")
    print()

    # Run ingestion script with multimodal flag
    cmd = [
        sys.executable,
        str(ingest_script),
        "--name", name,
        "--multimodal",
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Ingestion failed with return code: {result.returncode}")
        sys.exit(result.returncode)

    print()
    print("Ingestion complete!")
    print("Next step: Start services and run evaluation:")
    print("  docker compose up -d")
    print("  python -m benchmark.mrag_bench.evaluate --limit 10")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest MRAG-Bench corpus into Weaviate"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mrag_bench",
        help="Document set name",
    )
    args = parser.parse_args()

    ingest_mrag_bench(args.name)


if __name__ == "__main__":
    main()

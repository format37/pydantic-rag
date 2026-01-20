#!/usr/bin/env python3
"""Prepare MRAG-Bench corpus for ingestion by creating symlink."""

import argparse
import os
from pathlib import Path


def prepare_corpus(
    mrag_bench_dir: str = "data/mrag_bench",
    documents_dir: str = "data/documents",
) -> None:
    """Create symlink from corpus to documents directory for ingestion.

    Args:
        mrag_bench_dir: Directory containing downloaded MRAG-Bench dataset.
        documents_dir: Base documents directory for ingestion pipeline.
    """
    mrag_bench_path = Path(mrag_bench_dir)
    corpus_path = mrag_bench_path / "corpus"
    documents_path = Path(documents_dir)
    target_path = documents_path / "mrag_bench_corpus"

    # Verify corpus exists
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_path}\n"
            "Please run download.py first to download the dataset."
        )

    # Create documents directory if needed
    documents_path.mkdir(parents=True, exist_ok=True)

    # Remove existing symlink if present
    if target_path.is_symlink():
        target_path.unlink()
        print(f"Removed existing symlink: {target_path}")
    elif target_path.exists():
        raise FileExistsError(
            f"Target path exists and is not a symlink: {target_path}\n"
            "Please remove it manually if you want to proceed."
        )

    # Create symlink (use absolute path for reliability)
    corpus_abs = corpus_path.resolve()
    target_path.symlink_to(corpus_abs)

    # Count images in corpus
    image_count = sum(1 for _ in corpus_path.rglob("*.png"))
    scenario_count = len([d for d in corpus_path.iterdir() if d.is_dir()])

    print(f"Corpus prepared successfully!")
    print(f"  Source: {corpus_abs}")
    print(f"  Target: {target_path}")
    print(f"  Scenarios: {scenario_count}")
    print(f"  Images: {image_count}")
    print()
    print("Next step: Run ingestion with multimodal mode:")
    print("  python scripts/ingest.py --name mrag_bench --multimodal")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MRAG-Bench corpus for ingestion"
    )
    parser.add_argument(
        "--mrag-bench-dir",
        type=str,
        default="data/mrag_bench",
        help="Directory containing MRAG-Bench dataset",
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="data/documents",
        help="Base documents directory",
    )
    args = parser.parse_args()

    prepare_corpus(args.mrag_bench_dir, args.documents_dir)


if __name__ == "__main__":
    main()

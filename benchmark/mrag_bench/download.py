#!/usr/bin/env python3
"""Download MRAG-Bench dataset from HuggingFace."""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_mrag_bench(output_dir: str = "data/mrag_bench") -> None:
    """Download MRAG-Bench dataset and organize files.

    Args:
        output_dir: Directory to save the dataset.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    corpus_dir = output_path / "corpus"

    # Create directories
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    corpus_dir.mkdir(exist_ok=True)

    print("Loading MRAG-Bench dataset from HuggingFace...")
    dataset = load_dataset("uclanlp/MRAG-Bench")

    # Process the test split (MRAG-Bench uses test split)
    test_data = dataset["test"]

    metadata = []
    corpus_images_saved = set()

    print(f"Processing {len(test_data)} questions...")
    for idx, item in enumerate(tqdm(test_data, desc="Processing questions")):
        # Extract question data
        question_id = item.get("id", idx)
        question_text = item["question"]
        choices = item["choices"]  # List of choice strings
        answer_idx = item["answer"]  # Index of correct answer (0-3)
        scenario = item.get("scenario", "unknown")
        aspect = item.get("aspect", "unknown")

        # Save query image
        query_image = item["image"]
        if query_image is not None:
            query_image_path = images_dir / f"question_{idx}.png"
            if isinstance(query_image, Image.Image):
                query_image.save(query_image_path)
            else:
                # Handle case where image might be in different format
                Image.open(query_image).save(query_image_path)

        # Process ground-truth corpus images
        gt_images = item.get("gt_images", [])
        gt_image_paths = []

        # Create scenario subdirectory for corpus
        scenario_dir = corpus_dir / scenario
        scenario_dir.mkdir(exist_ok=True)

        for gt_idx, gt_image in enumerate(gt_images):
            if gt_image is not None:
                # Create unique filename for corpus image
                gt_filename = f"{scenario}_{question_id}_{gt_idx}.png"
                gt_path = scenario_dir / gt_filename

                # Only save if not already saved
                if gt_filename not in corpus_images_saved:
                    if isinstance(gt_image, Image.Image):
                        gt_image.save(gt_path)
                    else:
                        Image.open(gt_image).save(gt_path)
                    corpus_images_saved.add(gt_filename)

                gt_image_paths.append(str(gt_path.relative_to(output_path)))

        # Build metadata entry
        metadata_entry = {
            "id": question_id,
            "index": idx,
            "question": question_text,
            "choices": choices,
            "answer_idx": answer_idx,
            "answer_letter": chr(65 + answer_idx),  # A, B, C, or D
            "scenario": scenario,
            "aspect": aspect,
            "query_image": f"images/question_{idx}.png",
            "gt_images": gt_image_paths,
        }
        metadata.append(metadata_entry)

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\nDataset downloaded successfully!")
    print(f"  Output directory: {output_path}")
    print(f"  Questions: {len(metadata)}")
    print(f"  Query images: {len(list(images_dir.glob('*.png')))}")
    print(f"  Corpus images: {len(corpus_images_saved)}")
    print(f"  Metadata file: {metadata_path}")

    # Print scenario breakdown
    scenarios = {}
    for entry in metadata:
        scenario = entry["scenario"]
        scenarios[scenario] = scenarios.get(scenario, 0) + 1

    print("\nScenario breakdown:")
    for scenario, count in sorted(scenarios.items()):
        print(f"  {scenario}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Download MRAG-Bench dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/mrag_bench",
        help="Output directory for the dataset",
    )
    args = parser.parse_args()

    download_mrag_bench(args.output)


if __name__ == "__main__":
    main()

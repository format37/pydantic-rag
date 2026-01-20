#!/usr/bin/env python3
"""Compute and display metrics for MRAG-Bench evaluation results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .baselines import compare_with_baselines


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute accuracy metrics from evaluation results.

    Args:
        results: List of result dictionaries with keys:
            - correct: bool indicating if answer was correct
            - scenario: str scenario name
            - aspect: str aspect name
            - extracted_answer: str or None
            - ground_truth: str expected answer

    Returns:
        Dictionary with computed metrics.
    """
    if not results:
        return {
            "overall": {"accuracy": 0.0, "correct": 0, "total": 0},
            "by_scenario": {},
            "by_aspect": {},
            "extraction_failures": 0,
        }

    # Overall metrics
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    extraction_failures = sum(1 for r in results if r.get("extracted_answer") is None)

    # Per-scenario metrics
    by_scenario: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        scenario = r.get("scenario", "unknown")
        by_scenario[scenario]["total"] += 1
        if r.get("correct", False):
            by_scenario[scenario]["correct"] += 1

    # Calculate scenario accuracies
    scenario_metrics = {}
    for scenario, counts in by_scenario.items():
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        scenario_metrics[scenario] = {
            "accuracy": accuracy,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    # Per-aspect metrics
    by_aspect: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        aspect = r.get("aspect", "unknown")
        by_aspect[aspect]["total"] += 1
        if r.get("correct", False):
            by_aspect[aspect]["correct"] += 1

    # Calculate aspect accuracies
    aspect_metrics = {}
    for aspect, counts in by_aspect.items():
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        aspect_metrics[aspect] = {
            "accuracy": accuracy,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return {
        "overall": {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        },
        "by_scenario": scenario_metrics,
        "by_aspect": aspect_metrics,
        "extraction_failures": extraction_failures,
    }


def print_metrics(metrics: dict[str, Any], show_baselines: bool = True) -> None:
    """Print metrics in a formatted table.

    Args:
        metrics: Dictionary from compute_metrics().
        show_baselines: Whether to show comparison with published baselines.
    """
    overall = metrics["overall"]
    print("=" * 60)
    print("MRAG-Bench Evaluation Results")
    print("=" * 60)

    # Overall accuracy
    print(f"\nOverall Accuracy: {overall['accuracy']:.1%}")
    print(f"  Correct: {overall['correct']} / {overall['total']}")

    if metrics["extraction_failures"] > 0:
        print(f"  Extraction Failures: {metrics['extraction_failures']}")

    # Per-scenario breakdown
    if metrics["by_scenario"]:
        print("\n" + "-" * 40)
        print("Accuracy by Scenario:")
        print("-" * 40)
        for scenario in sorted(metrics["by_scenario"].keys()):
            data = metrics["by_scenario"][scenario]
            print(
                f"  {scenario:20} {data['accuracy']:6.1%} "
                f"({data['correct']:3}/{data['total']:3})"
            )

    # Per-aspect breakdown
    if metrics["by_aspect"]:
        print("\n" + "-" * 40)
        print("Accuracy by Aspect:")
        print("-" * 40)
        for aspect in sorted(metrics["by_aspect"].keys()):
            data = metrics["by_aspect"][aspect]
            print(
                f"  {aspect:20} {data['accuracy']:6.1%} "
                f"({data['correct']:3}/{data['total']:3})"
            )

    # Baseline comparison
    if show_baselines:
        print("\n" + "-" * 40)
        print("Comparison with Published Baselines:")
        print("-" * 40)
        comparison = compare_with_baselines(overall["accuracy"])
        for line in comparison:
            print(f"  {line}")

    print("=" * 60)


def load_results(filepath: str) -> list[dict]:
    """Load results from JSON file.

    Args:
        filepath: Path to results JSON file.

    Returns:
        List of result dictionaries.
    """
    with open(filepath) as f:
        data = json.load(f)

    # Handle both formats: list of results or dict with "results" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "results" in data:
        return data["results"]
    else:
        raise ValueError(f"Unexpected results format in {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from MRAG-Bench evaluation results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Don't show baseline comparison",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output metrics as JSON",
    )
    args = parser.parse_args()

    # Load results
    results = load_results(args.results_file)

    # Compute metrics
    metrics = compute_metrics(results)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_metrics(metrics, show_baselines=not args.no_baselines)


if __name__ == "__main__":
    main()

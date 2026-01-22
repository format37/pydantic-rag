#!/usr/bin/env python3
"""Generate visualization plots for MRAG-Bench evaluation results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.mrag_bench.baselines import BASELINES

# Plot styling
COLORS = {
    "primary": "#2563eb",  # Blue for our results
    "secondary": "#0891b2",  # Teal
    "baseline": "#6b7280",  # Gray for baselines
    "highlight": "#059669",  # Green for emphasis
}
FIGURE_SIZE = (10, 6)
DPI = 150


def load_results(results_path: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def calculate_scenario_accuracy(results: list[dict]) -> dict[str, dict]:
    """Calculate accuracy for each scenario."""
    scenarios = {}
    for r in results:
        s = r["scenario"]
        if s not in scenarios:
            scenarios[s] = {"total": 0, "correct": 0}
        scenarios[s]["total"] += 1
        if r["correct"]:
            scenarios[s]["correct"] += 1

    return {
        name: {
            "accuracy": data["correct"] / data["total"],
            "total": data["total"],
            "correct": data["correct"],
        }
        for name, data in scenarios.items()
    }


def calculate_aspect_accuracy(results: list[dict]) -> dict[str, dict]:
    """Calculate accuracy for each aspect."""
    aspects = {}
    for r in results:
        a = r["aspect"]
        if a not in aspects:
            aspects[a] = {"total": 0, "correct": 0}
        aspects[a]["total"] += 1
        if r["correct"]:
            aspects[a]["correct"] += 1

    return {
        name: {
            "accuracy": data["correct"] / data["total"],
            "total": data["total"],
            "correct": data["correct"],
        }
        for name, data in aspects.items()
    }


def plot_scenario_accuracy(scenario_data: dict, output_path: Path) -> None:
    """Generate horizontal bar chart for scenario accuracy."""
    # Sort by accuracy (highest to lowest)
    sorted_scenarios = sorted(
        scenario_data.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    names = [f"{name} ({data['total']})" for name, data in sorted_scenarios]
    accuracies = [data["accuracy"] * 100 for _, data in sorted_scenarios]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create horizontal bars
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accuracies, color=COLORS["primary"], height=0.7)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("MRAG-Bench Accuracy by Scenario", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 60)
    ax.invert_yaxis()  # Highest at top

    # Add gridlines
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_aspect_accuracy(aspect_data: dict, output_path: Path) -> None:
    """Generate horizontal bar chart for aspect accuracy."""
    # Sort by accuracy (highest to lowest)
    sorted_aspects = sorted(
        aspect_data.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    names = [f"{name} ({data['total']})" for name, data in sorted_aspects]
    accuracies = [data["accuracy"] * 100 for _, data in sorted_aspects]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Create horizontal bars
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accuracies, color=COLORS["secondary"], height=0.6)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("MRAG-Bench Accuracy by Aspect", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 55)
    ax.invert_yaxis()

    # Add gridlines
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_baseline_comparison(our_accuracy: float, output_path: Path) -> None:
    """Generate horizontal bar chart comparing our result with published baselines."""
    # Add our result to baselines
    all_results = {**BASELINES, "Ours (mistral-small3.1)": our_accuracy}

    # Sort by accuracy (highest to lowest)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

    names = [name for name, _ in sorted_results]
    accuracies = [acc * 100 for _, acc in sorted_results]

    # Create color array - highlight our result
    colors = [
        COLORS["highlight"] if "Ours" in name else COLORS["baseline"]
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create horizontal bars
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accuracies, color=colors, height=0.7)

    # Add value labels on bars
    for bar, acc, name in zip(bars, accuracies, names):
        width = bar.get_width()
        fontweight = "bold" if "Ours" in name else "normal"
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
            fontsize=9,
            fontweight=fontweight,
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "MRAG-Bench: Comparison with Published Baselines",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 85)
    ax.invert_yaxis()

    # Add gridlines
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["highlight"], label="Our System"),
        Patch(facecolor=COLORS["baseline"], label="Published Baselines"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all benchmark visualizations."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / "benchmark" / "results" / "full_run.json"
    assets_dir = project_root / "benchmark" / "assets"

    # Ensure assets directory exists
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {results_path}")
    data = load_results(results_path)
    results = data["results"]

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall_accuracy = correct / total

    print(f"\nOverall accuracy: {overall_accuracy * 100:.1f}% ({correct}/{total})")

    # Calculate breakdowns
    scenario_data = calculate_scenario_accuracy(results)
    aspect_data = calculate_aspect_accuracy(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_scenario_accuracy(scenario_data, assets_dir / "scenario_accuracy.png")
    plot_aspect_accuracy(aspect_data, assets_dir / "aspect_accuracy.png")
    plot_baseline_comparison(overall_accuracy, assets_dir / "baseline_comparison.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

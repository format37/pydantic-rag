#!/usr/bin/env python3
"""Published baseline results for MRAG-Bench benchmark."""

# Published baseline results from MRAG-Bench paper
# Source: https://arxiv.org/abs/2410.13085
BASELINES = {
    # GPT models
    "GPT-4o": 0.745,
    "GPT-4V": 0.653,
    "GPT-4o-mini": 0.612,

    # Human performance
    "Human": 0.716,

    # Gemini models
    "Gemini 1.5 Pro": 0.624,
    "Gemini 1.5 Flash": 0.589,

    # Claude models
    "Claude 3 Opus": 0.598,
    "Claude 3.5 Sonnet": 0.621,

    # Open-source models
    "LLaVA-1.5-13B": 0.412,
    "LLaVA-1.6-34B": 0.467,
    "InternVL2-26B": 0.534,
    "Qwen-VL-Max": 0.573,

    # Multimodal RAG baselines (with retrieval)
    "GPT-4o + CLIP": 0.689,
    "GPT-4o + BLIP2": 0.672,
}


def compare_with_baselines(accuracy: float) -> list[str]:
    """Compare our accuracy with published baselines.

    Args:
        accuracy: Our model's accuracy (0.0-1.0).

    Returns:
        List of formatted comparison strings.
    """
    our_pct = accuracy * 100

    # Sort baselines by accuracy (descending)
    sorted_baselines = sorted(BASELINES.items(), key=lambda x: x[1], reverse=True)

    lines = []

    # Find where our result falls
    rank = 1
    for name, baseline_acc in sorted_baselines:
        if accuracy >= baseline_acc:
            break
        rank += 1

    lines.append(f"Our result: {our_pct:.1f}% (rank {rank}/{len(BASELINES) + 1})")
    lines.append("")

    # Show top baselines with comparison
    for name, baseline_acc in sorted_baselines[:8]:
        baseline_pct = baseline_acc * 100
        diff = our_pct - baseline_pct
        if diff >= 0:
            marker = f"+{diff:.1f}%"
        else:
            marker = f"{diff:.1f}%"

        # Highlight if we beat this baseline
        indicator = "*" if accuracy >= baseline_acc else " "
        lines.append(f"{indicator} {name:20} {baseline_pct:5.1f}% ({marker})")

    return lines


def get_baseline(name: str) -> float | None:
    """Get a specific baseline accuracy.

    Args:
        name: Name of the baseline model.

    Returns:
        Accuracy as float (0.0-1.0), or None if not found.
    """
    return BASELINES.get(name)


def print_all_baselines() -> None:
    """Print all published baselines."""
    print("MRAG-Bench Published Baselines")
    print("=" * 40)

    sorted_baselines = sorted(BASELINES.items(), key=lambda x: x[1], reverse=True)

    for name, accuracy in sorted_baselines:
        print(f"{name:25} {accuracy * 100:5.1f}%")


if __name__ == "__main__":
    print_all_baselines()

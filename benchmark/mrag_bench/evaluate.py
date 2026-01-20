#!/usr/bin/env python3
"""Main evaluation script for MRAG-Bench benchmark using Gradio API."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from tqdm import tqdm

from .answer_extractor import extract_answer
from .metrics import compute_metrics, print_metrics


def format_question(question: str, choices: list[str]) -> str:
    """Format a multiple-choice question for the VLM.

    Args:
        question: The question text.
        choices: List of 4 answer choices.

    Returns:
        Formatted question string with choices labeled A-D.
    """
    formatted = f"{question}\n\n"
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D
        formatted += f"{letter}) {choice}\n"
    formatted += "\nPlease answer with just the letter (A, B, C, or D) of the correct choice."
    return formatted


def load_metadata(metadata_path: str) -> list[dict]:
    """Load question metadata from JSON file.

    Args:
        metadata_path: Path to metadata.json.

    Returns:
        List of question metadata dictionaries.
    """
    with open(metadata_path) as f:
        return json.load(f)


def load_checkpoint(checkpoint_path: str) -> dict[int, dict]:
    """Load existing results from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file.

    Returns:
        Dictionary mapping question index to result dict.
    """
    if not Path(checkpoint_path).exists():
        return {}

    with open(checkpoint_path) as f:
        data = json.load(f)

    # Convert results list to dict indexed by question index
    results = data.get("results", [])
    return {r["index"]: r for r in results}


def save_checkpoint(
    results: list[dict],
    output_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save results to checkpoint file.

    Args:
        results: List of result dictionaries.
        output_path: Path to output JSON file.
        metadata: Optional metadata to include.
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    if metadata:
        data["metadata"] = metadata

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def evaluate_question(
    client: Client,
    question_data: dict,
    data_dir: str,
    name_filter: list[str],
) -> dict:
    """Evaluate a single question using the Gradio API.

    Args:
        client: Gradio client instance.
        question_data: Question metadata dictionary.
        data_dir: Base directory containing the dataset.
        name_filter: Document set names to filter by.

    Returns:
        Result dictionary with question details and evaluation outcome.
    """
    idx = question_data["index"]
    question = question_data["question"]
    choices = question_data["choices"]
    ground_truth = question_data["answer_letter"]
    scenario = question_data.get("scenario", "unknown")
    aspect = question_data.get("aspect", "unknown")

    # Format the question
    formatted_question = format_question(question, choices)

    # Get query image path
    query_image_path = Path(data_dir) / question_data["query_image"]

    try:
        # Call the Gradio API
        # The API signature matches: rag_chat(message, query_image, history, message_history, rag_mode, name_filter)
        result = client.predict(
            message=formatted_question,
            query_image=handle_file(str(query_image_path)),
            history=[],
            message_history=[],
            rag_mode="Force",
            name_filter=name_filter,
            api_name="/rag_chat",
        )

        # Extract the response from result
        # Result format depends on outputs: [chatbot, msg_input, query_image, message_history_state, token_display]
        if isinstance(result, tuple) and len(result) >= 1:
            chatbot_history = result[0]
            if chatbot_history and len(chatbot_history) > 0:
                # Get the last assistant message
                last_message = chatbot_history[-1]
                if isinstance(last_message, dict):
                    response = last_message.get("content", "")
                elif isinstance(last_message, (list, tuple)) and len(last_message) > 1:
                    response = last_message[1]  # (user, assistant) format
                else:
                    response = str(last_message)
            else:
                response = ""
        else:
            response = str(result)

        # Extract answer from response
        extracted_answer = extract_answer(response)
        correct = extracted_answer == ground_truth

        return {
            "index": idx,
            "question": question,
            "choices": choices,
            "ground_truth": ground_truth,
            "extracted_answer": extracted_answer,
            "correct": correct,
            "scenario": scenario,
            "aspect": aspect,
            "response": response[:500],  # Truncate for storage
            "error": None,
        }

    except Exception as e:
        return {
            "index": idx,
            "question": question,
            "choices": choices,
            "ground_truth": ground_truth,
            "extracted_answer": None,
            "correct": False,
            "scenario": scenario,
            "aspect": aspect,
            "response": None,
            "error": str(e),
        }


def run_evaluation(
    gradio_url: str = "http://localhost:7860",
    data_dir: str = "data/mrag_bench",
    output_path: str = "benchmark/results/mrag_bench_results.json",
    limit: int | None = None,
    name_filter: list[str] | None = None,
    checkpoint_interval: int = 10,
    resume: bool = False,
) -> dict[str, Any]:
    """Run MRAG-Bench evaluation.

    Args:
        gradio_url: URL of the Gradio app.
        data_dir: Directory containing MRAG-Bench dataset.
        output_path: Path to save results.
        limit: Maximum number of questions to evaluate (None = all).
        name_filter: Document set names to filter by.
        checkpoint_interval: Save checkpoint every N questions.
        resume: Whether to resume from existing checkpoint.

    Returns:
        Dictionary with evaluation results and metrics.
    """
    # Load metadata
    metadata_path = Path(data_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please run download.py first."
        )

    questions = load_metadata(str(metadata_path))
    print(f"Loaded {len(questions)} questions from {metadata_path}")

    # Apply limit
    if limit:
        questions = questions[:limit]
        print(f"Limiting to first {limit} questions")

    # Default name filter for MRAG-Bench
    if name_filter is None:
        name_filter = ["mrag_bench"]

    # Load checkpoint if resuming
    completed = {}
    if resume:
        completed = load_checkpoint(output_path)
        print(f"Resuming from checkpoint: {len(completed)} questions already completed")

    # Connect to Gradio
    print(f"Connecting to Gradio at {gradio_url}...")
    client = Client(gradio_url)
    print("Connected!")

    # Run evaluation
    results = list(completed.values())
    remaining = [q for q in questions if q["index"] not in completed]

    print(f"\nEvaluating {len(remaining)} questions...")
    start_time = time.time()

    for i, question_data in enumerate(tqdm(remaining, desc="Evaluating")):
        result = evaluate_question(client, question_data, data_dir, name_filter)
        results.append(result)

        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(results, output_path)

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    # Compute metrics
    metrics = compute_metrics(results)

    # Save final results
    eval_metadata = {
        "gradio_url": gradio_url,
        "data_dir": data_dir,
        "name_filter": name_filter,
        "total_questions": len(questions),
        "evaluated": len(results),
        "elapsed_seconds": elapsed,
    }
    save_checkpoint(results, output_path, metadata=eval_metadata)
    print(f"Results saved to {output_path}")

    # Print metrics
    print_metrics(metrics)

    return {
        "results": results,
        "metrics": metrics,
        "metadata": eval_metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run MRAG-Bench evaluation using Gradio API"
    )
    parser.add_argument(
        "--gradio-url",
        type=str,
        default="http://localhost:7860",
        help="URL of the Gradio app",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mrag_bench",
        help="Directory containing MRAG-Bench dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmark/results/mrag_bench_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--name-filter",
        type=str,
        nargs="+",
        default=None,
        help="Document set names to filter by",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N questions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint",
    )
    args = parser.parse_args()

    run_evaluation(
        gradio_url=args.gradio_url,
        data_dir=args.data_dir,
        output_path=args.output,
        limit=args.limit,
        name_filter=args.name_filter,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

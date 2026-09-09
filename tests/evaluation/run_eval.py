#!/usr/bin/env python3
"""
CLI runner for ForgeMind RAG Evaluation Baseline.
Executes benchmark queries through existing RAG pipeline and records metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add backend and project root directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
project_root = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure DEFAULT_MODEL is valid for Groq if not explicitly set
os.environ.setdefault("DEFAULT_MODEL", "openai/gpt-oss-120b")

from app.rag.rag_service import RAGService
from tests.evaluation.evaluator import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run ForgeMind RAG Evaluation Baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).parent / "dataset.json"),
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "baseline_raw_results.json"),
        help="Path to save raw evaluation results JSON",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation questions from {dataset_path}")
    print("Initializing ForgeMind RAG pipeline...")

    rag_service = RAGService()
    evaluator = RAGEvaluator(rag_service=rag_service)

    print("Executing evaluation (capturing retrieval, generation, latency)...")
    eval_results = evaluator.run_evaluation(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nRaw evaluation results saved to: {output_path}\n")

    # Print summary report
    summary = eval_results["summary"]
    print("=" * 65)
    print("FORGEMIND RAG EVALUATION BASELINE REPORT")
    print("=" * 65)
    print(f"Total Test Cases      : {summary['total_cases']}")
    print(f"  - Answerable        : {summary['answerable_count']}")
    print(f"  - Unanswerable      : {summary['unanswerable_count']}")
    print("-" * 65)
    print("RETRIEVAL METRICS (Answerable cases):")
    print(f"  - Retrieval Hit Rate: {summary['retrieval_hit_rate'] * 100:.1f}%")
    print(f"  - Retrieval Recall  : {summary['retrieval_recall'] * 100:.1f}%")
    print(f"  - Context Precision : {summary['context_precision'] * 100:.1f}%")
    print("-" * 65)
    print("ANSWER & SAFETY METRICS:")
    print(f"  - Unanswerable Refusal Rate     : {summary['unanswerable_refusal_rate'] * 100:.1f}%")
    print(f"  - Unanswerable False Context    : {summary['unanswerable_false_context_rate'] * 100:.1f}%")
    print(f"  - In-Text Citation Presence     : {summary['citation_presence_in_answer_rate'] * 100:.1f}%")
    print("-" * 65)
    print("PERFORMANCE / LATENCY:")
    print(f"  - Average Latency   : {summary['avg_latency_seconds']:.3f}s")
    print(f"  - P95 Latency       : {summary['p95_latency_seconds']:.3f}s")
    print("-" * 65)
    print("CATEGORY BREAKDOWN:")
    for cat, stats in eval_results["category_breakdown"].items():
        print(f"  [{cat:<24}] Cases: {stats['count']:<2} | Hit Rate: {stats['hit_rate']*100:>5.1f}% | Recall: {stats['recall']*100:>5.1f}%")
    print("-" * 65)
    print(f"TOTAL FAILED CASES: {summary['total_failures']}")
    print("=" * 65)


if __name__ == "__main__":
    main()

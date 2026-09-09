import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalResultItem:
    id: str
    category: str
    answerability: str
    question: str
    expected_answer: str
    expected_sources: List[str]
    generated_answer: str
    retrieved_sources: List[str]
    retrieved_chunk_count: int
    latency_seconds: float
    retrieval_hit: bool
    retrieval_recall: float
    context_precision: float
    refusal_detected: bool
    citation_present_in_answer: bool
    error: Optional[str] = None
    passed_all_deterministic_checks: bool = False
    failure_reasons: List[str] = field(default_factory=list)


def normalize_source_name(source: str) -> str:
    """Extract clean filename from path or source string."""
    if not source:
        return ""
    p = Path(source)
    return p.name


def calculate_retrieval_hit(expected_sources: List[str], retrieved_sources: List[str]) -> bool:
    """Returns True if at least one expected source is present in retrieved sources."""
    if not expected_sources:
        return False
    norm_expected = {normalize_source_name(s) for s in expected_sources}
    norm_retrieved = {normalize_source_name(s) for s in retrieved_sources}
    return len(norm_expected.intersection(norm_retrieved)) > 0


def calculate_retrieval_recall(expected_sources: List[str], retrieved_sources: List[str]) -> float:
    """Calculates recall: fraction of expected sources present in retrieved sources."""
    if not expected_sources:
        return 0.0
    norm_expected = {normalize_source_name(s) for s in expected_sources}
    norm_retrieved = {normalize_source_name(s) for s in retrieved_sources}
    matched = norm_expected.intersection(norm_retrieved)
    return len(matched) / len(norm_expected)


def calculate_context_precision(expected_sources: List[str], retrieved_sources: List[str], k: int = 5) -> float:
    """Calculates precision@k at document level: fraction of top-k retrieved chunks from expected sources."""
    if k <= 0 or not retrieved_sources:
        return 0.0
    if not expected_sources:
        return 0.0
    norm_expected = {normalize_source_name(s) for s in expected_sources}
    top_k = retrieved_sources[:k]
    relevant_count = sum(1 for s in top_k if normalize_source_name(s) in norm_expected)
    return relevant_count / len(top_k)


def is_refusal_answer(answer: str) -> bool:
    """Detects if LLM correctly refused to answer when context is insufficient."""
    if not answer:
        return False
    refusal_phrases = [
        "couldn't find this information",
        "could not find this information",
        "not in the uploaded documents",
        "not found in the uploaded documents",
        "not mentioned in the provided",
        "not mentioned in the documents",
        "information is not available in the uploaded",
        "does not contain information",
        "no information provided",
    ]
    lower_answer = answer.lower()
    return any(phrase in lower_answer for phrase in refusal_phrases)


def check_citation_in_answer(answer: str, expected_sources: List[str]) -> bool:
    """Checks whether the generated answer text explicitly mentions any source document."""
    if not answer or not expected_sources:
        return False
    lower_answer = answer.lower()
    for s in expected_sources:
        base = normalize_source_name(s).lower()
        # check filename or extension-stripped name
        stem = Path(base).stem.lower()
        if base in lower_answer or (len(stem) > 5 and stem in lower_answer):
            return True
    return False


class RAGEvaluator:
    """
    Evaluates ForgeMind's RAG pipeline against a grounded benchmark dataset.
    """

    def __init__(self, rag_service: Optional[Any] = None, top_k: int = 5):
        self.rag_service = rag_service
        self.top_k = top_k

    def evaluate_case(self, item: Dict[str, Any]) -> EvalResultItem:
        """Evaluates a single evaluation test item."""
        case_id = item.get("id", "UNKNOWN")
        category = item.get("category", "unknown")
        answerability = item.get("answerability", "answerable")
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        expected_sources = item.get("expected_sources", [])

        if not question:
            raise ValueError(f"Evaluation case {case_id} missing required 'question' field")

        t0 = time.time()
        error_msg = None
        generated_answer = ""
        retrieved_docs = []

        for attempt in range(3):
            try:
                if self.rag_service is None:
                    raise RuntimeError("No RAG service configured for evaluation")
                result = self.rag_service.invoke(question)
                generated_answer = result.get("answer", "")
                retrieved_docs = result.get("documents", [])
                error_msg = None
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in str(e) and attempt < 2:
                    time.sleep(5.0)
                    continue
                break

        elapsed = time.time() - t0

        retrieved_sources = []
        for doc in retrieved_docs:
            src = doc.metadata.get("filename") or doc.metadata.get("source") or "Unknown"
            retrieved_sources.append(normalize_source_name(src))

        # Metrics calculation
        if answerability == "unanswerable":
            retrieval_hit = False
            retrieval_recall = 0.0
            context_precision = 0.0
            refusal_detected = is_refusal_answer(generated_answer)
            citation_present = False
        else:
            retrieval_hit = calculate_retrieval_hit(expected_sources, retrieved_sources)
            retrieval_recall = calculate_retrieval_recall(expected_sources, retrieved_sources)
            context_precision = calculate_context_precision(expected_sources, retrieved_sources, k=self.top_k)
            refusal_detected = is_refusal_answer(generated_answer)
            citation_present = check_citation_in_answer(generated_answer, expected_sources)

        # Failure reasons
        failure_reasons = []
        if error_msg:
            failure_reasons.append(f"Pipeline error: {error_msg}")
        elif answerability == "unanswerable":
            if not refusal_detected:
                failure_reasons.append("Unanswerable question was not refused (hallucination risk)")
        else:
            if not retrieval_hit:
                failure_reasons.append("Zero expected sources retrieved (retrieval miss)")
            elif retrieval_recall < 1.0:
                failure_reasons.append(f"Partial retrieval recall ({retrieval_recall:.2f} < 1.0)")
            if refusal_detected:
                failure_reasons.append("False refusal on answerable question")

        passed = len(failure_reasons) == 0

        return EvalResultItem(
            id=case_id,
            category=category,
            answerability=answerability,
            question=question,
            expected_answer=expected_answer,
            expected_sources=expected_sources,
            generated_answer=generated_answer,
            retrieved_sources=retrieved_sources,
            retrieved_chunk_count=len(retrieved_docs),
            latency_seconds=round(elapsed, 4),
            retrieval_hit=retrieval_hit,
            retrieval_recall=round(retrieval_recall, 4),
            context_precision=round(context_precision, 4),
            refusal_detected=refusal_detected,
            citation_present_in_answer=citation_present,
            error=error_msg,
            passed_all_deterministic_checks=passed,
            failure_reasons=failure_reasons,
        )

    def run_evaluation(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs evaluation across all items and generates aggregated metrics."""
        results: List[EvalResultItem] = []
        for item in dataset:
            res = self.evaluate_case(item)
            results.append(res)

        total_cases = len(results)
        answerable_cases = [r for r in results if r.answerability == "answerable"]
        unanswerable_cases = [r for r in results if r.answerability == "unanswerable"]

        # Retrieval metrics (evaluated on answerable cases)
        if answerable_cases:
            hit_rate = sum(1 for r in answerable_cases if r.retrieval_hit) / len(answerable_cases)
            avg_recall = sum(r.retrieval_recall for r in answerable_cases) / len(answerable_cases)
            avg_precision = sum(r.context_precision for r in answerable_cases) / len(answerable_cases)
            citation_accuracy = sum(1 for r in answerable_cases if r.citation_present_in_answer) / len(answerable_cases)
        else:
            hit_rate, avg_recall, avg_precision, citation_accuracy = 0.0, 0.0, 0.0, 0.0

        # Unanswerable metrics
        if unanswerable_cases:
            unanswerable_refusal_rate = sum(1 for r in unanswerable_cases if r.refusal_detected) / len(unanswerable_cases)
            unanswerable_false_context_rate = sum(1 for r in unanswerable_cases if r.retrieved_chunk_count > 0) / len(unanswerable_cases)
        else:
            unanswerable_refusal_rate, unanswerable_false_context_rate = 0.0, 0.0

        # Latency metrics
        latencies = sorted(r.latency_seconds for r in results)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[min(p95_index, len(latencies) - 1)] if latencies else 0.0

        # Category breakdown
        category_stats: Dict[str, Dict[str, Any]] = {}
        categories = sorted(list({r.category for r in results}))
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            cat_ans = [r for r in cat_results if r.answerability == "answerable"]
            cat_hit = sum(1 for r in cat_ans if r.retrieval_hit) / len(cat_ans) if cat_ans else 0.0
            cat_recall = sum(r.retrieval_recall for r in cat_ans) / len(cat_ans) if cat_ans else 0.0
            category_stats[cat] = {
                "count": len(cat_results),
                "answerable_count": len(cat_ans),
                "hit_rate": round(cat_hit, 4),
                "recall": round(cat_recall, 4),
                "pass_count": sum(1 for r in cat_results if r.passed_all_deterministic_checks),
            }

        failed_cases = [asdict(r) for r in results if not r.passed_all_deterministic_checks]

        return {
            "summary": {
                "total_cases": total_cases,
                "answerable_count": len(answerable_cases),
                "unanswerable_count": len(unanswerable_cases),
                "retrieval_hit_rate": round(hit_rate, 4),
                "retrieval_recall": round(avg_recall, 4),
                "context_precision": round(avg_precision, 4),
                "citation_presence_in_answer_rate": round(citation_accuracy, 4),
                "unanswerable_refusal_rate": round(unanswerable_refusal_rate, 4),
                "unanswerable_false_context_rate": round(unanswerable_false_context_rate, 4),
                "avg_latency_seconds": round(avg_latency, 4),
                "p95_latency_seconds": round(p95_latency, 4),
                "total_failures": len(failed_cases),
            },
            "category_breakdown": category_stats,
            "failed_cases": failed_cases,
            "all_results": [asdict(r) for r in results],
        }

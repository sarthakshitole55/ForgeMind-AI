import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

try:
    from tests.evaluation.evaluator import (
        RAGEvaluator,
        calculate_context_precision,
        calculate_retrieval_hit,
        calculate_retrieval_recall,
        check_citation_in_answer,
        is_refusal_answer,
        normalize_source_name,
    )
except ModuleNotFoundError:
    from evaluation.evaluator import (
        RAGEvaluator,
        calculate_context_precision,
        calculate_retrieval_hit,
        calculate_retrieval_recall,
        check_citation_in_answer,
        is_refusal_answer,
        normalize_source_name,
    )


class TestEvaluationHarness(unittest.TestCase):

    def setUp(self):
        self.dataset_path = Path(__file__).parent / "dataset.json"

    def test_dataset_schema(self):
        """Verifies dataset exists, is valid JSON, and adheres strictly to schema."""
        self.assertTrue(self.dataset_path.exists(), "dataset.json must exist")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertGreaterEqual(len(data), 20)
        self.assertLessEqual(len(data), 30)

        required_keys = {"id", "category", "answerability", "question", "expected_answer", "expected_sources"}
        valid_categories = {"easy_factual", "multi_hop", "source_specific", "unanswerable", "ambiguous_or_adversarial"}
        valid_answerability = {"answerable", "unanswerable"}

        for item in data:
            for key in required_keys:
                self.assertIn(key, item, f"Missing key {key} in item {item.get('id')}")
            self.assertIn(item["category"], valid_categories)
            self.assertIn(item["answerability"], valid_answerability)
            self.assertTrue(len(item["question"].strip()) > 0)
            self.assertTrue(len(item["expected_answer"].strip()) > 0)
            self.assertIsInstance(item["expected_sources"], list)

    def test_normalize_source_name(self):
        """Verifies path normalization to clean filename."""
        self.assertEqual(normalize_source_name("/var/data/doc.pdf"), "doc.pdf")
        self.assertEqual(normalize_source_name("data/doc.pdf"), "doc.pdf")
        self.assertEqual(normalize_source_name("doc.pdf"), "doc.pdf")
        self.assertEqual(normalize_source_name(""), "")

    def test_metric_retrieval_hit(self):
        """Tests hit rate calculation logic."""
        self.assertTrue(calculate_retrieval_hit(["a.pdf"], ["a.pdf", "b.pdf"]))
        self.assertTrue(calculate_retrieval_hit(["a.pdf"], ["/path/to/a.pdf"]))
        self.assertFalse(calculate_retrieval_hit(["a.pdf"], ["b.pdf", "c.pdf"]))
        self.assertFalse(calculate_retrieval_hit([], ["b.pdf"]))
        self.assertFalse(calculate_retrieval_hit(["a.pdf"], []))

    def test_metric_retrieval_recall(self):
        """Tests recall calculation across full, partial, and zero matches."""
        # Full match: 2 of 2
        self.assertEqual(calculate_retrieval_recall(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf", "c.pdf"]), 1.0)
        # Partial match: 1 of 2
        self.assertEqual(calculate_retrieval_recall(["a.pdf", "b.pdf"], ["a.pdf", "c.pdf"]), 0.5)
        # Zero match
        self.assertEqual(calculate_retrieval_recall(["a.pdf"], ["b.pdf"]), 0.0)
        # Empty expected
        self.assertEqual(calculate_retrieval_recall([], ["a.pdf"]), 0.0)
        # Empty retrieved
        self.assertEqual(calculate_retrieval_recall(["a.pdf"], []), 0.0)

    def test_metric_context_precision(self):
        """Tests precision@k calculation."""
        retrieved = ["a.pdf", "b.pdf", "a.pdf", "c.pdf", "d.pdf"]
        # Top 5: a.pdf appears twice -> 2/5 = 0.4
        self.assertEqual(calculate_context_precision(["a.pdf"], retrieved, k=5), 0.4)
        # Top 2: a.pdf is 1 of 2 -> 1/2 = 0.5
        self.assertEqual(calculate_context_precision(["a.pdf"], retrieved, k=2), 0.5)
        # Empty expected or invalid k
        self.assertEqual(calculate_context_precision([], retrieved, k=5), 0.0)
        self.assertEqual(calculate_context_precision(["a.pdf"], retrieved, k=0), 0.0)
        self.assertEqual(calculate_context_precision(["a.pdf"], [], k=5), 0.0)

    def test_refusal_detection(self):
        """Tests refusal phrase detection for unanswerable queries."""
        self.assertTrue(is_refusal_answer("I couldn't find this information in the uploaded documents."))
        self.assertTrue(is_refusal_answer("This is not in the uploaded documents."))
        self.assertTrue(is_refusal_answer("The information is not available in the uploaded manuals."))
        self.assertFalse(is_refusal_answer("Marcos Lopez de Prado explains that the Marcenko-Pastur theorem denoises matrices."))
        self.assertFalse(is_refusal_answer(""))

    def test_citation_detection(self):
        """Tests citation presence detection in answers."""
        answer = "As described in Machine_Learning_for_Asset_Managers.pdf, the ONC algorithm clusters matrices."
        self.assertTrue(check_citation_in_answer(answer, ["Machine_Learning_for_Asset_Managers.pdf"]))
        self.assertFalse(check_citation_in_answer(answer, ["fastai_deep_learning.pdf"]))
        self.assertFalse(check_citation_in_answer("", ["doc.pdf"]))

    def test_empty_retrieval_handling(self):
        """Ensures evaluator gracefully handles zero documents retrieved from RAG."""
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {"answer": "I couldn't find this information in the uploaded documents.", "documents": []}

        evaluator = RAGEvaluator(rag_service=mock_rag)
        item = {
            "id": "TEST-01",
            "category": "easy_factual",
            "answerability": "answerable",
            "question": "What is X?",
            "expected_answer": "X is Y.",
            "expected_sources": ["manual.pdf"],
        }
        res = evaluator.evaluate_case(item)
        self.assertFalse(res.retrieval_hit)
        self.assertEqual(res.retrieval_recall, 0.0)
        self.assertEqual(res.retrieved_chunk_count, 0)
        self.assertIn("Zero expected sources retrieved", res.failure_reasons[0])

    def test_unanswerable_cases_handling(self):
        """Ensures unanswerable questions verify refusal and zero expected source handling."""
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "I couldn't find this information in the uploaded documents.",
            "documents": [Document(page_content="Unrelated text", metadata={"source": "other.pdf"})],
        }

        evaluator = RAGEvaluator(rag_service=mock_rag)
        item = {
            "id": "TEST-UN",
            "category": "unanswerable",
            "answerability": "unanswerable",
            "question": "What is the secret recipe?",
            "expected_answer": "I couldn't find this information in the uploaded documents.",
            "expected_sources": [],
        }
        res = evaluator.evaluate_case(item)
        self.assertTrue(res.refusal_detected)
        self.assertTrue(res.passed_all_deterministic_checks)

    def test_malformed_evaluation_records(self):
        """Ensures evaluator raises ValueError if required question field is missing."""
        evaluator = RAGEvaluator(rag_service=MagicMock())
        with self.assertRaises(ValueError):
            evaluator.evaluate_case({"id": "BAD-01", "question": ""})

    def test_mock_eval_run_offline(self):
        """Runs complete offline evaluation with mock RAG service and asserts summary metrics."""
        mock_rag = MagicMock()
        mock_rag.invoke.side_effect = [
            # Item 1: Hit and answered
            {
                "answer": "Marcos Lopez de Prado authored the book according to Machine_Learning_for_Asset_Managers.pdf",
                "documents": [Document(page_content="Content", metadata={"source": "doc1.pdf"})],
            },
            # Item 2: Unanswerable correctly refused
            {
                "answer": "I couldn't find this information in the uploaded documents.",
                "documents": [Document(page_content="Noise", metadata={"source": "noise.pdf"})],
            },
        ]

        test_dataset = [
            {
                "id": "MOCK-01",
                "category": "easy_factual",
                "answerability": "answerable",
                "question": "Who authored the book?",
                "expected_answer": "Marcos",
                "expected_sources": ["doc1.pdf"],
            },
            {
                "id": "MOCK-02",
                "category": "unanswerable",
                "answerability": "unanswerable",
                "question": "What is the recipe for cake?",
                "expected_answer": "I couldn't find this information in the uploaded documents.",
                "expected_sources": [],
            },
        ]

        evaluator = RAGEvaluator(rag_service=mock_rag)
        results = evaluator.run_evaluation(test_dataset)

        self.assertEqual(results["summary"]["total_cases"], 2)
        self.assertEqual(results["summary"]["retrieval_hit_rate"], 1.0)
        self.assertEqual(results["summary"]["unanswerable_refusal_rate"], 1.0)
        self.assertEqual(len(results["all_results"]), 2)


if __name__ == "__main__":
    unittest.main()

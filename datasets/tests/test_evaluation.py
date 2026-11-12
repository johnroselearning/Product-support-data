import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from evaluation.evaluate import evaluate
from evaluation.validation import validate_case


class EvaluationTests(unittest.TestCase):
    def approved_case(self, case_id, label):
        return {
            "case_id": case_id,
            "task": "classification",
            "input": {"ticket": "The service fails to start.", "context": None},
            "gold": {
                "labels": {"category": label},
                "root_cause": None,
                "resolution": None,
                "must_include": ["restart service"],
                "must_not": ["delete database"],
                "expected_tools": ["search_logs"],
                "answerable": True
            },
            "tags": ["classification", "test"],
            "provenance": {
                "source": "it_support",
                "source_case_id": case_id,
                "source_revision": "main",
                "license": "cc-by-4.0",
                "source_split": "test"
            },
            "selection": {
                "method": "unit_test",
                "stratum": "classification|category|Software",
                "seed": 1
            },
            "review": {
                "status": "approved",
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-11-12T12:00:00Z",
                "notes": "Reviewed fixture."
            }
        }

    def test_deterministic_metrics(self):
        gold = [self.approved_case("a", "Software")]
        predictions = [{
            "case_id": "a",
            "labels": {"category": "Software"},
            "answerable": True,
            "answer": "Restart service after checking logs.",
            "tool_calls": ["search_logs"]
        }]
        report = evaluate(gold, predictions)
        self.assertEqual(report["overall_label_exact_match_accuracy"], 1.0)
        self.assertEqual(report["answerability"]["accuracy"], 1.0)
        self.assertEqual(report["tool_use"]["f1"], 1.0)
        self.assertEqual(report["required_fact_checks"]["compliance"], 1.0)
        self.assertEqual(report["forbidden_fact_checks"]["violation_rate"], 0.0)

    def test_case_schema_rejects_missing_required_input(self):
        case = self.approved_case("a", "Software")
        del case["input"]
        self.assertTrue(validate_case(case))

    def test_case_schema_rejects_approved_case_without_reviewer(self):
        case = self.approved_case("a", "Software")
        case["review"]["reviewer"] = None
        self.assertTrue(validate_case(case))

    def test_pending_case_is_rejected(self):
        gold = [self.approved_case("a", "Software")]
        gold[0]["review"]["status"] = "pending"
        with self.assertRaises(ValueError):
            evaluate(gold, [{"case_id": "a", "labels": {"category": "Software"}}])


if __name__ == "__main__":
    unittest.main()

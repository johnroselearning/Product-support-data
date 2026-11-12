import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest

from evaluation.promote_eval import promote


class PromoteEvaluationTests(unittest.TestCase):
    def make_case(self, status="approved", split="test"):
        return {
            "case_id": "case-1",
            "task": "classification",
            "input": {
                "ticket": "VPN login fails after password reset.",
                "context": None
            },
            "gold": {
                "labels": {"category": "Software"},
                "root_cause": None,
                "resolution": None,
                "must_include": [],
                "must_not": [],
                "expected_tools": [],
                "answerable": None
            },
            "tags": ["unit_test", "needs_human_review"],
            "provenance": {
                "source": "unit_test",
                "source_case_id": "source-case-1",
                "source_revision": "fixture-v1",
                "license": "unknown",
                "source_split": split
            },
            "selection": {
                "method": "unit_test",
                "stratum": "classification|category|Software",
                "seed": 20261112
            },
            "review": {
                "status": status,
                "reviewer": "human-reviewer",
                "reviewed_at": "2026-11-12T12:00:00Z",
                "notes": "Complete schema-valid test fixture."
            }
        }

    def test_promotes_approved_case_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "queue.jsonl"
            output = root / "eval_v1.jsonl"
            manifest = root / "eval_v1_manifest.json"
            source.write_text(
                json.dumps(self.make_case()) + "\n",
                encoding="utf-8"
            )
            result = promote(source, output, manifest, "test")
            self.assertTrue(output.exists())
            self.assertTrue(manifest.exists())
            self.assertEqual(result["case_count"], 1)
            self.assertEqual(len(result["sha256"]), 64)

    def test_pending_case_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "queue.jsonl"
            source.write_text(
                json.dumps(self.make_case("pending")) + "\n",
                encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not approved"):
                promote(source, root / "eval_v1.jsonl",
                        root / "manifest.json", "test")

    def test_existing_version_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "queue.jsonl"
            source.write_text(
                json.dumps(self.make_case()) + "\n",
                encoding="utf-8"
            )
            output = root / "eval_v1.jsonl"
            output.write_text("released\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                promote(source, output, root / "manifest.json", "test")


if __name__ == "__main__":
    unittest.main()

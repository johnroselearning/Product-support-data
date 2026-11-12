"""Backward-compatible tests for the modular pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from adapters.common import digest, build_record
from quality.privacy import detect_flags
from normalization.sft import to_sft
class PrepareTests(unittest.TestCase):
    def test_fingerprint_normalizes_whitespace(self):
        self.assertEqual(digest(" HELLO   world "), digest("hello world"))
    def test_privacy_email(self):
        self.assertIn("email", detect_flags("contact test@example.com"))
    def test_record_keeps_unknown_cause(self):
        source={"id":"test","license":"cc-by-4.0"}
        rec=build_record(source,"train",0,"HTTP 500","classification",{"category":"Software"},"g1")
        self.assertEqual(rec["root_cause"],"unknown")
        self.assertEqual(rec["evidence"],[])
    def test_sft_target(self):
        source={"id":"it_support","license":"cc-by-4.0"}
        rec=build_record(source,"train",0,"printer offline","classification",{"category":"Hardware"},"g1")
        self.assertEqual(to_sft(rec)["messages"][-1]["content"],'{"category": "Hardware"}')
if __name__ == "__main__": unittest.main()

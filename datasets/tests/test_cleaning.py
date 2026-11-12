import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from cleaning.clean import clean_record, clean_text


class CleaningTests(unittest.TestCase):
    def make_record(self, description):
        return {
            "source": "unit_test",
            "task": "classification",
            "upstream_split": "train",
            "upstream_row": 1,
            "ticket": {"description": description},
            "labels": {"priority_source": "Highest", "category": "Software"}
        }

    def clean_regex_only(self, record):
        # Unit tests exercise deterministic regex behavior without depending on
        # whether the optional spaCy model is installed on the developer machine.
        # Production ingestion continues to require NER unless explicitly opted out.
        return clean_record(record, allow_regex_only=True)

    def test_trim_and_line_endings(self):
        self.assertEqual(clean_text("  hello\r\nworld  "), "hello\nworld")

    def test_preserves_label_identity(self):
        cleaned, errors, redaction_counts = self.clean_regex_only(
            self.make_record("  login issue ")
        )
        self.assertFalse(errors)
        self.assertEqual(cleaned["labels"]["priority_source"], "Highest")
        self.assertEqual(cleaned["labels"]["category"], "Software")
        self.assertEqual(redaction_counts, {})

    def test_does_not_drop_short_ticket(self):
        cleaned, errors, _ = self.clean_regex_only(
            self.make_record("VPN down")
        )
        self.assertFalse(errors)
        self.assertEqual(cleaned["ticket"]["description"], "VPN down")

    def test_redacts_email_before_returning_cleaned_record(self):
        cleaned, errors, redaction_counts = self.clean_regex_only(
            self.make_record("email me user@example.com")
        )
        self.assertFalse(errors)
        self.assertNotIn("user@example.com", cleaned["ticket"]["description"])
        self.assertIn("[REDACTED_PII]", cleaned["ticket"]["description"])
        self.assertEqual(redaction_counts, {"email": 1})

    def test_empty_text_rejected(self):
        cleaned, errors, redaction_counts = self.clean_regex_only(
            self.make_record("  \x00 ")
        )
        self.assertIsNone(cleaned)
        self.assertIn("empty_after_trim", errors)
        self.assertEqual(redaction_counts, {})


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from quality.privacy import redact_text, privacy_capabilities


class PrivacyTests(unittest.TestCase):
    def test_regex_redaction_removes_sensitive_values(self):
        text = "email: user@example.com ip: 10.0.0.1 token: Bearer abcdefghijklmnop"
        safe, flags, counts = redact_text(text, require_ner=False)
        self.assertNotIn("user@example.com", safe)
        self.assertNotIn("10.0.0.1", safe)
        self.assertNotIn("abcdefghijklmnop", safe)
        self.assertIn("email", flags)
        self.assertIn("ipv4", flags)
        self.assertIn("bearer_token", flags)

    def test_capabilities_report_ner_availability(self):
        result = privacy_capabilities()
        self.assertTrue(result["regex_screening"])
        self.assertIn("person_and_location_ner", result)


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from cleaning.clean import clean_record,clean_text
class CleaningTests(unittest.TestCase):
    def test_trim_and_line_endings(self):
        self.assertEqual(clean_text("  hello\r\nworld  "),"hello\nworld")
    def test_preserves_label_identity(self):
        rec={"ticket":{"description":"  login issue "},"labels":{"priority_source":"Highest"}}
        cleaned,errors=clean_record(rec)
        self.assertFalse(errors)
        self.assertEqual(cleaned["labels"]["priority_source"],"Highest")
    def test_does_not_drop_short_ticket(self):
        rec={"ticket":{"description":"VPN down"},"labels":{"category":"Software"}}
        cleaned,errors=clean_record(rec)
        self.assertFalse(errors)
        self.assertEqual(cleaned["ticket"]["description"],"VPN down")
    def test_quarantines_email(self):
        rec={"ticket":{"description":"email me user@example.com"},"labels":{"category":"Software"}}
        cleaned,errors=clean_record(rec)
        self.assertIsNone(cleaned)
        self.assertIn("privacy:email",errors)
    def test_empty_text_rejected(self):
        rec={"ticket":{"description":"  \x00 "},"labels":{"category":"Software"}}
        cleaned,errors=clean_record(rec)
        self.assertIsNone(cleaned)
        self.assertIn("empty_after_trim",errors)
if __name__=="__main__": unittest.main()

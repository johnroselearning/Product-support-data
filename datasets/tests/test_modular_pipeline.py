import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from adapters.registry import get_adapter
from splitting.group_split import assign_records
from quality.privacy import detect_flags
class ModularPipelineTests(unittest.TestCase):
    def test_adapter_registry(self):
        self.assertTrue(callable(get_adapter("it_support")))
        self.assertTrue(callable(get_adapter("helpdesk")))
        with self.assertRaises(ValueError): get_adapter("unknown")
    def test_test_group_is_held_out(self):
        rows=[{"source":"s","source_group":"g","upstream_split":"train"},{"source":"s","source_group":"g","upstream_split":"test"}]
        self.assertEqual(assign_records(rows)[("s","g")],"test")
    def test_privacy_detector(self):
        self.assertIn("email",detect_flags("contact me user@example.com"))
    def test_it_support_adapter(self):
        src={"id":"it_support","license":"cc-by-4.0","attribution":"test"}
        row=get_adapter("it_support")(src,{"text":"printer offline","label":"Hardware"},{},"train",0)
        self.assertEqual(row["labels"],{"category":"Hardware"})
        self.assertEqual(row["root_cause"],"unknown")
    def test_helpdesk_adapter(self):
        src={"id":"helpdesk","license":"cc-by-4.0","attribution":"test"}
        row=get_adapter("helpdesk")(src,{"issue_id":"42","text":"login issue","issue_priority":"High"},"","train",0)
        self.assertEqual(row["source_group"],"42")
        self.assertEqual(row["labels"],{"priority_source":"High"})
if __name__ == "__main__": unittest.main()

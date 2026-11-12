import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from adapters.registry import get_adapter
from adapters.common import build_record
from splitting.group_split import assign_records
from quality.privacy import detect_flags
from prepare import deduplicate_records

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
    def test_case_ids_unique_for_same_text_in_different_source_rows(self):
        source={"id":"it_support","license":"cc-by-4.0"}
        first=build_record(source,"train",3,"same text","classification",{"category":"A"},"g")
        second=build_record(source,"test",7,"same text","classification",{"category":"A"},"g")
        self.assertNotEqual(first["case_id"],second["case_id"])
    def test_conflicting_duplicate_labels_quarantine_all(self):
        base={"source":"s","fingerprint":"fp","upstream_split":"train","upstream_row":0,
              "case_id":"s-train-0","labels":{"category":"A"}}
        other={**base,"upstream_split":"test","upstream_row":2,"case_id":"s-test-2",
               "labels":{"category":"B"}}
        chosen,audit,quarantine,stats=deduplicate_records([base,other])
        self.assertEqual(chosen,[])
        self.assertEqual(len(quarantine),2)
        self.assertEqual(audit[0]["decision"],"quarantine_all_candidates_pending_review")
        self.assertEqual(stats["duplicate_label_conflict_groups"],1)
    def test_same_label_duplicates_keep_publisher_test_copy(self):
        base={"source":"s","fingerprint":"fp","upstream_split":"train","upstream_row":0,
              "case_id":"s-train-0","labels":{"category":"A"}}
        other={**base,"upstream_split":"test","upstream_row":2,"case_id":"s-test-2"}
        chosen,audit,quarantine,stats=deduplicate_records([base,other])
        self.assertEqual([r["case_id"] for r in chosen],["s-test-2"])
        self.assertEqual(len(audit),1)
        self.assertEqual(quarantine,[])
if __name__ == "__main__": unittest.main()

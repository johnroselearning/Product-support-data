import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from adapters.registry import get_adapter
from adapters.common import build_record
from splitting.group_split import assign_records
from quality.privacy import detect_flags, redact_text
from prepare import deduplicate_records


class ModularPipelineTests(unittest.TestCase):
    def test_adapter_registry(self):
        self.assertTrue(callable(get_adapter("it_support")))
        self.assertTrue(callable(get_adapter("helpdesk")))
        with self.assertRaises(ValueError):
            get_adapter("unknown")

    def test_test_group_is_held_out(self):
        rows = [
            {"source": "source-a", "source_group": "g", "split_group": "same",
             "upstream_split": "train"},
            {"source": "source-b", "source_group": "h", "split_group": "same",
             "upstream_split": "test"}
        ]
        assignments = assign_records(rows)
        self.assertEqual(assignments[("source-a", "same")], "test")
        self.assertEqual(assignments[("source-b", "same")], "test")


    def test_all_messages_from_same_incident_stay_in_one_split(self):
        rows = [
            {"source": "helpdesk", "source_group": "issue-hash",
             "source_group_type": "incident_id", "split_group": "fp-1",
             "upstream_split": "train"},
            {"source": "helpdesk", "source_group": "issue-hash",
             "source_group_type": "incident_id", "split_group": "fp-2",
             "upstream_split": "validation"}
        ]
        assignments = assign_records(rows)
        self.assertEqual(
            assignments[("helpdesk", "fp-1")],
            assignments[("helpdesk", "fp-2")]
        )

    def test_privacy_detector(self):
        flags = detect_flags("contact me user@example.com")
        self.assertIn("email", flags)

    def test_privacy_redacts_email_ip_account_and_secret(self):
        text = (
            "Contact jane@example.com from 192.168.1.10; "
            "account id: CUST-12345; api_key=sk_test_1234567890"
        )
        safe, flags, counts = redact_text(text, require_ner=False)
        self.assertNotIn("jane@example.com", safe)
        self.assertNotIn("192.168.1.10", safe)
        self.assertNotIn("CUST-12345", safe)
        self.assertNotIn("sk_test_1234567890", safe)
        self.assertTrue({"email", "ipv4", "account_identifier", "secret_assignment"}.issubset(flags))
        self.assertGreaterEqual(sum(counts.values()), 4)

    def test_it_support_adapter_uses_text_fingerprint_as_identity_hint(self):
        src = {"id": "it_support", "license": "cc-by-4.0", "attribution": "test"}
        row = get_adapter("it_support")(
            src, {"text": "printer offline", "label": "Hardware"}, {}, "train", 0
        )
        self.assertEqual(row["labels"], {"category": "Hardware"})
        self.assertEqual(row["root_cause"], "unknown")
        self.assertEqual(row["source_group_type"], "text_fingerprint")
        self.assertEqual(row["split_group"], row["fingerprint"])

    def test_helpdesk_incident_id_is_hashed(self):
        src = {"id": "helpdesk", "license": "cc-by-4.0", "attribution": "test"}
        row = get_adapter("helpdesk")(
            src, {"issue_id": "42", "text": "login issue", "issue_priority": "High"},
            "", "train", 0
        )
        self.assertNotEqual(row["source_group"], "42")
        self.assertEqual(row["source_group_type"], "incident_id")
        self.assertEqual(row["labels"], {"priority_source": "High"})

    def test_case_ids_unique_for_same_text_in_different_source_rows(self):
        source = {"id": "it_support", "license": "cc-by-4.0"}
        first = build_record(source, "train", 3, "same text", "classification",
                             {"category": "A"}, "same text", "text_fingerprint")
        second = build_record(source, "test", 7, "same text", "classification",
                              {"category": "A"}, "same text", "text_fingerprint")
        self.assertNotEqual(first["case_id"], second["case_id"])
        self.assertEqual(first["split_group"], second["split_group"])

    def test_conflicting_duplicate_labels_quarantine_all(self):
        base = {
            "source": "s", "fingerprint": "fp", "task": "classification", "upstream_split": "train",
            "upstream_row": 0, "case_id": "s-train-0",
            "source_group": "g1", "source_group_type": "incident_id",
            "labels": {"category": "A"}
        }
        other = {
            **base, "upstream_split": "test", "upstream_row": 2,
            "case_id": "s-test-2", "source_group": "g2",
            "labels": {"category": "B"}
        }
        chosen, audit, quarantine, stats = deduplicate_records([base, other])
        self.assertEqual(chosen, [])
        self.assertEqual(len(quarantine), 2)
        self.assertEqual(audit[0]["decision"], "quarantine_all_candidates_pending_review")
        self.assertEqual(stats["duplicate_label_conflict_groups"], 1)

    def test_same_identity_exact_duplicates_collapse(self):
        base = {
            "source": "s", "fingerprint": "fp", "upstream_split": "train",
            "upstream_row": 0, "case_id": "s-train-0",
            "source_group": "incident-1", "source_group_type": "incident_id",
            "labels": {"category": "A"}
        }
        other = {
            **base, "upstream_split": "test", "upstream_row": 2,
            "case_id": "s-test-2"
        }
        chosen, audit, quarantine, stats = deduplicate_records([base, other])
        self.assertEqual([r["case_id"] for r in chosen], ["s-test-2"])
        self.assertEqual(len(audit), 1)
        self.assertEqual(quarantine, [])

 
    def test_cross_source_conflicting_exact_inputs_are_quarantined(self):
        base = {
            "source": "source-a", "fingerprint": "same-fp", "task": "classification",
            "upstream_split": "train", "upstream_row": 0, "case_id": "a-0",
            "source_group": "incident-a", "source_group_type": "incident_id",
            "labels": {"category": "A"}
        }
        other = {
            **base, "source": "source-b", "case_id": "b-0",
            "source_group": "incident-b", "labels": {"category": "B"}
        }
        chosen, audit, quarantine, stats = deduplicate_records([base, other])
        self.assertEqual(chosen, [])
        self.assertEqual(len(quarantine), 2)
        self.assertEqual(audit[0]["decision"], "quarantine_all_exact_input_candidates_across_sources")

    def test_distinct_incidents_with_same_text_are_preserved(self):
        base = {
            "source": "s", "fingerprint": "fp", "upstream_split": "train",
            "upstream_row": 0, "case_id": "s-train-0",
            "source_group": "incident-1", "source_group_type": "incident_id",
            "labels": {"category": "A"}
        }
        other = {
            **base, "upstream_row": 1, "case_id": "s-train-1",
            "source_group": "incident-2"
        }
        chosen, audit, quarantine, stats = deduplicate_records([base, other])
        self.assertEqual(len(chosen), 2)
        self.assertEqual(audit[0]["decision"], "preserve_same_text_distinct_incident_identities")
        self.assertEqual(quarantine, [])


if __name__ == "__main__":
    unittest.main()

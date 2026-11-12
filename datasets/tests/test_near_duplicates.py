import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from quality.near_duplicates import find_near_duplicates


class NearDuplicateTests(unittest.TestCase):
    def test_similar_texts_are_flagged_but_not_removed(self):
        rows = [
            {"case_id": "a", "source": "s", "split": "train",
             "fingerprint": "1", "labels": {"category": "A"},
             "text": "user cannot log in to the corporate vpn after password reset"},
            {"case_id": "b", "source": "s", "split": "test",
             "fingerprint": "2", "labels": {"category": "A"},
             "text": "user cannot log in to corporate vpn after a password reset"}
        ]
        pairs, _ = find_near_duplicates(rows, threshold=0.65)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["cross_split"])
        self.assertEqual(pairs[0]["decision"], "human_review_required_no_automatic_removal")


if __name__ == "__main__":
    unittest.main()

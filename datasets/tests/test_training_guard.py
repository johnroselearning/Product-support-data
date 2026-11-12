import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
from training.guard import assert_safe_training_input


class TrainingGuardTests(unittest.TestCase):
    def test_allows_train_sft_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft" / "train.jsonl"
            path.parent.mkdir()
            path.write_text("{}\n")
            self.assertEqual(assert_safe_training_input(path, "train"), path)

    def test_rejects_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft" / "test.jsonl"
            path.parent.mkdir()
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                assert_safe_training_input(path, "train")

    def test_rejects_eval_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation" / "train.jsonl"
            path.parent.mkdir()
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                assert_safe_training_input(path, "train")


if __name__ == "__main__":
    unittest.main()

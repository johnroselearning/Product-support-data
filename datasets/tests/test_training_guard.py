import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
from training.guard import assert_safe_training_input, DATASETS_ROOT


class TrainingGuardTests(unittest.TestCase):
    def test_allows_train_sft_file(self):
        base = DATASETS_ROOT / "cleaned" / "sft"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text("{}\n")
            self.assertEqual(assert_safe_training_input(path, "train"), path.resolve())

    def test_rejects_test_file(self):
        base = DATASETS_ROOT / "cleaned" / "sft"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            path = Path(tmp) / "test.jsonl"
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                assert_safe_training_input(path, "train")

    def test_rejects_eval_file(self):
        base = DATASETS_ROOT / "evaluation"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                assert_safe_training_input(path, "train")


if __name__ == "__main__":
    unittest.main()

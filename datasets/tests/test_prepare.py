import importlib.util
import pathlib
import unittest
path = pathlib.Path(__file__).resolve().parents[1] / "prepare.py"
spec = importlib.util.spec_from_file_location("prepare", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class Tests(unittest.TestCase):
    def test_fingerprint(self):
        self.assertEqual(module.digest(" HELLO   world "), module.digest("hello world"))
    def test_pii(self):
        self.assertIn("email", module.pii("hello test@example.com"))
    def test_split(self):
        self.assertEqual(module.split_for(module.digest("example")), module.split_for(module.digest("example")))
    def test_unknown_root_cause(self):
        row = module.normalize("it_support", {"text": "HTTP 500", "label": "api"}, {}, "train", 0)
        self.assertEqual(row["root_cause"], "unknown")
if __name__ == "__main__":
    unittest.main()

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
        row = module.normalize("it_support", {"text": "HTTP 500", "label": "Software"}, {}, "train", 0)
        self.assertEqual(row["root_cause"], "unknown")
        self.assertEqual(row["labels"]["category"], "Software")
        self.assertEqual(row["evidence"], [])
    def test_helpdesk_priority(self):
        row = module.normalize("helpdesk", {"issue_id":"ISS-1","text":"Login failed","issue_priority":"High"}, {}, "train", 0)
        self.assertEqual(row["source_group"], "train:ISS-1")
        self.assertEqual(row["labels"], {"priority_source": "High"})
        self.assertNotIn("category", row["labels"])
    def test_sft(self):
        row = module.normalize("it_support", {"text":"Printer fails","label":"Computer-Services"}, {}, "train", 0)
        self.assertEqual(module.to_sft(row)["messages"][-1]["content"], '{"category": "Computer-Services"}')
if __name__ == "__main__":
    unittest.main()

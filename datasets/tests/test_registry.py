import json
from pathlib import Path
import unittest

REGISTRY = Path(__file__).resolve().parents[1] / "sources.json"

class RegistryTests(unittest.TestCase):
    def test_registry_has_expected_sources(self):
        sources = json.loads(REGISTRY.read_text())["sources"]
        ids = [s["id"] for s in sources]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({"it_support","polaris","glaive","swe_extra","bitext","tobi","itbench","loghub"}.issubset(ids))
    def test_noncommercial_sources_not_enabled(self):
        sources = json.loads(REGISTRY.read_text())["sources"]
        for s in sources:
            if "nc" in s["license"].lower():
                self.assertNotEqual(s["status"], "enabled")
    def test_default_adapter_is_supported(self):
        sources = json.loads(REGISTRY.read_text())["sources"]
        self.assertEqual([s["id"] for s in sources if s["status"] == "enabled"], ["it_support"])

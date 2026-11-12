import json
from pathlib import Path
import unittest
REGISTRY = Path(__file__).resolve().parents[1] / "sources.json"
class RegistryTests(unittest.TestCase):
    def test_registry_has_expected_sources(self):
        sources = json.loads(REGISTRY.read_text())["sources"]
        ids = [s["id"] for s in sources]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({"it_support","helpdesk","polaris","glaive","swe_extra","bitext","tobi","itbench","loghub","infra_ops","incident_management_events","planning2interaction"}.issubset(ids))
    def test_noncommercial_sources_not_enabled(self):
        for s in json.loads(REGISTRY.read_text())["sources"]:
            if "-nc-" in s["license"].lower():
                self.assertNotEqual(s["status"], "enabled")
    def test_enabled_adapters_supported(self):
        enabled = {s["id"] for s in json.loads(REGISTRY.read_text())["sources"] if s["status"] == "enabled"}
        self.assertEqual(enabled, {"it_support", "helpdesk"})

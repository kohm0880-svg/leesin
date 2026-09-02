from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import v4_mvp.module_workshop as workshop


class ModuleWorkshopTests(unittest.TestCase):
    def test_inspects_function_signature(self):
        info = workshop.inspect_python('def average(values, scale=1):\n    """Average values."""\n    return sum(values) / len(values) * scale\n')
        self.assertEqual(info["recommendedFunction"], "average")
        fn = info["functions"][0]
        self.assertEqual(fn["docstring"], "Average values.")
        self.assertEqual([p["name"] for p in fn["parameters"]], ["values", "scale"])
        self.assertTrue(fn["parameters"][0]["required"])
        self.assertFalse(fn["parameters"][1]["required"])

    def test_parses_excel_style_tsv(self):
        table = workshop.parse_table("N\truntime_ms\n5\t0.1\n10\t0.2\n")
        self.assertEqual(table["columns"], ["N", "runtime_ms"])
        self.assertEqual(table["rowCount"], 2)
        self.assertEqual(table["rows"][0]["N"], 5)

    def test_suggests_mapping_and_whole_table(self):
        code = "def summarize(runtime_ms, rows):\n    return len(rows)\n"
        prepared = workshop.prepare_workshop(code, "runtime_ms,algorithm\n1.0,a\n2.0,b\n")
        self.assertEqual(prepared["suggestedMapping"]["runtime_ms"], "runtime_ms")
        self.assertEqual(prepared["suggestedMapping"]["rows"], "__rows__")

    def test_runs_simple_function(self):
        result = workshop.run_workshop(
            "def average(values):\n    return sum(values) / len(values)\n",
            "average",
            "values\n1\n2\n3\n",
            {"values": "values"},
        )
        self.assertEqual(result["result"], 2.0)
        self.assertEqual(result["rowCount"], 3)

    def test_allows_math_import(self):
        result = workshop.run_workshop(
            "import math\ndef roots(values):\n    return [math.sqrt(v) for v in values]\n",
            "roots",
            "values\n1\n4\n9\n",
            {"values": "values"},
        )
        self.assertEqual(result["result"], [1.0, 2.0, 3.0])

    def test_allows_safe_import_alias_and_from_import(self):
        result = workshop.run_workshop(
            "import statistics as stats\nfrom math import sqrt as root\ndef summary(values):\n    return [stats.mean(values), root(9)]\n",
            "summary",
            "values\n1\n2\n3\n",
            {"values": "values"},
        )
        self.assertEqual(result["result"], [2, 3.0])

    def test_rejects_imports(self):
        with self.assertRaisesRegex(ValueError, "Import"):
            workshop.run_workshop(
                "import os\ndef f(values):\n    return values\n",
                "f",
                "values\n1\n",
                {"values": "values"},
            )

    def test_saves_module(self):
        original = workshop.MODULE_STORE_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workshop.MODULE_STORE_PATH = Path(tmp) / "modules.json"
                saved = workshop.save_module(
                    code="def average(values):\n    return sum(values) / len(values)\n",
                    function_name="average",
                )
                self.assertEqual(saved["title"], "average")
                self.assertEqual(len(workshop.list_saved_modules()), 1)
        finally:
            workshop.MODULE_STORE_PATH = original


if __name__ == "__main__":
    unittest.main()

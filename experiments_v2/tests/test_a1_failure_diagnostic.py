from __future__ import annotations

import unittest

import numpy as np

from experiments_v2.diagnostics.a1_failure import (
    describe,
    failure_distribution,
    select_failures,
)


class A1FailureDiagnosticTests(unittest.TestCase):
    def test_selects_one_per_class_and_prefers_preserved_exact_error(self):
        missing = [
            {"split": "train", "category": "low", "key": "low-first", "csv_line": 1},
            {
                "split": "train",
                "category": "low",
                "key": "low-exact",
                "csv_line": 2,
                "reported_error": "IndexError: evidence",
            },
            {"split": "val", "category": "mid", "key": "mid", "csv_line": 3},
            {"split": "test", "category": "high", "key": "high", "csv_line": 4},
        ]
        self.assertEqual(
            [item["key"] for item in select_failures(missing)],
            ["low-exact", "mid", "high"],
        )

    def test_distribution_is_split_and_category_aware(self):
        missing = [
            {"split": "train", "category": "low"},
            {"split": "train", "category": "high"},
            {"split": "test", "category": "low"},
        ]
        result = failure_distribution(missing)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["by_split"], {"test": 1, "train": 2})
        self.assertEqual(result["by_category"], {"high": 1, "low": 2})
        self.assertEqual(result["by_split_category"]["val"]["mid"], 0)

    def test_describe_records_array_structure_without_values(self):
        array = np.ones((1, 8), dtype=np.float32)
        result = describe(array)
        self.assertEqual(result["shape"], [1, 8])
        self.assertEqual(result["dtype"], "float32")
        self.assertNotIn("value", result)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments_v2.certification.preflight import _v2_feature_artifact_check


class V2PreflightFeatureTests(unittest.TestCase):
    def test_recognizes_only_complete_immutable_v2_feature(self):
        records = [("train", "videos/Low/view1.mp4", 0, "view1", "low")]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = (
                root
                / "features"
                / "affect"
                / "METHOD_A1"
                / "MODEL_TEST"
                / "FEATURE_TEST"
            )
            data = artifact / "data"
            (data / "train" / "low").mkdir(parents=True)
            (data / "train" / "low" / "view1.npy").write_bytes(b"inventory-only")
            (data / "extraction_manifest.json").write_text("{}\n", encoding="utf-8")
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "category": "affect",
                        "method_id": "METHOD_A1",
                        "model_id": "MODEL_TEST",
                        "feature_id": "FEATURE_TEST",
                        "feature_dim": 8,
                        "shape_per_video": [8, 8],
                        "validated_files": 1,
                    }
                ),
                encoding="utf-8",
            )

            result = _v2_feature_artifact_check(
                root, records, category="affect", method_id="METHOD_A1"
            )
            self.assertTrue(result["ready"])
            self.assertEqual(result["source_type"], "v2_immutable_feature_artifact")
            self.assertEqual(result["feature_id"], "FEATURE_TEST")

    def test_rejects_incomplete_manifest(self):
        records = [("train", "videos/Low/view1.mp4", 0, "view1", "low")]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = (
                root
                / "features"
                / "affect"
                / "METHOD_A1"
                / "MODEL_TEST"
                / "FEATURE_TEST"
            )
            artifact.mkdir(parents=True)
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "category": "affect",
                        "method_id": "METHOD_A1",
                        "validated_files": 1,
                    }
                ),
                encoding="utf-8",
            )
            result = _v2_feature_artifact_check(
                root, records, category="affect", method_id="METHOD_A1"
            )
            self.assertFalse(result["ready"])
            self.assertEqual(len(result["rejected_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()

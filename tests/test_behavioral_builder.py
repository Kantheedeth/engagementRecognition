from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.feature_schema import (
    AFFECT_COLUMNS,
    AFFECT_FEATURE_SCHEMA,
    BEHAVIORAL_FEATURE_SCHEMA,
    TRACK_INTERACTION_COLUMNS,
    TRACK_INTERACTION_FEATURE_SCHEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BehavioralBuilderTests(unittest.TestCase):
    def test_builds_versioned_48_column_matrices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_dir = root / "features"
            output_dir = root / "matrices"
            interaction_root = feature_dir / "interaction_track_features"
            affect_root = feature_dir / "affect_track_features"
            interaction_root.mkdir(parents=True)
            affect_root.mkdir(parents=True)
            (interaction_root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "feature_schema": TRACK_INTERACTION_FEATURE_SCHEMA,
                        "shape_per_video": [8, 40],
                        "columns": list(TRACK_INTERACTION_COLUMNS),
                    }
                ),
                encoding="utf-8",
            )
            (affect_root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "feature_schema": AFFECT_FEATURE_SCHEMA,
                        "shape_per_video": [8, 8],
                        "columns": list(AFFECT_COLUMNS),
                    }
                ),
                encoding="utf-8",
            )

            cases = (
                ("train", "low", "a", 0),
                ("val", "mid", "b", 1),
                ("test", "high", "c", 2),
            )
            for split, category, video_name, label in cases:
                (root / f"{split}.csv").write_text(
                    f"videos/{category}/{video_name}.mp4 {label}\n",
                    encoding="utf-8",
                )
                interaction_dir = interaction_root / split / category
                affect_dir = affect_root / split / category
                interaction_dir.mkdir(parents=True)
                affect_dir.mkdir(parents=True)
                np.save(interaction_dir / f"{video_name}.npy", np.ones((8, 40), np.float32))
                np.save(affect_dir / f"{video_name}.npy", np.full((8, 8), 2.0, np.float32))

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src/data/build_behavioral_matrices.py"),
                    "--feature_dir",
                    str(feature_dir),
                    "--output_dir",
                    str(output_dir),
                    "--csv_dir",
                    str(root),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            matrix = np.load(output_dir / "train/a_label0.npy", allow_pickle=False)
            self.assertEqual(matrix.shape, (8, 48))
            np.testing.assert_array_equal(matrix[:, :40], 1.0)
            np.testing.assert_array_equal(matrix[:, 40:], 2.0)
            manifest = json.loads((output_dir / "build_manifest.json").read_text())
            self.assertEqual(manifest["feature_schema"], BEHAVIORAL_FEATURE_SCHEMA)
            self.assertEqual(manifest["shape_per_video"], [8, 48])
            self.assertEqual(manifest["total_videos"], 3)
            self.assertEqual(
                manifest["split_counts"], {"train": 1, "val": 1, "test": 1}
            )


if __name__ == "__main__":
    unittest.main()

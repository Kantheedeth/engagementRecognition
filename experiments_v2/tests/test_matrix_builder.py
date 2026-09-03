import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments_v2.core.contracts import (
    FeatureArtifact,
    FeatureLayoutEntry,
    PairDefinition,
)
from experiments_v2.pipeline.matrix_builder import build_pair_matrices


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is not installed in this environment")
class MatrixBuilderTests(unittest.TestCase):
    def test_interaction_precedes_affect_and_dimensions_are_dynamic(self):
        import numpy as np

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            affect_root = root / "affect"
            interaction_root = root / "interaction"
            split_files = {}
            for split in ("train", "val", "test"):
                split_files[split] = root / f"{split}.csv"
                split_files[split].write_text("videos/low/sample.mp4 0\n", encoding="utf-8")
                affect_dir = affect_root / split / "low"
                interaction_dir = interaction_root / split / "low"
                affect_dir.mkdir(parents=True)
                interaction_dir.mkdir(parents=True)
                np.save(affect_dir / "sample.npy", np.full((8, 2), 2, dtype=np.float32))
                np.save(
                    interaction_dir / "sample.npy",
                    np.full((8, 3), 1, dtype=np.float32),
                )
            affect = FeatureArtifact(
                "FEATURE_A",
                "METHOD_A",
                "MODEL_A",
                "affect",
                "fa",
                affect_root,
                affect_root,
                2,
                {},
            )
            interaction = FeatureArtifact(
                "FEATURE_I",
                "METHOD_I",
                "MODEL_I",
                "interaction",
                "fi",
                interaction_root,
                interaction_root,
                3,
                {},
            )
            pair = PairDefinition(
                pair_id="PAIR_TEST",
                feature_layout=(
                    FeatureLayoutEntry(
                        "interaction", "METHOD_I", "MODEL_I", "FEATURE_I", 3, 0, 3
                    ),
                    FeatureLayoutEntry(
                        "affect", "METHOD_A", "MODEL_A", "FEATURE_A", 2, 3, 5
                    ),
                ),
                temporal_frames=8,
                directory=root,
            )
            output = root / "matrices"
            manifest = build_pair_matrices(
                pair=pair,
                features={"affect": affect, "interaction": interaction},
                split_files=split_files,
                output_dir=output,
            )
            matrix = np.load(output / "train" / "sample_label0.npy")
            self.assertEqual(matrix.shape, (8, 5))
            self.assertTrue(np.all(matrix[:, :3] == 1))
            self.assertTrue(np.all(matrix[:, 3:] == 2))
            self.assertEqual(manifest["matrix_order"], ["interaction", "affect"])
            self.assertEqual(manifest["feature_layout"][0]["start"], 0)
            self.assertEqual(manifest["feature_layout"][0]["end"], 3)
            self.assertEqual(manifest["feature_layout"][1]["start"], 3)
            self.assertEqual(manifest["feature_layout"][1]["end"], 5)


if __name__ == "__main__":
    unittest.main()

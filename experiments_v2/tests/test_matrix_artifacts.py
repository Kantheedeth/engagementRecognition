import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments_v2.core.contracts import FeatureArtifact, ModelArtifact
from experiments_v2.pipeline.matrix_artifacts import MatrixStore
from experiments_v2.pipeline.pairs import PairStore


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is not installed in this environment")
class MatrixArtifactTests(unittest.TestCase):
    def test_smoke_build_validation_and_immutable_reuse(self):
        import numpy as np

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_files = {}
            for split in ("train", "val", "test"):
                split_files[split] = root / f"{split}.csv"
                split_files[split].write_text(
                    f"videos/Low/sample_{split}.mp4 0\n", encoding="utf-8"
                )

            def model(code: str, category: str) -> ModelArtifact:
                return ModelArtifact(
                    f"MODEL_{code}",
                    f"METHOD_{code}",
                    category,
                    f"model-{code}",
                    root / f"MODEL_{code}",
                    {},
                )

            def feature(code: str, category: str, dim: int, value: float):
                directory = root / f"FEATURE_{code}"
                for split in ("train", "val", "test"):
                    data = directory / "data" / split / "low"
                    data.mkdir(parents=True)
                    np.save(
                        data / f"sample_{split}.npy",
                        np.full((8, dim), value, dtype=np.float32),
                    )
                manifest = {"method_code": code}
                if category == "affect":
                    manifest["compatibility_fix"] = {"id": "singleton-test"}
                return FeatureArtifact(
                    f"FEATURE_{code}",
                    f"METHOD_{code}",
                    f"MODEL_{code}",
                    category,
                    f"feature-{code}",
                    directory,
                    directory / "data",
                    dim,
                    manifest,
                )

            affect_model = model("A1", "affect")
            interaction_model = model("I1", "interaction")
            affect = feature("A1", "affect", 2, 2.0)
            interaction = feature("I1", "interaction", 3, 1.0)
            pair = PairStore(
                root / "artifacts",
                feature_order=["interaction", "affect"],
                temporal_frames=8,
            ).resolve_or_create(
                affect=(affect_model, affect),
                interaction=(interaction_model, interaction),
                git_commit="test",
            )
            store = MatrixStore(root / "artifacts")
            arguments = {
                "pair": pair,
                "features": {"affect": affect, "interaction": interaction},
                "split_files": split_files,
                "dataset_identity": {
                    "fingerprint": "dataset",
                    "splits": {split: {"sha256": split} for split in split_files},
                    "preprocessed_inputs": {"file_count": 3},
                    "preprocessing": {"num_frames": 8},
                },
                "git": {"commit": "test"},
                "environment": {"python": "test"},
                "dataset_name": "test dataset",
                "preprocessing_provenance": "test source",
                "smoke_samples": ["train/low/sample_train"],
            }
            first = store.resolve_or_build(**arguments)
            second = store.resolve_or_build(**arguments)
            matrix = np.load(first.data_dir / "train" / "sample_train_label0.npy")

            self.assertTrue(first.matrix_id.startswith("MATRIX_"))
            self.assertEqual(first.matrix_id, second.matrix_id)
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertTrue(np.all(matrix[:, :3] == 1.0))
            self.assertTrue(np.all(matrix[:, 3:] == 2.0))
            self.assertEqual(first.manifest["validated_files"], 3)
            self.assertEqual(
                first.manifest["compatibility_fixes"]["affect"]["id"],
                "singleton-test",
            )


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping
import unittest

from experiments_v2.core.contracts import MethodSpec, ModelArtifact
from experiments_v2.pipeline.feature_cache import FeatureCache
from experiments_v2.pipeline.pairs import PairStore


class FakeAdapter:
    def __init__(self, code: str, method_id: str, category: str, dim: int):
        self.spec = MethodSpec(
            code=code,
            method_id=method_id,
            name="fake_" + category,
            category=category,
            version="1",
            feature_dim=dim,
            feature_schema="fake_v1",
            input_kind="fake",
        )
        self.calls = 0

    def model_identity(self, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"architecture": "fake", "components": []}

    def extract_features(self, **kwargs):
        self.calls += 1
        output_dir = kwargs["output_dir"]
        (output_dir / "marker.txt").write_text("complete", encoding="utf-8")
        kwargs["log_path"].write_text("fake", encoding="utf-8")
        return {"validated_files": 1, "command": ["fake"]}

    def validate_features(self, output_dir: Path):
        if not (output_dir / "marker.txt").is_file():
            raise FileNotFoundError("missing fake feature")
        return {"validated_files": 1}


def fake_model(model_id: str, method_id: str, category: str, root: Path) -> ModelArtifact:
    return ModelArtifact(
        model_id=model_id,
        method_id=method_id,
        category=category,
        fingerprint="fingerprint-" + model_id,
        directory=root / model_id,
        manifest={"model_id": model_id, "checkpoint_size_mb": None},
    )


class FeatureCacheAndPairTests(unittest.TestCase):
    def test_valid_legacy_cache_is_snapshotted_without_extraction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "marker.txt").write_text("complete", encoding="utf-8")
            cache = FeatureCache(root / "artifacts", root)
            adapter = FakeAdapter("A9", "METHOD_A9", "affect", 3)
            model = fake_model("MODEL_TEST_A", adapter.spec.method_id, "affect", root)
            adopted = cache.resolve_or_extract(
                adapter=adapter,
                model=model,
                dataset_identity={
                    "fingerprint": "dataset-one",
                    "preprocessing": {"num_frames": 8},
                },
                input_dir=root,
                parameters={},
                force_extract=False,
                legacy_feature_dir=legacy,
                git_commit="test",
            )
            self.assertTrue(adopted.reused)
            self.assertEqual(adapter.calls, 0)
            self.assertTrue((adopted.data_dir / "marker.txt").is_file())
            self.assertEqual(
                adopted.manifest["adopted_from_legacy"], str(legacy.resolve())
            )

    def test_cache_reuse_and_force_create_new_feature_versions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = FeatureCache(root, root)
            adapter = FakeAdapter("A9", "METHOD_A9", "affect", 3)
            model = fake_model("MODEL_TEST_A", adapter.spec.method_id, "affect", root)
            identity = {
                "fingerprint": "dataset-one",
                "preprocessing": {"num_frames": 8},
            }
            first = cache.resolve_or_extract(
                adapter=adapter,
                model=model,
                dataset_identity=identity,
                input_dir=root,
                parameters={},
                force_extract=False,
                legacy_feature_dir=None,
                git_commit="test",
            )
            second = cache.resolve_or_extract(
                adapter=adapter,
                model=model,
                dataset_identity=identity,
                input_dir=root,
                parameters={},
                force_extract=False,
                legacy_feature_dir=None,
                git_commit="test",
            )
            original_manifest = (first.directory / "manifest.json").read_bytes()
            forced = cache.resolve_or_extract(
                adapter=adapter,
                model=model,
                dataset_identity=identity,
                input_dir=root,
                parameters={},
                force_extract=True,
                legacy_feature_dir=None,
                git_commit="test",
            )
            self.assertEqual(first.feature_id, second.feature_id)
            self.assertTrue(second.reused)
            self.assertEqual(
                (first.directory / "manifest.json").read_bytes(), original_manifest
            )
            self.assertNotEqual(first.feature_id, forced.feature_id)
            self.assertTrue(forced.feature_id.startswith("FEATURE_"))
            self.assertTrue((first.directory / "data" / "marker.txt").is_file())
            self.assertEqual(adapter.calls, 2)

    def test_cartesian_pairs_are_version_specific_and_reused(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = FeatureCache(root, root)
            affect_adapter = FakeAdapter("A9", "METHOD_A9", "affect", 3)
            interaction_adapter = FakeAdapter("I9", "METHOD_I9", "interaction", 5)
            affect_model = fake_model("MODEL_TEST_A", "METHOD_A9", "affect", root)
            interaction_model = fake_model(
                "MODEL_TEST_I", "METHOD_I9", "interaction", root
            )
            identity = {
                "fingerprint": "dataset-one",
                "preprocessing": {"num_frames": 8},
            }
            affect_feature = cache.resolve_or_extract(
                adapter=affect_adapter,
                model=affect_model,
                dataset_identity=identity,
                input_dir=root,
                parameters={},
                force_extract=False,
                legacy_feature_dir=None,
                git_commit="test",
            )
            interaction_feature = cache.resolve_or_extract(
                adapter=interaction_adapter,
                model=interaction_model,
                dataset_identity=identity,
                input_dir=root,
                parameters={},
                force_extract=False,
                legacy_feature_dir=None,
                git_commit="test",
            )
            store = PairStore(
                root,
                feature_order=["interaction", "affect"],
                temporal_frames=8,
            )
            first = store.generate_cartesian(
                affect=[(affect_model, affect_feature)],
                interaction=[(interaction_model, interaction_feature)],
                git_commit="test",
            )[0]
            second = store.generate_cartesian(
                affect=[(affect_model, affect_feature)],
                interaction=[(interaction_model, interaction_feature)],
                git_commit="test",
            )[0]
            self.assertEqual(first.pair_id, second.pair_id)
            self.assertEqual(first.matrix_dim, 8)


if __name__ == "__main__":
    unittest.main()

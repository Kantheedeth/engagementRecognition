from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

from experiments_v2.core.contracts import FeatureArtifact, ModelArtifact
from experiments_v2.pipeline.engagement_model import create_engagement_model
from experiments_v2.pipeline.matrix_builder import pair_matrix_contract
from experiments_v2.pipeline.pairs import PairStore
from experiments_v2.registry.builtins import create_builtin_registry


def model(root: Path, code: str, category: str) -> ModelArtifact:
    model_id = f"MODEL_{code}"
    return ModelArtifact(
        model_id=model_id,
        method_id=f"METHOD_{code}",
        category=category,
        fingerprint=f"model-{code}",
        directory=root / model_id,
        manifest={},
    )


def feature(root: Path, code: str, category: str, dim: int) -> FeatureArtifact:
    feature_id = f"FEATURE_{code}"
    model_id = f"MODEL_{code}"
    return FeatureArtifact(
        feature_id=feature_id,
        method_id=f"METHOD_{code}",
        model_id=model_id,
        category=category,
        fingerprint=f"feature-{code}",
        directory=root / feature_id,
        data_dir=root / feature_id / "data",
        feature_dim=dim,
        manifest={},
    )


def make_pair(root: Path, affect_code: str, affect_dim: int, interaction_code: str, interaction_dim: int):
    affect_model = model(root, affect_code, "affect")
    interaction_model = model(root, interaction_code, "interaction")
    return PairStore(
        root,
        feature_order=["interaction", "affect"],
        temporal_frames=8,
    ).resolve_or_create(
        affect=(affect_model, feature(root, affect_code, "affect", affect_dim)),
        interaction=(
            interaction_model,
            feature(root, interaction_code, "interaction", interaction_dim),
        ),
        git_commit="test",
    )


class RecordingClassifier:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DimensionAwarenessTests(unittest.TestCase):
    def test_builtin_legacy_dimensions_and_pair_contract(self):
        registry = create_builtin_registry()
        self.assertEqual(registry.spec("A1").feature_dim, 8)
        self.assertEqual(registry.spec("I1").feature_dim, 32)
        with TemporaryDirectory() as temporary:
            pair = make_pair(Path(temporary), "A1", 8, "I1", 32)
            self.assertEqual(pair.matrix_dim, 40)
            self.assertEqual(pair.matrix_order, ("interaction", "affect"))
            self.assertEqual(pair.entry("interaction").start, 0)
            self.assertEqual(pair.entry("interaction").end, 32)
            self.assertEqual(pair.entry("affect").start, 32)
            self.assertEqual(pair.entry("affect").end, 40)

    def test_future_dimensions_are_derived_without_assuming_forty(self):
        expected = {
            ("A1", 8, "I1", 32): 40,
            ("A2", 64, "I1", 32): 96,
            ("A1", 8, "I2", 128): 136,
            ("A2", 64, "I2", 128): 192,
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (dimensions, total) in enumerate(expected.items()):
                affect_code, affect_dim, interaction_code, interaction_dim = dimensions
                pair = make_pair(
                    root / str(index),
                    affect_code,
                    affect_dim,
                    interaction_code,
                    interaction_dim,
                )
                self.assertEqual(pair.matrix_dim, total)
                self.assertEqual(pair.entry("interaction").feature_dim, interaction_dim)
                self.assertEqual(pair.entry("affect").feature_dim, affect_dim)

    def test_engagement_factory_matches_pair_matrix_input(self):
        with TemporaryDirectory() as temporary:
            pair = make_pair(Path(temporary), "A2", 64, "I2", 128)
            classifier, resolved = create_engagement_model(
                pair=pair,
                model_config={
                    "architecture": "legacy_pure_behavioral_attention",
                    "branch_dim": 48,
                    "num_heads": 4,
                    "dropout": 0.15,
                },
                model_class=RecordingClassifier,
            )
            matrix_contract = pair_matrix_contract(pair)
            self.assertEqual(
                resolved["raw_input_dim"], matrix_contract["shape_per_video"][1]
            )
            self.assertEqual(resolved["raw_input_dim"], 192)
            self.assertEqual(classifier.kwargs["dim_inter"], 128)
            self.assertEqual(classifier.kwargs["dim_affect"], 64)
            self.assertEqual(
                classifier.kwargs["dim_inter"] + classifier.kwargs["dim_affect"],
                pair.matrix_dim,
            )

    def test_factory_rejects_dimension_or_order_mismatch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = make_pair(root / "normal", "A1", 8, "I1", 32)
            with self.assertRaises(ValueError):
                create_engagement_model(
                    pair=pair,
                    model_config={
                        "dim_inter": 999,
                        "branch_dim": 48,
                        "num_heads": 4,
                        "dropout": 0.15,
                    },
                    model_class=RecordingClassifier,
                )

            affect_model = model(root, "A1", "affect")
            interaction_model = model(root, "I1", "interaction")
            reversed_pair = PairStore(
                root / "reversed",
                feature_order=["affect", "interaction"],
                temporal_frames=8,
            ).resolve_or_create(
                affect=(affect_model, feature(root, "A1", "affect", 8)),
                interaction=(
                    interaction_model,
                    feature(root, "I1", "interaction", 32),
                ),
                git_commit="test",
            )
            with self.assertRaises(ValueError):
                create_engagement_model(
                    pair=reversed_pair,
                    model_config={
                        "branch_dim": 48,
                        "num_heads": 4,
                        "dropout": 0.15,
                    },
                    model_class=RecordingClassifier,
                )


if __name__ == "__main__":
    unittest.main()

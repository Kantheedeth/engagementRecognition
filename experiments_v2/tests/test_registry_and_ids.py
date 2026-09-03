from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from experiments_v2.core.artifacts import new_id, write_json_exclusive
from experiments_v2.registry.builtins import create_builtin_registry
from experiments_v2.registry.model_registry import ModelRegistry
from experiments_v2.pipeline.baselines import BaselineStore
from experiments_v2.pipeline.comparison import baseline_deltas
from experiments_v2.pipeline.runs import RunStore


class RegistryAndIdTests(unittest.TestCase):
    def test_builtin_method_registration(self):
        registry = create_builtin_registry()
        self.assertEqual(registry.spec("A1").method_id, "METHOD_A1")
        self.assertEqual(registry.spec("I1").method_id, "METHOD_I1")
        self.assertEqual(registry.create("A1").spec.name, "legacy_affect")
        self.assertEqual(registry.create("I1").spec.name, "legacy_interaction")

    def test_ids_have_requested_prefix(self):
        for prefix in ("MODEL", "FEATURE", "PAIR", "RUN"):
            self.assertTrue(new_id(prefix).startswith(prefix + "_"))

    def test_exclusive_json_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_json_exclusive(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"value": 2})

    def test_pretrained_model_bundle_is_reused(self):
        registry = create_builtin_registry()
        adapter = registry.create("I1")
        with TemporaryDirectory() as temporary:
            models = ModelRegistry(Path(temporary))
            identity = adapter.model_identity({"model": "missing-yolov8n.pt"})
            first = models.resolve_pretrained(
                spec=adapter.spec,
                identity=identity,
                force_train=False,
                git_commit="test",
            )
            second = models.resolve_pretrained(
                spec=adapter.spec,
                identity=identity,
                force_train=False,
                git_commit="test",
            )
            self.assertEqual(first.model_id, second.model_id)
            with self.assertRaises(ValueError):
                models.resolve_pretrained(
                    spec=adapter.spec,
                    identity=identity,
                    force_train=True,
                    git_commit="test",
                )

    def test_engagement_checkpoint_registry_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"immutable checkpoint")
            models = ModelRegistry(root / "artifacts")
            registered = models.register_engagement_checkpoint(
                model_id="MODEL_FIXED",
                pair_id="PAIR_FIXED",
                run_id="RUN_FIXED",
                checkpoint_path=checkpoint,
                model_config={"raw_input_dim": 40},
                parameter_count=123,
                validation_metric={"f1": 0.5},
                git_commit="test",
            )
            original_manifest = (registered.directory / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                models.register_engagement_checkpoint(
                    model_id="MODEL_FIXED",
                    pair_id="PAIR_OTHER",
                    run_id="RUN_OTHER",
                    checkpoint_path=checkpoint,
                    model_config={"raw_input_dim": 192},
                    parameter_count=456,
                    validation_metric={"f1": 0.9},
                    git_commit="test",
                )
            self.assertEqual(
                (registered.directory / "manifest.json").read_bytes(), original_manifest
            )

    def test_run_store_creates_unique_runs_and_refuses_collision(self):
        with TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary))
            first_id, first_dir = store.create()
            second_id, second_dir = store.create()
            self.assertTrue(first_id.startswith("RUN_"))
            self.assertTrue(second_id.startswith("RUN_"))
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first_dir, second_dir)

            first_marker = first_dir / "marker.txt"
            first_marker.write_text("preserved", encoding="utf-8")
            with patch("experiments_v2.pipeline.runs.new_id", return_value=first_id):
                with self.assertRaises(FileExistsError):
                    store.create()
            self.assertEqual(first_marker.read_text(encoding="utf-8"), "preserved")

            metrics_path = first_dir / "metrics.json"
            metrics_path.write_text("{}\n", encoding="utf-8")
            metrics = {
                "status": "complete",
                "pair_id": "PAIR_A1_I1",
                "run_id": first_id,
                "identity": {
                    "affect_method_id": "METHOD_A1",
                    "affect_model_id": "MODEL_A1",
                    "affect_feature_id": "FEATURE_A1",
                    "interaction_method_id": "METHOD_I1",
                    "interaction_model_id": "MODEL_I1",
                    "interaction_feature_id": "FEATURE_I1",
                    "engagement_model_id": "MODEL_ENGAGEMENT",
                    "dataset_fingerprint": "dataset",
                    "random_seed": 42,
                    "git": {"commit": "abc"},
                },
                "performance": {
                    "accuracy": 0.84,
                    "precision_macro": 0.81,
                    "recall_macro": 0.84,
                    "f1_macro": 0.82,
                    "confusion_matrix": [[1, 0], [0, 1]],
                    "inference_seconds": 1.0,
                    "inference_ms_per_video": 10.0,
                    "engagement_fps": 100.0,
                },
                "training": {"training_seconds": 2.0},
                "model_cost": {
                    "engagement_checkpoint_size_mb": 5.0,
                    "engagement_parameter_count": 100,
                },
                "matrix_manifest": {
                    "shape_per_video": [8, 40],
                    "matrix_order": ["interaction", "affect"],
                    "split_counts": {"train": 939, "val": 124, "test": 132},
                },
                "environment": {
                    "certification_ready": True,
                    "selected_certification_path": "reuse_legacy_artifacts",
                },
            }
            baselines = BaselineStore(Path(temporary))
            baseline = baselines.publish_official(
                metrics=metrics,
                metrics_path=metrics_path,
                affect_code="A1",
                interaction_code="I1",
            )
            self.assertTrue(baseline["baseline_id"].startswith("BASELINE_"))
            self.assertEqual(baselines.official()["baseline_id"], baseline["baseline_id"])
            with self.assertRaises(FileExistsError):
                baselines.publish_official(
                    metrics=metrics,
                    metrics_path=metrics_path,
                    affect_code="A1",
                    interaction_code="I1",
                )

            candidate = deepcopy(metrics)
            candidate["performance"]["accuracy"] = 0.90
            candidate["performance"]["f1_macro"] = 0.86
            candidate["model_cost"]["engagement_checkpoint_size_mb"] = 6.5
            candidate["model_cost"]["engagement_parameter_count"] = 125
            deltas = baseline_deltas(candidate, baseline)
            self.assertAlmostEqual(deltas["accuracy_delta"], 6.0)
            self.assertAlmostEqual(deltas["f1_delta"], 4.0)
            self.assertAlmostEqual(deltas["size_delta_mb"], 1.5)
            self.assertEqual(deltas["parameter_delta"], 25)


if __name__ == "__main__":
    unittest.main()

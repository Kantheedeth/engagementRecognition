"""Lecture snapshot isolation, loss accounting and synthetic end-to-end smoke test."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from src.data.feature_schema import BEHAVIORAL_FEATURE_SCHEMA, BEHAVIORAL_SHAPE
from src.data.prepare_lecture_subset import LABELS, SPLITS, prepare_subset, verify_subset, selection_file_keys
from src.models.dataset import feature_manifests_compatible
from src.training.train_behavioral import WeightedLossMeter, calculate_class_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LectureSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.selected = self.root / "selected"
        self.csv = self.root / "csv"
        self.output = self.root / "snapshot"
        self.csv.mkdir()
        self.source.mkdir()
        self.manifest = {"feature_schema": BEHAVIORAL_FEATURE_SCHEMA,
                         "shape_per_video": list(BEHAVIORAL_SHAPE),
                         "split_counts": dict.fromkeys(SPLITS, 3), "total_videos": 9}
        (self.source / "build_manifest.json").write_text(json.dumps(self.manifest))
        for category in LABELS:
            (self.selected / category).mkdir(parents=True)
        for number, split in enumerate(SPLITS, 1):
            (self.source / split).mkdir()
            lines = []
            for category, label in LABELS.items():
                # Deliberately repeat numeric basename across different classes.
                (self.selected / category / f"view{number}.mp4").touch()
                lines.append(f"videos/{category}/view{number}.mp4 {label}")
                np.save(self.source / split / f"view{number}_label{label}.npy",
                        np.full(BEHAVIORAL_SHAPE, label / 3, dtype=np.float32))
            (self.csv / f"{split}.csv").write_text("\n".join(lines) + "\n")

    def prepare(self):
        return prepare_subset(self.selected, self.source, self.csv, self.output, 9)

    def test_snapshot_copies_all_classes_with_colliding_basenames(self):
        result = self.prepare()
        self.assertEqual(result["total_videos"], 9)
        self.assertEqual(verify_subset(self.output)["class_counts"]["train"],
                         {"low": 1, "mid": 1, "high": 1})
        for record in result["records"]:
            target = self.output / "matrices" / record["split"] / record["matrix_name"]
            self.assertEqual(target.read_bytes(), Path(record["source_matrix"]).read_bytes())
            self.assertFalse(target.is_symlink())
        subset_manifest = json.loads((self.output / "matrices/build_manifest.json").read_text())
        self.assertFalse(feature_manifests_compatible(self.manifest, subset_manifest))

    def test_filters_unselected_source_rows_and_preserves_sources(self):
        extra = self.source / "train/view99_label0.npy"
        np.save(extra, np.zeros(BEHAVIORAL_SHAPE, dtype=np.float32))
        csv = self.csv / "train.csv"
        csv.write_text(csv.read_text() + "videos/Low/view99.mp4 0\n")
        before = csv.read_bytes()
        self.prepare()
        self.assertTrue(extra.is_file())
        self.assertEqual(csv.read_bytes(), before)
        self.assertFalse((self.output / "matrices/train" / extra.name).exists())

    def test_existing_destination_not_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_stale_extra_matrix_rejected(self):
        self.prepare()
        np.save(self.output / "matrices/train/extra_label0.npy", np.zeros(BEHAVIORAL_SHAPE))
        with self.assertRaisesRegex(ValueError, "membership mismatch"):
            verify_subset(self.output)

    def test_changed_matrix_rejected(self):
        self.prepare()
        np.save(self.output / "matrices/train/view1_label0.npy", np.ones(BEHAVIORAL_SHAPE))
        with self.assertRaisesRegex(ValueError, "content changed"):
            verify_subset(self.output)

    def test_changed_split_list_rejected(self):
        self.prepare()
        (self.output / "splits/train.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "split list changed"):
            verify_subset(self.output)

    def test_missing_source_matrix_fails_without_publishing(self):
        (self.source / "train/view1_label0.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_matrix_fails_without_publishing(self):
        np.save(self.source / "train/view1_label0.npy", np.full(BEHAVIORAL_SHAPE, np.nan))
        with self.assertRaisesRegex(ValueError, "Invalid matrix"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_clip_across_splits_rejected(self):
        path = self.csv / "test.csv"
        path.write_text(path.read_text() + "videos/low/view1.mp4 0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate class/filename"):
            self.prepare()

    def test_unmatched_selected_video_rejected(self):
        (self.selected / "low/view99.mp4").touch()
        with self.assertRaisesRegex(ValueError, "absent from source splits"):
            prepare_subset(self.selected, self.source, self.csv, self.output, 10)

    def test_portable_selection_without_local_video_folder(self):
        path = self.root / "selection.txt"
        path.write_text("# class-qualified clip identities\n" + "\n".join(
            f"{category}/view{number}.mp4" for number in (1, 2, 3) for category in LABELS
        ))
        result = prepare_subset(self.root / "nonexistent", self.source, self.csv,
                                self.output, 9, selection_file=path)
        self.assertEqual(result["total_videos"], 9)
        self.assertIsNone(result["selected_video_dir"])
        self.assertEqual(verify_subset(self.output)["total_videos"], 9)

    def test_portable_selection_rejects_invalid_and_duplicate_keys(self):
        path = self.root / "selection.txt"
        for content in ("../view1.mp4", "/low/view1.mp4", "low/sub/view1.mp4",
                        "low/view1.mp4\nLow/view1.mp4", ""):
            path.write_text(content)
            with self.assertRaises(ValueError):
                selection_file_keys(path)

    def test_synthetic_train_and_evaluate_with_isolated_outputs(self):
        experiment = self.root / "experiment"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_lecture_behavioral.py"),
             "--stage", "all", "--experiment_dir", str(experiment),
             "--selected_dir", str(self.selected), "--source_dir", str(self.source),
             "--csv_dir", str(self.csv), "--expected_count", "9", "--run_name", "smoke",
             "--device", "cpu", "--epochs", "1", "--branch_dim", "8", "--batch_size", "2"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run = experiment / "runs/smoke"
        metrics = json.loads((run / "reports/metrics.json").read_text())
        self.assertEqual(metrics["test_samples"], 3)
        self.assertEqual(len(metrics["predictions"]), 3)
        self.assertTrue((run / "reports/confusion_matrix.png").is_file())
        checkpoint = torch.load(run / "checkpoints/best_model_behavioral.pth",
                                map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["training_config"]["seed"], 42)
        self.assertEqual(checkpoint["class_weights"], [1, 1, 1])
        self.assertEqual(checkpoint["feature_manifest"]["total_videos"], 9)
        history = json.loads((run / "checkpoints/training_history.json").read_text())
        self.assertEqual(len(history), 1)
        retry = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_lecture_behavioral.py"), "--stage", "train",
             "--experiment_dir", str(experiment), "--expected_count", "9", "--run_name", "smoke"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("FileExistsError", retry.stderr)


class LossTests(unittest.TestCase):
    def test_epoch_weighted_loss_matches_whole_dataset_for_uneven_batches(self):
        torch.manual_seed(1)
        logits = torch.randn(7, 3)
        targets = torch.tensor([0, 0, 0, 0, 1, 2, 2])
        weights = torch.tensor([.7, 1.4, 1.1])
        for smoothing in (0., .05):
            criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
            meter = WeightedLossMeter()
            for start, stop in [(0, 2), (2, 5), (5, 7)]:
                meter.update(criterion(logits[start:stop], targets[start:stop]),
                             targets[start:stop], weights)
            self.assertAlmostEqual(meter.mean, criterion(logits, targets).item(), places=6)

    def test_weights_use_training_labels(self):
        labels = [0] * 133 + [1] * 55 + [2] * 64
        dataset = [(torch.zeros(8, 40), torch.tensor(label)) for label in labels]
        actual = calculate_class_weights(dataset).numpy()
        expected = np.sqrt(252 / (3 * np.array([133, 55, 64])))
        np.testing.assert_allclose(actual, expected / expected.mean(), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

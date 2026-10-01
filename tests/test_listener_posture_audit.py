"""Regression tests for audit parity and train-only sampling safeguards."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Results

from src.tools.create_listener_posture_audit import inspect_result, select_inputs


def sample_result(speaker=True, face_visible=True, shoulders_visible=True):
    listener_box = [300, 180, 460, 560, .9, 0]
    boxes = [[20, 100, 120, 580, .9, 0], listener_box] if speaker else [listener_box]
    keypoints = np.zeros((len(boxes), 17, 3), dtype=np.float32)
    keypoints[-1, 0] = [380, 220, .9 if face_visible else .1]
    keypoints[-1, 1] = [370, 210, .9 if face_visible else .1]
    keypoints[-1, 2] = [390, 210, .9 if face_visible else .1]
    keypoints[-1, 5] = [330, 290, .9 if shoulders_visible else .1]
    keypoints[-1, 6] = [430, 290, .9 if shoulders_visible else .1]
    return Results(np.zeros((640, 640, 3), dtype=np.uint8), path="fixture",
                   names={0: "person"}, boxes=torch.tensor(boxes, dtype=torch.float32), keypoints=torch.from_numpy(keypoints))


class AuditTests(unittest.TestCase):
    def test_visible_listener_and_speaker_target(self):
        rows, target, speaker = inspect_result(sample_result())
        self.assertEqual(speaker, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["upright_flag"], 1)
        self.assertEqual(rows[0]["slumped_flag"], 0)
        np.testing.assert_allclose(target, [70 / 640, 340 / 640])

    def test_zone_center_fallback(self):
        rows, target, speaker = inspect_result(sample_result(speaker=False))
        self.assertIsNone(speaker)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(target, [.135, .5])

    def test_hidden_face_is_unknown_not_sleep(self):
        row = inspect_result(sample_result(face_visible=False))[0][0]
        self.assertFalse(row["elevation_observed"])
        self.assertEqual(row["slumped_flag"], 0)
        self.assertEqual(row["slouching_flag"], 0)
        self.assertGreater(row["orientation_score"], 0)

    def test_missing_shoulders_exposes_default_without_asserting_slump(self):
        row = inspect_result(sample_result(shoulders_visible=False))[0][0]
        self.assertFalse(row["elevation_observed"])
        self.assertEqual(row["slouching_flag"], 0)
        self.assertEqual(row["slumped_flag"], 0)

    def test_empty_frame(self):
        result = Results(np.zeros((640, 640, 3), dtype=np.uint8), path="fixture",
                         names={0: "person"}, boxes=torch.zeros((0, 6), dtype=torch.float32))
        rows, target, speaker = inspect_result(result)
        self.assertEqual(rows, [])
        self.assertIsNone(speaker)

    def test_sampling_excludes_held_out_zero_padded_and_ambiguous_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train.csv").write_text("videos/mid/view12.mp4 1\nvideos/mid/view13.mp4 1\nvideos/Low/view14.mp4 0\nvideos/high/view14.mp4 2\nvideos/mid/view15.mp4 1\nvideos/mid/view15.mp4 1\n")
            (root / "val.csv").write_text("videos/mid/view0012.mp4 1\n")
            (root / "test.csv").write_text("")
            for category, name in [("mid", "view13"), ("low", "view14"), ("high", "view14"), ("mid", "view15")]:
                path = root / "train" / category / (name + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            selected, info = select_inputs(root, root, 2)
            self.assertEqual([p.stem for p in selected], ["view13", "view15"])
            self.assertEqual(info["excluded_train_ids_in_val_test"], [12])
            self.assertEqual(info["excluded_ambiguous_train_ids"], [14])


if __name__ == "__main__":
    unittest.main()

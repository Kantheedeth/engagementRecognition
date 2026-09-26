"""Meaningful association, motion, sampled tracking and provenance safeguards."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Results

from src.data.person_tracking import SampledPersonTracker, matched_motion, track_sampled_frames
from src.data.extract_interaction_features import extract_video_interaction_descriptors, validate_reusable_provenance
from src.data.build_feature_matrices import read_interaction_manifest


def result(boxes):
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6)
    keypoints = torch.zeros((len(b), 17, 3))
    for i, box in enumerate(b):
        cx = float((box[0] + box[2]) / 2)
        y = float(box[1])
        for k, dx, dy in [(0, 0, 40), (1, -8, 30), (2, 8, 30), (5, -30, 100), (6, 30, 100)]:
            keypoints[i, k] = torch.tensor([cx + dx, y + dy, .9])
    return Results(np.zeros((640, 640, 3), np.uint8), path="fixture", names={0: "person"}, boxes=b, keypoints=keypoints)


A = [220, 100, 310, 440, .9, 0]
B = [440, 100, 550, 440, .9, 0]
TEACHER = [10, 70, 100, 500, .9, 0]


class TrackingTests(unittest.TestCase):
    @staticmethod
    def detection(box):
        box = np.asarray(box[:4], dtype=np.float32)
        kpts = np.zeros((17, 2), dtype=np.float32)
        kpts[:11] = (box[:2] + box[2:]) / 2
        return {"box": box, "center": (box[:2] + box[2:]) / 1280,
                "keypoints": kpts, "conf": np.ones(17, dtype=np.float32)}

    def test_ids_follow_boxes_when_detection_order_changes(self):
        tracker = SampledPersonTracker()
        first = tracker.update([self.detection(A), self.detection(B)], 0)
        second = tracker.update([self.detection(B), self.detection(A)], 1)
        self.assertTrue(all(i is not None for i in first))
        self.assertEqual(second, first[::-1])

    def test_short_occlusion_and_new_clip_reset(self):
        tracker = SampledPersonTracker(max_gap=2)
        first = tracker.update([self.detection(A)], 0)
        self.assertEqual(tracker.update([], 1), [])
        self.assertEqual(tracker.update([self.detection(A)], 2), first)
        self.assertEqual(SampledPersonTracker().update([self.detection(A)], 0), first)

    def test_motion_uses_identity_not_nearest_neighbor(self):
        previous = {1: np.array([0., 0.]), 2: np.array([1., 0.])}
        current = {1: np.array([1., 0.]), 2: np.array([0., 0.]), 3: np.array([.5, .5])}
        self.assertEqual(matched_motion(previous, current), (1., 0., 0., 1.))
        self.assertEqual(matched_motion({}, current), (0., 0., 0., 0.))

    def test_extractor_keeps_posture_but_matches_motion_by_id(self):
        first, second = result([A, B]), result([B, A])
        legacy = extract_video_interaction_descriptors([first, second])
        tracked = extract_video_interaction_descriptors([first, second], [[1, 2], [1, 2]])
        np.testing.assert_allclose(tracked[:, :24], legacy[:, :24])
        self.assertGreater(tracked[1, 24], 3)
        self.assertEqual(legacy[1, 24], 0)
        self.assertEqual(tracked[1, 27], 1)

    def test_known_speaker_excluded_after_leaving_zone(self):
        frame = result([A, B])
        features = extract_video_interaction_descriptors([frame], [[1, 2]], [{0}])
        self.assertAlmostEqual(float(features[0, 16]), 1 / 15)
        self.assertEqual(features[0, 28], 0)

    def test_different_speaker_id_has_no_motion(self):
        frame = result([TEACHER, A])
        moved_teacher = [30, 70, 120, 500, .9, 0]
        features = extract_video_interaction_descriptors([frame, result([moved_teacher, A])], [[1, 2], [3, 2]])
        self.assertEqual(features[1, 29], 0)

    def test_one_batched_call_tracks_eight_stored_frames_and_resets(self):
        class Model:
            calls = 0
            def predict(self, **kwargs):
                self.calls += 1
                if len(kwargs["source"]) != 8:
                    raise AssertionError("expected one eight-image batch")
                return [result([TEACHER, A])] * 8
        frames = np.zeros((8, 640, 640, 3), dtype=np.uint8)
        settings = dict(max_center_distance=.22, max_gap=2,
                        center_weight=.55, iou_weight=.30, pose_weight=.15)
        model = Model()
        args = (model, frames, "cpu", .25, settings, (0, .27, 0, 1), np.array([.135, .5]))
        results, ids, excluded, details = track_sampled_frames(*args)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(results), 8)
        self.assertEqual(details["tracking_updates"], 8)
        self.assertEqual(len({row[1] for row in ids}), 1)
        self.assertTrue(all(0 in e for e in excluded))
        self.assertEqual(track_sampled_frames(*args)[1][0], ids[0])

    def test_legacy_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "interaction_features").mkdir()
            (root / "interaction_features" / "extraction_manifest.json").write_text(
                json.dumps({"feature_schema": "yolov8_person_geometry_32_v1", "shape_per_video": [8, 32]}))
            with self.assertRaisesRegex(ValueError, "Unsupported interaction schema"):
                read_interaction_manifest(root)
        with self.assertRaises(RuntimeError):
            validate_reusable_provenance({"feature_schema": "old"}, {"feature_schema": "new"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.data.extract_interaction_features import (
    reset_tracker_state,
    result_observations,
)


class _BoxesWithoutIds:
    xyxy = torch.tensor([[10.0, 20.0, 110.0, 220.0]])
    conf = torch.tensor([0.8])
    cls = torch.tensor([0.0])
    id = None

    def __len__(self):
        return 1


class _Tracker:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class InteractionExtractorTests(unittest.TestCase):
    def test_missing_boxes_id_becomes_untracked_unknown_evidence(self):
        result = SimpleNamespace(
            boxes=_BoxesWithoutIds(),
            keypoints=None,
            orig_shape=(640, 640),
        )
        observations = result_observations(
            result,
            frame_index=0,
            zone=(0.0, 0.27, 0.0, 1.0),
        )
        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0]["track_id"])
        self.assertEqual(observations[0]["pose_confidence"], 0.0)

    def test_reset_clears_every_tracker_and_video_path(self):
        trackers = [_Tracker(), _Tracker()]
        predictor = SimpleNamespace(trackers=trackers, vid_path=["a", "b"])
        reset_tracker_state(SimpleNamespace(predictor=predictor))
        self.assertEqual([tracker.reset_count for tracker in trackers], [1, 1])
        self.assertEqual(predictor.vid_path, [None, None])


if __name__ == "__main__":
    unittest.main()

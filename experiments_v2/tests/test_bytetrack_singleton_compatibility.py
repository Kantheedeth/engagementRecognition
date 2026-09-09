from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments_v2.compatibility.bytetrack_singleton import (
    COMPATIBILITY_FIX_ID,
    apply_bytetrack_singleton_numpy_mask_v1,
)


class FakeBoxes:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def numpy(self):
        if isinstance(self.data, torch.Tensor):
            return FakeBoxes(self.data.detach().cpu().numpy())
        return self


class FakeByteTracker:
    def __init__(self):
        self.calls = []

    def update(self, results, img=None):
        self.calls.append((results, img))
        return results.data


class ByteTrackSingletonCompatibilityTests(unittest.TestCase):
    def test_only_singleton_is_normalized_without_value_change(self):
        tracker = FakeByteTracker()
        module = SimpleNamespace(tracker=SimpleNamespace(_tracker=tracker))
        original = tracker.update
        shim = apply_bytetrack_singleton_numpy_mask_v1(module)

        singleton = FakeBoxes(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
        multiple = FakeBoxes(torch.ones((2, 3), dtype=torch.float32))
        empty = FakeBoxes(torch.empty((0, 3), dtype=torch.float32))

        singleton_result = tracker.update(singleton)
        tracker.update(multiple)
        tracker.update(empty)

        self.assertIsInstance(singleton_result, np.ndarray)
        np.testing.assert_array_equal(singleton_result, singleton.data.numpy())
        self.assertIs(tracker.calls[1][0], multiple)
        self.assertIs(tracker.calls[2][0], empty)
        self.assertEqual(shim.trigger_count, 1)
        self.assertEqual(shim.metadata["id"], COMPATIBILITY_FIX_ID)

        shim.restore()
        self.assertEqual(tracker.update, original)


if __name__ == "__main__":
    unittest.main()

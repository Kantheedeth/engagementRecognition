import tempfile
import unittest
from pathlib import Path

from src.tools.create_high_group_visual_audit import (
    episode_for,
    unavailable_reason,
    view_number,
)


class HighGroupVisualAuditTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            {"episode_id": "H01", "start_view": 2227, "end_view": 2239},
            {"episode_id": "H02", "start_view": 2299, "end_view": 2303},
            {"episode_id": "H03", "start_view": 2348, "end_view": 2352},
            {"episode_id": "H04", "start_view": 2470, "end_view": 2532},
        ]

    def test_view_and_episode_mapping(self):
        self.assertEqual(view_number("high/view2470.mp4"), 2470)
        self.assertEqual(episode_for(2470, self.episodes), "H04")
        self.assertEqual(episode_for(2303, self.episodes), "H02")

    def test_ambiguous_or_non_high_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            view_number("low/view2470.mp4")
        with self.assertRaises(ValueError):
            episode_for(2400, self.episodes)

    def test_local_files_are_available_and_missing_files_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "local.npz"
            local.write_bytes(b"local")
            self.assertIsNone(unavailable_reason((local,)))
            reason = unavailable_reason((Path(directory) / "missing.npz",))
            self.assertIn("missing", reason)


if __name__ == "__main__":
    unittest.main()

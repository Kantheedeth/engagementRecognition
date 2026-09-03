from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.split_integrity import audit_split_integrity


class SplitIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "train.csv").write_text("videos/low/a.mp4 0\n")
        (self.root / "val.csv").write_text("videos/mid/b.mp4 1\n")
        (self.root / "test.csv").write_text("videos/high/c.mp4 2\n")

    def tearDown(self):
        self.temporary.cleanup()

    def write_groups(self, rows):
        path = self.root / "groups.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("video_path", "session_id", "golden_pair_id"),
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_authoritative_groups_pass(self):
        groups = self.write_groups(
            [
                {"video_path": "videos/low/a.mp4", "session_id": "s1", "golden_pair_id": ""},
                {"video_path": "videos/mid/b.mp4", "session_id": "s2", "golden_pair_id": "p2"},
                {"video_path": "videos/high/c.mp4", "session_id": "s3", "golden_pair_id": "p3"},
            ]
        )
        report = audit_split_integrity(self.root, groups)
        self.assertEqual(report["status"], "verified_no_cross_split_groups")
        self.assertEqual(report["record_count"], 3)

    def test_session_crossing_splits_fails(self):
        groups = self.write_groups(
            [
                {"video_path": "videos/low/a.mp4", "session_id": "shared", "golden_pair_id": ""},
                {"video_path": "videos/mid/b.mp4", "session_id": "shared", "golden_pair_id": ""},
                {"video_path": "videos/high/c.mp4", "session_id": "s3", "golden_pair_id": ""},
            ]
        )
        with self.assertRaisesRegex(ValueError, "leakage detected"):
            audit_split_integrity(self.root, groups)

    def test_golden_pair_crossing_splits_fails(self):
        groups = self.write_groups(
            [
                {"video_path": "videos/low/a.mp4", "session_id": "s1", "golden_pair_id": "shared-pair"},
                {"video_path": "videos/mid/b.mp4", "session_id": "s2", "golden_pair_id": "shared-pair"},
                {"video_path": "videos/high/c.mp4", "session_id": "s3", "golden_pair_id": ""},
            ]
        )
        with self.assertRaisesRegex(ValueError, "golden-pair leakage detected"):
            audit_split_integrity(self.root, groups)

    def test_missing_video_mapping_fails(self):
        groups = self.write_groups(
            [
                {"video_path": "videos/low/a.mp4", "session_id": "s1", "golden_pair_id": ""},
                {"video_path": "videos/mid/b.mp4", "session_id": "s2", "golden_pair_id": ""},
            ]
        )
        with self.assertRaisesRegex(ValueError, "missing 1 dataset videos"):
            audit_split_integrity(self.root, groups)

    def test_duplicate_inside_one_split_fails(self):
        with (self.root / "train.csv").open("a", encoding="utf-8") as file:
            file.write("videos/low/a.mp4 0\n")
        groups = self.write_groups(
            [
                {"video_path": "videos/low/a.mp4", "session_id": "s1", "golden_pair_id": ""},
                {"video_path": "videos/mid/b.mp4", "session_id": "s2", "golden_pair_id": ""},
                {"video_path": "videos/high/c.mp4", "session_id": "s3", "golden_pair_id": ""},
            ]
        )
        with self.assertRaisesRegex(ValueError, "Exact video paths"):
            audit_split_integrity(self.root, groups)


if __name__ == "__main__":
    unittest.main()

"""Episode-held-out fold construction tests."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.feature_schema import BEHAVIORAL_FEATURE_SCHEMA, BEHAVIORAL_SHAPE
from src.data.create_matched_random_protocol import create_matched_random_protocol
from src.data.prepare_episode_cv import prepare_episode_cv, verify_episode_cv
from src.data.prepare_lecture_subset import LABELS, SPLITS, prepare_subset


class EpisodeCVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source, selected, csv_dir = (self.root / "source", self.root / "selected",
                                     self.root / "csv")
        source.mkdir()
        csv_dir.mkdir()
        (source / "build_manifest.json").write_text(json.dumps({
            "feature_schema": BEHAVIORAL_FEATURE_SCHEMA,
            "shape_per_video": list(BEHAVIORAL_SHAPE),
            "split_counts": dict.fromkeys(SPLITS, 3),
            "total_videos": 9,
        }))
        for category in LABELS:
            (selected / category).mkdir(parents=True)
        for number, split in enumerate(SPLITS, 1):
            (source / split).mkdir()
            lines = []
            for category, label in LABELS.items():
                (selected / category / f"view{number}.mp4").touch()
                lines.append(f"videos/{category}/view{number}.mp4 {label}")
                np.save(source / split / f"view{number}_label{label}.npy",
                        np.full(BEHAVIORAL_SHAPE, label + number / 10, dtype=np.float32))
            (csv_dir / f"{split}.csv").write_text("\n".join(lines) + "\n")
        self.base = self.root / "base"
        prepare_subset(selected, source, csv_dir, self.base, expected_count=9)
        self.protocol = self.root / "protocol.json"
        episodes = []
        prefixes = {"low": "L", "mid": "M", "high": "H"}
        for category in LABELS:
            for number in (1, 2, 3):
                episodes.append({"episode_id": f"{prefixes[category]}{number}",
                                 "class": category, "start_view": number,
                                 "end_view": number, "outer_fold": number})
        self.protocol.write_text(json.dumps({
            "format_version": 1,
            "outer_folds": 3,
            "limitation": "synthetic test",
            "validation_episodes": {
                "1": ["L2", "M2", "H2"],
                "2": ["L3", "M3", "H3"],
                "3": ["L1", "M1", "H1"],
            },
            "episodes": episodes,
        }))

    def test_every_clip_is_tested_once_without_episode_leakage(self):
        output = self.root / "folds"
        summary = prepare_episode_cv(self.base, self.protocol, output)
        self.assertEqual(summary["total_videos"], 9)
        self.assertEqual(summary["episodes"], 9)
        verified = verify_episode_cv(output, self.protocol)
        self.assertEqual(set(verified["folds"]), {"1", "2", "3"})
        for fold in (1, 2, 3):
            selection = json.loads((output / f"fold_{fold}/selection_manifest.json").read_text())
            self.assertEqual(selection["class_counts"], {
                "train": {"low": 1, "mid": 1, "high": 1},
                "val": {"low": 1, "mid": 1, "high": 1},
                "test": {"low": 1, "mid": 1, "high": 1},
            })
            episode_sets = [set(selection["split_episodes"][split]) for split in SPLITS]
            self.assertFalse(episode_sets[0] & episode_sets[1])
            self.assertFalse(episode_sets[0] & episode_sets[2])
            self.assertFalse(episode_sets[1] & episode_sets[2])

    def test_protocol_rejects_validation_test_overlap(self):
        protocol = json.loads(self.protocol.read_text())
        protocol["validation_episodes"]["1"] = ["L1", "M2", "H2"]
        self.protocol.write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_episode_cv(self.base, self.protocol, self.root / "folds")

    def test_random_control_exactly_matches_target_class_counts(self):
        target = self.root / "target_folds"
        prepare_episode_cv(self.base, self.protocol, target)
        random_protocol = self.root / "random_protocol.json"
        generated = create_matched_random_protocol(
            self.base, target, random_protocol, split_seed=123
        )
        self.assertEqual(len(generated["episodes"]), 9)
        output = self.root / "random_folds"
        prepare_episode_cv(self.base, random_protocol, output)
        verify_episode_cv(output, random_protocol)
        for fold in (1, 2, 3):
            target_selection = json.loads(
                (target / f"fold_{fold}/selection_manifest.json").read_text()
            )
            random_selection = json.loads(
                (output / f"fold_{fold}/selection_manifest.json").read_text()
            )
            self.assertEqual(random_selection["class_counts"],
                             target_selection["class_counts"])


if __name__ == "__main__":
    unittest.main()

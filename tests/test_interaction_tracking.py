from __future__ import annotations

import unittest

import numpy as np

from src.data.feature_schema import TRACK_INTERACTION_COLUMNS
from src.data.interaction_tracking import (
    assign_track_roles,
    build_frame_descriptor,
    compute_instruction_alignment_proxy,
    compute_teacher_score,
    pairwise_distance_aggregation,
    reliability_weighted_mean,
    teacher_zone_membership,
    track_coverage,
)


class InteractionTrackingTests(unittest.TestCase):
    def test_teacher_zone_membership_includes_boundaries(self):
        zone = (0.0, 0.27, 0.0, 1.0)
        self.assertTrue(teacher_zone_membership(0.0, 0.0, zone))
        self.assertTrue(teacher_zone_membership(0.27, 1.0, zone))
        self.assertFalse(teacher_zone_membership(0.271, 0.5, zone))

    def test_track_coverage_counts_unique_frames(self):
        observations = [
            {"frame_index": 0},
            {"frame_index": 0},
            {"frame_index": 3},
        ]
        self.assertEqual(track_coverage(observations, 8), 0.25)

    def test_teacher_score_uses_recorded_weights(self):
        score = compute_teacher_score(0.75, 0.50, 0.90, (0.70, 0.20, 0.10))
        self.assertAlmostEqual(score, 0.715)

    def test_role_assignment_does_not_force_teacher(self):
        summaries = {
            1: {
                "teacher_score": 0.59,
                "coverage": 1.0,
                "mean_detection_confidence": 0.9,
                "zone_fraction": 0.0,
            },
            2: {
                "teacher_score": 0.2,
                "coverage": 0.125,
                "mean_detection_confidence": 0.9,
                "zone_fraction": 0.0,
            },
        }
        teacher_id, roles = assign_track_roles(summaries)
        self.assertIsNone(teacher_id)
        self.assertEqual(roles[1]["role"], "student")
        self.assertEqual(roles[2]["role"], "unknown")
        self.assertEqual(roles[2]["exclusion_reason"], "low_track_coverage")

    def test_role_assignment_selects_one_teacher_and_excludes_zone_ambiguity(self):
        summaries = {
            4: {
                "teacher_score": 0.92,
                "coverage": 0.875,
                "mean_detection_confidence": 0.95,
                "zone_fraction": 1.0,
            },
            5: {
                "teacher_score": 0.80,
                "coverage": 0.75,
                "mean_detection_confidence": 0.90,
                "zone_fraction": 0.80,
            },
            6: {
                "teacher_score": 0.25,
                "coverage": 0.75,
                "mean_detection_confidence": 0.90,
                "zone_fraction": 0.0,
            },
        }
        teacher_id, roles = assign_track_roles(summaries)
        self.assertEqual(teacher_id, 4)
        self.assertEqual(roles[4]["role"], "teacher")
        self.assertEqual(roles[5]["role"], "unknown")
        self.assertEqual(roles[6]["role"], "student")

    def test_weighted_mean_marks_missing_evidence_unreliable(self):
        self.assertEqual(reliability_weighted_mean([0.5], [0.0]), (0.0, 0.0))
        mean, reliability = reliability_weighted_mean([0.2, 0.8], [0.25, 0.75])
        self.assertAlmostEqual(mean, 0.65)
        self.assertAlmostEqual(reliability, 0.5)

    def test_missing_pose_has_zero_alignment_reliability(self):
        value, reliability, method, vector = compute_instruction_alignment_proxy(
            (0.8, 0.5), (0.1, 0.5), None, None
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(reliability, 0.0)
        self.assertIsNone(method)
        self.assertIsNone(vector)

    def test_pairwise_pooling_is_permutation_invariant(self):
        centers = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.2]]
        forward = pairwise_distance_aggregation(centers, [1.0, 0.8, 0.6])
        reverse = pairwise_distance_aggregation(
            list(reversed(centers)), list(reversed([1.0, 0.8, 0.6]))
        )
        for key in forward:
            self.assertAlmostEqual(forward[key], reverse[key])

    def test_descriptor_uses_all_students_without_ranked_slots(self):
        observations = []
        summaries = {}
        assignments = {}
        for track_id in range(1, 9):
            observations.append(
                {
                    "track_id": track_id,
                    "center": [0.3 + track_id * 0.05, 0.5],
                    "size": [0.1, 0.2],
                    "size_x": 0.1,
                    "size_y": 0.2,
                    "detection_confidence": 0.9,
                    "pose_confidence": 0.8,
                    "instruction_alignment_proxy": 0.7,
                    "instruction_alignment_reliability": 0.8,
                    "orientation_vector": [-1.0, 0.0],
                    "orientation_reliability": 0.8,
                }
            )
            summaries[track_id] = {
                "coverage": 1.0,
                "mean_detection_confidence": 0.9,
                "mean_pose_confidence": 0.8,
                "mean_motion": 0.01,
            }
            assignments[track_id] = {
                "role": "student",
                "role_confidence": 0.9,
                "exclusion_reason": None,
            }

        descriptor = build_frame_descriptor(
            observations,
            summaries,
            assignments,
            None,
            close_distance=0.15,
            alignment_threshold=0.55,
        )
        reversed_descriptor = build_frame_descriptor(
            list(reversed(observations)),
            summaries,
            assignments,
            None,
            close_distance=0.15,
            alignment_threshold=0.55,
        )
        index = {name: position for position, name in enumerate(TRACK_INTERACTION_COLUMNS)}
        self.assertEqual(descriptor.shape, (40,))
        self.assertEqual(descriptor[index["visible_student_count"]], 8.0)
        np.testing.assert_allclose(descriptor, reversed_descriptor, atol=1e-7)

    def test_missing_pose_keeps_count_but_zeros_pooling_reliability(self):
        observation = {
            "track_id": 1,
            "center": [0.5, 0.5],
            "size": [0.1, 0.2],
            "size_x": 0.1,
            "size_y": 0.2,
            "detection_confidence": 0.9,
            "pose_confidence": 0.0,
            "instruction_alignment_proxy": 0.0,
            "instruction_alignment_reliability": 0.0,
            "orientation_vector": None,
            "orientation_reliability": 0.0,
        }
        summaries = {
            1: {
                "coverage": 1.0,
                "mean_detection_confidence": 0.9,
                "mean_pose_confidence": 0.0,
                "mean_motion": 0.0,
            }
        }
        assignments = {
            1: {
                "role": "student",
                "role_confidence": 0.9,
                "exclusion_reason": None,
            }
        }
        descriptor = build_frame_descriptor(
            [observation],
            summaries,
            assignments,
            None,
            close_distance=0.15,
            alignment_threshold=0.55,
        )
        index = {name: position for position, name in enumerate(TRACK_INTERACTION_COLUMNS)}
        self.assertEqual(descriptor[index["visible_student_count"]], 1.0)
        self.assertEqual(descriptor[index["mean_student_x"]], 0.0)
        self.assertEqual(descriptor[index["student_interaction_reliability"]], 0.0)


if __name__ == "__main__":
    unittest.main()

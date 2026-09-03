"""Pure helpers for role-aware, track-pooled classroom interaction features."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from src.data.feature_schema import TRACK_INTERACTION_COLUMNS


def teacher_zone_membership(
    center_x: float,
    center_y: float,
    zone: Sequence[float],
) -> bool:
    """Return whether a normalized point lies inside ``(xmin, xmax, ymin, ymax)``."""
    if len(zone) != 4:
        raise ValueError("teacher zone must contain xmin, xmax, ymin, ymax")
    x_min, x_max, y_min, y_max = (float(value) for value in zone)
    if not (0.0 <= x_min <= x_max <= 1.0 and 0.0 <= y_min <= y_max <= 1.0):
        raise ValueError(f"invalid normalized teacher zone: {tuple(zone)}")
    return x_min <= float(center_x) <= x_max and y_min <= float(center_y) <= y_max


def track_coverage(observations: Sequence[Mapping], total_frames: int) -> float:
    """Fraction of clip frames containing at least one observation of a track."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    frames = {
        int(observation["frame_index"])
        for observation in observations
        if 0 <= int(observation["frame_index"]) < total_frames
    }
    return min(1.0, len(frames) / float(total_frames))


def reliability_weighted_mean(
    values: Iterable[float],
    weights: Iterable[float],
) -> tuple[float, float]:
    """Return a weighted mean and mean usable reliability; empty evidence is (0, 0)."""
    value_array = np.asarray(list(values), dtype=np.float64)
    weight_array = np.asarray(list(weights), dtype=np.float64)
    if value_array.shape != weight_array.shape:
        raise ValueError("values and weights must have the same shape")
    if value_array.ndim != 1:
        raise ValueError("values and weights must be one-dimensional")
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.any(valid):
        return 0.0, 0.0
    usable_values = value_array[valid]
    usable_weights = np.clip(weight_array[valid], 0.0, 1.0)
    weight_sum = float(usable_weights.sum())
    if weight_sum <= 0.0:
        return 0.0, 0.0
    mean = float(np.dot(usable_values, usable_weights) / weight_sum)
    reliability = float(np.clip(usable_weights.mean(), 0.0, 1.0))
    return mean, reliability


def reliability_weighted_std(
    values: Iterable[float],
    weights: Iterable[float],
) -> tuple[float, float]:
    """Return weighted standard deviation with the same reliability convention."""
    value_array = np.asarray(list(values), dtype=np.float64)
    weight_array = np.asarray(list(weights), dtype=np.float64)
    mean, reliability = reliability_weighted_mean(value_array, weight_array)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.any(valid):
        return 0.0, 0.0
    usable_values = value_array[valid]
    usable_weights = np.clip(weight_array[valid], 0.0, 1.0)
    variance = float(
        np.dot((usable_values - mean) ** 2, usable_weights) / usable_weights.sum()
    )
    return float(np.sqrt(max(0.0, variance))), reliability


def compute_teacher_score(
    zone_fraction: float,
    coverage: float,
    mean_detection_confidence: float,
    weights: Sequence[float] = (0.70, 0.20, 0.10),
) -> float:
    """Transparent teacher-candidate score; this is a role heuristic, not identity."""
    if len(weights) != 3 or any(weight < 0.0 for weight in weights):
        raise ValueError("teacher score weights must contain three non-negative values")
    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("teacher score weights must have a positive sum")
    evidence = np.clip(
        [zone_fraction, coverage, mean_detection_confidence], 0.0, 1.0
    )
    return float(np.dot(evidence, np.asarray(weights, dtype=np.float64)) / weight_sum)


def assign_track_roles(
    summaries: Mapping[int, Mapping[str, float]],
    *,
    minimum_teacher_score: float = 0.60,
    minimum_teacher_coverage: float = 0.50,
    minimum_teacher_detection_confidence: float = 0.45,
    minimum_student_coverage: float = 0.25,
    ambiguous_zone_fraction: float = 0.50,
) -> tuple[int | None, dict[int, dict]]:
    """Assign one clip-level role per track without forcing a teacher."""
    candidates = []
    for track_id in sorted(summaries):
        summary = summaries[track_id]
        if (
            summary["teacher_score"] >= minimum_teacher_score
            and summary["coverage"] >= minimum_teacher_coverage
            and summary["mean_detection_confidence"]
            >= minimum_teacher_detection_confidence
        ):
            candidates.append(track_id)

    teacher_track_id = None
    if candidates:
        teacher_track_id = max(
            candidates,
            key=lambda track_id: (
                summaries[track_id]["teacher_score"],
                summaries[track_id]["coverage"],
                summaries[track_id]["mean_detection_confidence"],
                -track_id,
            ),
        )

    assignments: dict[int, dict] = {}
    for track_id in sorted(summaries):
        summary = summaries[track_id]
        if track_id == teacher_track_id:
            role = "teacher"
            reason = None
            confidence = summary["teacher_score"]
        elif summary["coverage"] < minimum_student_coverage:
            role = "unknown"
            reason = "low_track_coverage"
            confidence = summary["coverage"]
        elif summary["zone_fraction"] >= ambiguous_zone_fraction:
            role = "unknown"
            reason = "nonselected_instruction_zone_track"
            confidence = summary["zone_fraction"]
        else:
            role = "student"
            reason = None
            confidence = float(
                np.clip(
                    summary["coverage"] * summary["mean_detection_confidence"],
                    0.0,
                    1.0,
                )
            )
        assignments[track_id] = {
            "role": role,
            "role_confidence": float(confidence),
            "exclusion_reason": reason,
        }
    return teacher_track_id, assignments


def _valid_keypoint(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    index: int,
    minimum_confidence: float,
) -> bool:
    return bool(
        index < len(keypoints)
        and index < len(confidences)
        and np.isfinite(keypoints[index]).all()
        and np.isfinite(confidences[index])
        and confidences[index] >= minimum_confidence
        and not np.allclose(keypoints[index], 0.0)
    )


def orientation_vector_from_pose(
    keypoints: Sequence[Sequence[float]] | None,
    keypoint_confidences: Sequence[float] | None,
    *,
    minimum_confidence: float = 0.30,
) -> tuple[np.ndarray | None, float, str | None]:
    """Estimate a weak 2-D head/body-axis proxy and its reliability."""
    if keypoints is None or keypoint_confidences is None:
        return None, 0.0, None
    points = np.asarray(keypoints, dtype=np.float64)
    confidences = np.asarray(keypoint_confidences, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or confidences.ndim != 1:
        return None, 0.0, None

    nose_valid = _valid_keypoint(points, confidences, 0, minimum_confidence)
    eye_indices = [
        index
        for index in (1, 2)
        if _valid_keypoint(points, confidences, index, minimum_confidence)
    ]
    if nose_valid and eye_indices:
        eye_midpoint = points[eye_indices].mean(axis=0)
        vector = points[0] - eye_midpoint
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            used = [0, *eye_indices]
            return (
                (vector / norm).astype(np.float32),
                float(np.clip(confidences[used].mean(), 0.0, 1.0)),
                "nose_eye_axis",
            )

    shoulder_indices = [
        index
        for index in (5, 6)
        if _valid_keypoint(points, confidences, index, minimum_confidence)
    ]
    if nose_valid and len(shoulder_indices) == 2:
        shoulder_midpoint = points[shoulder_indices].mean(axis=0)
        vector = points[0] - shoulder_midpoint
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            used = [0, *shoulder_indices]
            return (
                (vector / norm).astype(np.float32),
                float(np.clip(confidences[used].mean(), 0.0, 1.0)),
                "nose_shoulder_axis",
            )
    return None, 0.0, None


def alignment_from_vector(
    center: Sequence[float],
    target: Sequence[float],
    orientation_vector: Sequence[float] | None,
    orientation_reliability: float,
) -> tuple[float, float]:
    """Compare an orientation proxy with a target direction."""
    if orientation_vector is None or orientation_reliability <= 0.0:
        return 0.0, 0.0
    center_array = np.asarray(center, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    orientation = np.asarray(orientation_vector, dtype=np.float64)
    target_vector = target_array - center_array
    target_norm = float(np.linalg.norm(target_vector))
    orientation_norm = float(np.linalg.norm(orientation))
    if target_norm <= 1e-6 or orientation_norm <= 1e-6:
        return 0.0, 0.0
    cosine = float(
        np.dot(orientation / orientation_norm, target_vector / target_norm)
    )
    value = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
    return value, float(np.clip(orientation_reliability, 0.0, 1.0))


def compute_instruction_alignment_proxy(
    center: Sequence[float],
    target: Sequence[float],
    keypoints: Sequence[Sequence[float]] | None,
    keypoint_confidences: Sequence[float] | None,
    *,
    minimum_confidence: float = 0.30,
) -> tuple[float, float, str | None, list[float] | None]:
    """Return an orientation-to-instruction proxy, not a gaze or attention label."""
    vector, reliability, method = orientation_vector_from_pose(
        keypoints,
        keypoint_confidences,
        minimum_confidence=minimum_confidence,
    )
    value, reliability = alignment_from_vector(center, target, vector, reliability)
    vector_list = None if vector is None else [float(vector[0]), float(vector[1])]
    return value, reliability, method, vector_list


def pairwise_distance_aggregation(
    centers: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    *,
    close_distance: float = 0.15,
) -> dict[str, float]:
    """Permutation-invariant student-distance summary with explicit reliability."""
    center_array = np.asarray(centers, dtype=np.float64)
    if center_array.size == 0:
        center_array = np.empty((0, 2), dtype=np.float64)
    if center_array.ndim != 2 or center_array.shape[1] != 2:
        raise ValueError("centers must have shape (N, 2)")
    if weights is None:
        weight_array = np.ones(len(center_array), dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (len(center_array),):
        raise ValueError("weights must contain one value per center")

    valid = (
        np.isfinite(center_array).all(axis=1)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )
    center_array = center_array[valid]
    weight_array = np.clip(weight_array[valid], 0.0, 1.0)
    if len(center_array) < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "minimum": 0.0,
            "close_fraction": 0.0,
            "compactness": 0.0,
            "reliability": 0.0,
        }

    distances = []
    pair_weights = []
    for first in range(len(center_array) - 1):
        for second in range(first + 1, len(center_array)):
            distances.append(float(np.linalg.norm(center_array[first] - center_array[second])))
            pair_weights.append(float(weight_array[first] * weight_array[second]))
    distance_array = np.asarray(distances, dtype=np.float64)
    pair_weight_array = np.asarray(pair_weights, dtype=np.float64)
    mean, reliability = reliability_weighted_mean(distance_array, pair_weight_array)
    std, _ = reliability_weighted_std(distance_array, pair_weight_array)
    close_fraction, _ = reliability_weighted_mean(
        (distance_array <= close_distance).astype(np.float64), pair_weight_array
    )
    centroid = np.average(center_array, axis=0, weights=weight_array)
    compactness, _ = reliability_weighted_mean(
        np.linalg.norm(center_array - centroid, axis=1), weight_array
    )
    return {
        "mean": mean,
        "std": std,
        "minimum": float(distance_array.min()),
        "close_fraction": close_fraction,
        "compactness": compactness,
        "reliability": reliability,
    }


def summarize_track(
    observations: Sequence[Mapping],
    total_frames: int,
    teacher_score_weights: Sequence[float],
) -> dict[str, float]:
    """Summarize a local track across the full clip."""
    ordered = sorted(observations, key=lambda item: int(item["frame_index"]))
    coverage = track_coverage(ordered, total_frames)
    detection_confidences = np.asarray(
        [item["detection_confidence"] for item in ordered], dtype=np.float64
    )
    pose_confidences = np.asarray(
        [item["pose_confidence"] for item in ordered], dtype=np.float64
    )
    centers = np.asarray([item["center"] for item in ordered], dtype=np.float64)
    sizes = np.asarray([item["size"] for item in ordered], dtype=np.float64)
    zone_fraction = float(np.mean([bool(item["inside_teacher_zone"]) for item in ordered]))
    mean_detection = float(np.clip(detection_confidences.mean(), 0.0, 1.0))
    mean_pose = float(np.clip(pose_confidences.mean(), 0.0, 1.0))

    motions = []
    for previous, current in zip(ordered, ordered[1:]):
        frame_delta = max(1, int(current["frame_index"]) - int(previous["frame_index"]))
        motions.append(
            float(np.linalg.norm(np.asarray(current["center"]) - np.asarray(previous["center"])))
            / frame_delta
        )
    mean_motion = float(np.mean(motions)) if motions else 0.0
    alignment, alignment_reliability = reliability_weighted_mean(
        [item.get("instruction_alignment_proxy", 0.0) for item in ordered],
        [
            item.get("instruction_alignment_reliability", 0.0)
            * item["detection_confidence"]
            for item in ordered
        ],
    )
    spatial_weights = np.clip(detection_confidences, 0.0, 1.0)
    return {
        "coverage": coverage,
        "zone_fraction": zone_fraction,
        "mean_detection_confidence": mean_detection,
        "mean_pose_confidence": mean_pose,
        "mean_motion": mean_motion,
        "mean_x": reliability_weighted_mean(centers[:, 0], spatial_weights)[0],
        "mean_y": reliability_weighted_mean(centers[:, 1], spatial_weights)[0],
        "mean_width": reliability_weighted_mean(sizes[:, 0], spatial_weights)[0],
        "mean_height": reliability_weighted_mean(sizes[:, 1], spatial_weights)[0],
        "instruction_alignment_proxy": alignment,
        "instruction_alignment_reliability": alignment_reliability,
        "teacher_score": compute_teacher_score(
            zone_fraction, coverage, mean_detection, teacher_score_weights
        ),
    }


def summarize_tracks(
    tracks: Mapping[int, Sequence[Mapping]],
    total_frames: int,
    teacher_score_weights: Sequence[float],
) -> dict[int, dict[str, float]]:
    return {
        int(track_id): summarize_track(
            observations, total_frames, teacher_score_weights
        )
        for track_id, observations in tracks.items()
    }


def build_frame_descriptor(
    observations: Sequence[Mapping],
    summaries: Mapping[int, Mapping[str, float]],
    assignments: Mapping[int, Mapping],
    teacher_track_id: int | None,
    *,
    close_distance: float,
    alignment_threshold: float,
) -> np.ndarray:
    """Build one permutation-invariant 40-D frame descriptor."""
    tracked = [item for item in observations if item.get("track_id") is not None]
    untracked_count = sum(item.get("track_id") is None for item in observations)
    by_track = {int(item["track_id"]): item for item in tracked}
    teacher_observation = by_track.get(teacher_track_id) if teacher_track_id is not None else None
    teacher_summary = summaries.get(teacher_track_id, {}) if teacher_track_id is not None else {}
    teacher_assignment = assignments.get(teacher_track_id, {}) if teacher_track_id is not None else {}
    teacher_frame_reliability = (
        float(teacher_observation["detection_confidence"])
        if teacher_observation is not None
        else 0.0
    )

    student_track_ids = [
        track_id for track_id, assignment in assignments.items() if assignment["role"] == "student"
    ]
    student_observations = [
        by_track[track_id] for track_id in student_track_ids if track_id in by_track
    ]
    spatial_weights = [
        float(
            np.clip(
                summaries[int(item["track_id"])]["coverage"]
                * summaries[int(item["track_id"])]["mean_detection_confidence"]
                * summaries[int(item["track_id"])]["mean_pose_confidence"]
                * item["detection_confidence"],
                0.0,
                1.0,
            )
        )
        for item in student_observations
    ]

    def student_mean(key: str) -> float:
        return reliability_weighted_mean(
            [item[key] for item in student_observations], spatial_weights
        )[0]

    mean_x = reliability_weighted_mean(
        [item["center"][0] for item in student_observations], spatial_weights
    )[0]
    mean_y = reliability_weighted_mean(
        [item["center"][1] for item in student_observations], spatial_weights
    )[0]
    spread_x = reliability_weighted_std(
        [item["center"][0] for item in student_observations], spatial_weights
    )[0]
    spread_y = reliability_weighted_std(
        [item["center"][1] for item in student_observations], spatial_weights
    )[0]

    alignment_weights = [
        weight * item["instruction_alignment_reliability"]
        for weight, item in zip(spatial_weights, student_observations)
    ]
    instruction_alignment, instruction_reliability = reliability_weighted_mean(
        [item["instruction_alignment_proxy"] for item in student_observations],
        alignment_weights,
    )
    aligned_fraction, aligned_reliability = reliability_weighted_mean(
        [
            float(item["instruction_alignment_proxy"] >= alignment_threshold)
            for item in student_observations
        ],
        alignment_weights,
    )

    pairwise = pairwise_distance_aggregation(
        [item["center"] for item in student_observations],
        spatial_weights,
        close_distance=close_distance,
    )

    teacher_distances = []
    teacher_distance_weights = []
    if teacher_observation is not None:
        for item, weight in zip(student_observations, spatial_weights):
            teacher_distances.append(
                float(
                    np.linalg.norm(
                        np.asarray(item["center"], dtype=np.float64)
                        - np.asarray(teacher_observation["center"], dtype=np.float64)
                    )
                )
            )
            teacher_distance_weights.append(weight * teacher_frame_reliability)
    mean_teacher_distance, teacher_distance_reliability = reliability_weighted_mean(
        teacher_distances, teacher_distance_weights
    )

    peer_alignment_values = []
    peer_alignment_weights = []
    if len(student_observations) >= 2:
        for index, (item, weight) in enumerate(zip(student_observations, spatial_weights)):
            other_centers = [
                other["center"]
                for other_index, other in enumerate(student_observations)
                if other_index != index
            ]
            target = np.mean(np.asarray(other_centers, dtype=np.float64), axis=0)
            value, reliability = alignment_from_vector(
                item["center"],
                target,
                item.get("orientation_vector"),
                item.get("orientation_reliability", 0.0),
            )
            peer_alignment_values.append(float(value >= alignment_threshold))
            peer_alignment_weights.append(weight * reliability)
    peer_fraction, peer_reliability = reliability_weighted_mean(
        peer_alignment_values, peer_alignment_weights
    )

    visibility_fraction = (
        len(student_observations) / len(student_track_ids) if student_track_ids else 0.0
    )
    interaction_reliability = (
        float(np.mean(spatial_weights)) * visibility_fraction if spatial_weights else 0.0
    )
    unknown_count = sum(
        assignments.get(int(item["track_id"]), {}).get("role") == "unknown"
        for item in tracked
    )
    detection_reliability = (
        float(np.mean([item["detection_confidence"] for item in observations]))
        if observations
        else 0.0
    )

    values = [
        float(teacher_observation is not None),
        float(teacher_observation["center"][0]) if teacher_observation is not None else 0.0,
        float(teacher_observation["center"][1]) if teacher_observation is not None else 0.0,
        float(teacher_summary.get("coverage", 0.0)),
        float(teacher_summary.get("mean_detection_confidence", 0.0)),
        float(teacher_summary.get("mean_motion", 0.0)),
        float(teacher_assignment.get("role_confidence", 0.0)),
        teacher_frame_reliability,
        float(len(student_observations)),
        float(len(student_track_ids)),
        mean_x,
        mean_y,
        spread_x,
        spread_y,
        student_mean("size_x"),
        student_mean("size_y"),
        reliability_weighted_mean(
            [summaries[int(item["track_id"])]["coverage"] for item in student_observations],
            spatial_weights,
        )[0],
        reliability_weighted_mean(
            [item["detection_confidence"] for item in student_observations], spatial_weights
        )[0],
        reliability_weighted_mean(
            [item["pose_confidence"] for item in student_observations], spatial_weights
        )[0],
        reliability_weighted_mean(
            [summaries[int(item["track_id"])]["mean_motion"] for item in student_observations],
            spatial_weights,
        )[0],
        instruction_alignment,
        instruction_reliability,
        aligned_fraction,
        aligned_reliability,
        interaction_reliability,
        pairwise["mean"],
        pairwise["std"],
        pairwise["minimum"],
        pairwise["close_fraction"],
        pairwise["compactness"],
        pairwise["reliability"],
        mean_teacher_distance,
        teacher_distance_reliability,
        peer_fraction,
        peer_reliability,
        float(unknown_count),
        float(untracked_count),
        float(len(summaries)),
        float(visibility_fraction),
        detection_reliability,
    ]
    descriptor = np.asarray(values, dtype=np.float32)
    if descriptor.shape != (len(TRACK_INTERACTION_COLUMNS),):
        raise AssertionError(f"internal interaction shape error: {descriptor.shape}")
    if not np.isfinite(descriptor).all():
        raise ValueError("interaction descriptor contains NaN or Inf")
    return descriptor

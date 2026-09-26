"""Lightweight clip-local association across the eight preprocessed frames."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matched_motion(previous, current):
    distances = [float(np.linalg.norm(position - previous[track]))
                 for track, position in current.items() if track in previous]
    if not distances:
        return 0.0, 0.0, 0.0, 0.0
    values = np.asarray(distances)
    return (float(values.mean()), float(values.std()),
            float((values < .01).mean()), float((values > .04).mean()))


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _pose_distance(a, b):
    valid = (a["conf"][:11] > .3) & (b["conf"][:11] > .3)
    if not valid.any():
        return .5
    scale = max(a["box"][3] - a["box"][1], b["box"][3] - b["box"][1], 1.0)
    return float(np.linalg.norm(a["keypoints"][:11][valid] - b["keypoints"][:11][valid], axis=1).mean() / scale)


class SampledPersonTracker:
    """Hungarian box/pose association; IDs exist only within one eight-frame clip."""
    def __init__(self, max_center_distance=.22, max_gap=2,
                 center_weight=.55, iou_weight=.30, pose_weight=.15):
        self.max_center_distance = max_center_distance
        self.max_gap = max_gap
        self.weights = (center_weight, iou_weight, pose_weight)
        if max_center_distance <= 0 or max_gap < 0 or not np.isclose(sum(self.weights), 1):
            raise ValueError("Invalid sampled tracking settings")
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, sample_index):
        active = [(track_id, state) for track_id, state in self.tracks.items()
                  if sample_index - state["sample_index"] <= self.max_gap + 1]
        ids = [None] * len(detections)
        if active and detections:
            cost = np.full((len(active), len(detections)), 1e3, dtype=np.float32)
            for row, (_, previous) in enumerate(active):
                for col, current in enumerate(detections):
                    center_distance = float(np.linalg.norm(previous["center"] - current["center"]))
                    if center_distance <= self.max_center_distance:
                        cost[row, col] = (self.weights[0] * center_distance / self.max_center_distance
                                          + self.weights[1] * (1 - _iou(previous["box"], current["box"]))
                                          + self.weights[2] * min(_pose_distance(previous, current), 1.0))
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] < 1.0:
                    ids[col] = active[row][0]
        for index, detection in enumerate(detections):
            if ids[index] is None:
                ids[index] = self.next_id
                self.next_id += 1
            self.tracks[ids[index]] = {**detection, "sample_index": sample_index}
        return ids


def _detections(result):
    if len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.numpy()
    keypoints = result.keypoints.xy.numpy()
    confidences = (result.keypoints.conf.numpy() if result.keypoints.conf is not None
                   else np.ones((len(boxes), 17), dtype=np.float32))
    return [{"box": box, "center": (box[:2] + box[2:]) / 1280.0,
             "keypoints": keypoints[i], "conf": confidences[i]}
            for i, box in enumerate(boxes)]


def track_sampled_frames(model, frames_rgb, device, confidence, settings,
                         zone, fallback, preview_path=None):
    """Detect once as an 8-image batch, then associate only those observations."""
    if frames_rgb.shape != (8, 640, 640, 3) or frames_rgb.dtype != np.uint8:
        raise ValueError("Expected eight 640x640 uint8 RGB frames")
    frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_rgb]
    results = model.predict(source=frames_bgr, conf=confidence, classes=[0], imgsz=640,
                            device=device, verbose=False)
    if len(results) != 8:
        raise ValueError("Pose model did not return exactly eight results")
    results = [result.cpu() for result in results]
    if any(result.keypoints is None for result in results):
        raise ValueError("A YOLO pose checkpoint is required")
    tracker = SampledPersonTracker(**settings)
    known_speakers, summaries = set(), {}
    all_ids, all_exclusions, samples = [], [], []
    zone_min_x, zone_max_x, zone_min_y, zone_max_y = zone
    writer = None
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640, 640))
        if not writer.isOpened():
            raise OSError("Could not create preview " + str(preview_path))
    try:
        for sample_index, (frame, result) in enumerate(zip(frames_bgr, results)):
            detections = _detections(result)
            ids = tracker.update(detections, sample_index)
            centers = [d["center"] for d in detections]
            speaker = next((i for i, (x, y) in enumerate(centers)
                            if zone_min_x <= x <= zone_max_x and zone_min_y <= y <= zone_max_y), None)
            if speaker is not None:
                known_speakers.add(ids[speaker])
            exclusions = {i for i, track_id in enumerate(ids) if track_id in known_speakers}
            target = centers[speaker] if speaker is not None else np.asarray(fallback)
            records = []
            for i, (detection, track_id) in enumerate(zip(detections, ids)):
                summary = summaries.setdefault(track_id, {"track_id": track_id,
                    "first_sample": sample_index, "last_sample": sample_index,
                    "observations": 0, "gap_returns": 0})
                if summary["observations"] and sample_index > summary["last_sample"] + 1:
                    summary["gap_returns"] += 1
                summary["last_sample"] = sample_index
                summary["observations"] += 1
                in_zone = zone_min_x <= detection["center"][0] <= zone_max_x
                records.append({"detection_index": i, "track_id": track_id,
                    "box_xyxy": detection["box"].tolist(), "confidence": float(result.boxes.conf[i]),
                    "role": "zone_person" if in_zone else "known_speaker" if i in exclusions else "listener",
                    "keypoints_xy": detection["keypoints"].tolist(),
                    "keypoint_confidence": detection["conf"].tolist()})
            all_ids.append(ids); all_exclusions.append(exclusions)
            samples.append({"sample_index": sample_index,
                "target_source": "speaker" if speaker is not None else "zone_center",
                "target_xy": np.asarray(target).tolist(), "people": records})
            if writer is not None:
                overlay = frame.copy()
                cv2.rectangle(overlay, (int(zone_min_x * 640), int(zone_min_y * 640)),
                              (int(zone_max_x * 640), int(zone_max_y * 640)), (255, 190, 0), 2)
                for i, detection in enumerate(detections):
                    x1, y1, x2, y2 = detection["box"].astype(int)
                    color = (255, 190, 0) if i in exclusions else (50, 230, 100)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(overlay, f"ID {ids[i]}", (x1, max(18, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
                writer.write(overlay)
    finally:
        if writer is not None:
            writer.release()
    details = {"tracking_updates": 8, "sample_indices": list(range(8)),
               "tracks": list(summaries.values()), "samples": samples,
               "id_scope": "anonymous within this clip; eight sampled frames only",
               "known_speaker_track_ids": sorted(known_speakers)}
    return results, all_ids, all_exclusions, details

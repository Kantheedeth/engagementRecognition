#!/usr/bin/env python3
"""Render an audit view from saved interaction-track JSON without rerunning inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TRACK_DIR = PROJECT_ROOT / "debug_validation" / "interaction_tracks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "audit_outputs" / "interaction_tracks"
ROLE_COLORS = {
    "teacher": (255, 190, 40),
    "student": (40, 210, 70),
    "unknown": (160, 160, 160),
}


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.42,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def render_frame(
    frame_rgb: np.ndarray,
    frame_details: dict,
    teacher_zone: dict,
    video_name: str,
    panel_rows: int,
) -> np.ndarray:
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    x1 = int(teacher_zone["x_min"] * width)
    x2 = int(teacher_zone["x_max"] * width)
    y1 = int(teacher_zone["y_min"] * height)
    y2 = int(teacher_zone["y_max"] * height)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 190, 40), -1)
    image = cv2.addWeighted(overlay, 0.14, image, 0.86, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 190, 40), 2)
    put_text(image, "Configured instruction zone", (x1 + 8, max(20, y1 + 22)))

    for detection in frame_details.get("detections", []):
        box = [int(round(value)) for value in detection["bbox_xyxy"]]
        box_x1, box_y1, box_x2, box_y2 = box
        role = detection.get("role", "unknown")
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, 2)
        track_id = detection.get("track_id")
        identifier = "?" if track_id is None else str(track_id)
        reliability = detection.get("instruction_alignment_reliability", 0.0)
        put_text(
            image,
            f"ID {identifier} | {role}",
            (box_x1, max(50, box_y1 - 5)),
            color=color,
        )
        vector = detection.get("orientation_vector")
        if vector is not None and reliability > 0.0:
            center = detection["center"]
            start = (int(center[0] * width), int(center[1] * height))
            end = (
                int(start[0] + vector[0] * 45),
                int(start[1] + vector[1] * 45),
            )
            cv2.arrowedLine(image, start, end, color, 2, tipLength=0.25)

    banner = image.copy()
    cv2.rectangle(banner, (0, 0), (width, 36), (15, 15, 15), -1)
    image = cv2.addWeighted(banner, 0.85, image, 0.15, 0)
    put_text(
        image,
        f"{video_name} | frame {frame_details['frame_index'] + 1} | saved inference audit",
        (10, 24),
        scale=0.50,
    )

    panel_height = 31 + panel_rows * 19
    canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
    canvas[:height] = image
    put_text(
        canvas,
        "track / role | detector | pose | instruction alignment (reliability)",
        (8, height + 19),
        scale=0.38,
    )
    detections = frame_details.get("detections", [])
    for index, detection in enumerate(detections):
        column = index // panel_rows
        row = index % panel_rows
        x = 8 + column * (width // 2)
        y = height + 39 + row * 19
        role = detection.get("role", "unknown")
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        track_id = detection.get("track_id")
        identifier = "?" if track_id is None else str(track_id)
        det_confidence = detection.get("detection_confidence", 0.0)
        pose_confidence = detection.get("pose_confidence", 0.0)
        alignment = detection.get("instruction_alignment_proxy", 0.0)
        reliability = detection.get("instruction_alignment_reliability", 0.0)
        reason = detection.get("exclusion_reason")
        suffix = f" | excluded: {reason}" if reason else ""
        put_text(
            canvas,
            f"ID {identifier} {role} | {det_confidence:.2f} | {pose_confidence:.2f} | "
            f"{alignment:.2f} ({reliability:.2f}){suffix}",
            (x, y),
            color=color,
            scale=0.34,
        )
    return canvas


def render_track_json(track_json: Path, output_dir: Path) -> Path:
    with track_json.open(encoding="utf-8") as file:
        details = json.load(file)
    if details.get("track_id_scope") != "local_to_this_video_only":
        raise ValueError(f"Unsupported or missing track scope in {track_json}")
    video_path = Path(details["video"])
    if not video_path.is_file():
        raise FileNotFoundError(f"Saved source frame archive not found: {video_path}")
    with np.load(video_path, allow_pickle=False) as archive:
        frames_rgb = archive["frames"]
    frame_details = details.get("frames", [])
    if frames_rgb.shape != (8, 640, 640, 3) or len(frame_details) != 8:
        raise ValueError(
            f"Expected 8 saved 640x640 frames and 8 detail records for {track_json}"
        )

    destination = output_dir / f"{video_path.parent.parent.name}_{video_path.parent.name}_{video_path.stem}"
    destination.mkdir(parents=True, exist_ok=True)
    maximum_detections = max(
        (len(record.get("detections", [])) for record in frame_details),
        default=0,
    )
    panel_rows = max(1, math.ceil(maximum_detections / 2))
    rendered = []
    for frame_rgb, record in zip(frames_rgb, frame_details):
        image = render_frame(
            frame_rgb,
            record,
            details["teacher_zone"],
            video_path.stem,
            panel_rows,
        )
        rendered.append(image)
        cv2.imwrite(
            str(destination / f"frame_{record['frame_index'] + 1}.png"), image
        )

    montage = np.vstack((np.hstack(rendered[:4]), np.hstack(rendered[4:])))
    montage_path = output_dir / f"montage_{video_path.parent.name}_{video_path.stem}.png"
    cv2.imwrite(str(montage_path), montage)
    print(f"Saved interaction audit montage: {montage_path}")
    return montage_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_json", type=Path)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--track_dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.track_json is not None:
        paths = [args.track_json.expanduser().resolve()]
    elif args.sample:
        paths = []
        for category in ("low", "mid", "high"):
            candidates = sorted(args.track_dir.rglob(f"{category}/*.json"))
            if candidates:
                paths.append(candidates[0])
    else:
        parser.error("provide --track_json or --sample")
    if not paths:
        raise FileNotFoundError(
            f"No track JSON found under {args.track_dir}; extract with "
            "--save_track_details first."
        )
    for path in paths:
        render_track_json(path, args.output_dir)


if __name__ == "__main__":
    main()

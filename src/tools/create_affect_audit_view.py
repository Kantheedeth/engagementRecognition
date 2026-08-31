"""Create visual and tabular audits from saved per-face affect tracks.

This script does not run face detection, tracking, or emotion inference again.
It renders the exact frames and values already saved by
``extract_affect_features.py --save_track_details``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "debug_validation" / "affect_audits"
TRACK_COLORS = (
    (63, 208, 244),
    (76, 175, 80),
    (244, 140, 66),
    (156, 106, 222),
    (66, 165, 245),
    (219, 112, 147),
    (0, 188, 212),
    (139, 195, 74),
)
UNCONFIRMED_COLOR = (155, 155, 155)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an auditable view of saved anonymous affect tracks"
    )
    parser.add_argument(
        "--track_json",
        type=Path,
        required=True,
        help="JSON produced by extract_affect_features.py --save_track_details",
    )
    parser.add_argument(
        "--frames_npz",
        type=Path,
        help="Override the preprocessed .npz path recorded in the track JSON",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--seconds_per_frame", type=float, default=1.5)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return UNCONFIRMED_COLOR
    return TRACK_COLORS[(int(track_id) - 1) % len(TRACK_COLORS)]


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (238, 238, 238),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def put_boxed_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, baseline_y = origin
    x = max(0, min(x, image.shape[1] - width - 6))
    baseline_y = max(height + 5, min(baseline_y, image.shape[0] - baseline - 3))
    cv2.rectangle(
        image,
        (x, baseline_y - height - 5),
        (x + width + 6, baseline_y + baseline + 3),
        color,
        -1,
    )
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (20, 20, 20) if luminance > 145 else (255, 255, 255)
    cv2.putText(
        image,
        text,
        (x + 3, baseline_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def face_crop(frame_bgr: np.ndarray, box: list[float], size: int = 76) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = max(4.0, (x2 - x1) * 0.35)
    margin_y = max(4.0, (y2 - y1) * 0.25)
    left = max(0, int(math.floor(x1 - margin_x)))
    top = max(0, int(math.floor(y1 - margin_y)))
    right = min(width, int(math.ceil(x2 + margin_x)))
    bottom = min(height, int(math.ceil(y2 + margin_y)))
    if right <= left or bottom <= top:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame_bgr[top:bottom, left:right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def render_card(
    frame_rgb: np.ndarray,
    frame_details: dict,
    video_name: str,
    frame_count: int,
    card_height: int,
) -> np.ndarray:
    source_height, source_width = frame_rgb.shape[:2]
    if (source_width, source_height) != (640, 640):
        raise ValueError(
            f"Audit expects 640x640 saved frames, got {source_width}x{source_height}"
        )

    header_height = 72
    side_width = 440
    card_width = source_width + side_width
    canvas = np.full((card_height, card_width, 3), (27, 30, 35), dtype=np.uint8)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    canvas[header_height : header_height + source_height, :source_width] = frame_bgr

    frame_index = int(frame_details["frame_index"])
    group_affect = frame_details["group_affect"]
    group_top = max(group_affect, key=group_affect.get)
    group_confidence = float(group_affect[group_top])
    reliability = float(frame_details["group_reliability"])

    put_text(
        canvas,
        f"{video_name}  |  sampled frame {frame_index + 1}/{frame_count}",
        (18, 30),
        scale=0.72,
        thickness=2,
    )
    put_text(
        canvas,
        (
            f"Group estimate: {group_top} {group_confidence:.1%}  |  "
            f"reliability {reliability:.1%}"
        ),
        (18, 58),
        scale=0.56,
        color=(190, 205, 220),
    )

    for track in frame_details["tracks"]:
        box = [float(value) for value in track["bbox_xyxy"]]
        x1, y1, x2, y2 = box
        x1 = int(np.clip(round(x1), 0, source_width - 1))
        x2 = int(np.clip(round(x2), 0, source_width - 1))
        y1 = int(np.clip(round(y1), 0, source_height - 1)) + header_height
        y2 = int(np.clip(round(y2), 0, source_height - 1)) + header_height
        color = track_color(track["track_id"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        identity = (
            f"ID {track['track_id']}"
            if track["track_id"] is not None
            else "unconfirmed"
        )
        label = (
            f"{identity} | {track['top_emotion']} "
            f"{float(track['emotion_confidence']):.0%}"
        )
        label_y = y1 - 5 if y1 - 5 > header_height + 14 else y2 + 18
        put_boxed_text(canvas, label, (x1, label_y), color)

    panel_x = source_width + 18
    put_text(canvas, "GROUP AFFECT", (panel_x, 106), scale=0.57, thickness=2)
    group_ranked = sorted(group_affect.items(), key=lambda item: item[1], reverse=True)
    group_line = "  ".join(
        f"{name} {float(value):.0%}" for name, value in group_ranked[:3]
    )
    put_text(canvas, group_line, (panel_x, 132), scale=0.46, color=(198, 211, 222))
    put_text(
        canvas,
        f"Reliability: {reliability:.1%}",
        (panel_x, 157),
        scale=0.49,
        color=(198, 211, 222),
    )
    cv2.line(canvas, (panel_x, 174), (card_width - 18, 174), (72, 77, 84), 1)

    row_top = 190
    for row_index, track in enumerate(frame_details["tracks"]):
        y = row_top + row_index * 112
        color = track_color(track["track_id"])
        crop = face_crop(frame_bgr, track["bbox_xyxy"])
        canvas[y : y + 76, panel_x : panel_x + 76] = crop
        cv2.rectangle(canvas, (panel_x, y), (panel_x + 76, y + 76), color, 2)

        identity = (
            f"TRACK ID {track['track_id']}"
            if track["track_id"] is not None
            else "UNCONFIRMED DETECTION"
        )
        text_x = panel_x + 92
        put_text(canvas, identity, (text_x, y + 17), scale=0.49, color=color, thickness=2)
        put_text(
            canvas,
            f"{track['top_emotion']}: {float(track['emotion_confidence']):.1%}",
            (text_x, y + 41),
            scale=0.48,
        )
        put_text(
            canvas,
            f"face detection: {float(track['detection_confidence']):.1%}",
            (text_x, y + 63),
            scale=0.43,
            color=(188, 198, 208),
        )
        ranked = sorted(
            track["emotion_probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        alternatives = " / ".join(
            f"{name} {float(value):.0%}" for name, value in ranked[1:3]
        )
        put_text(
            canvas,
            alternatives,
            (text_x, y + 84),
            scale=0.39,
            color=(158, 170, 181),
        )
        cv2.line(
            canvas,
            (panel_x, y + 100),
            (card_width - 18, y + 100),
            (58, 63, 70),
            1,
        )

    footer_y = card_height - 22
    put_text(
        canvas,
        "Anonymous per-video tracks. Facial-expression estimates are not identity or ground truth.",
        (18, footer_y),
        scale=0.45,
        color=(155, 165, 175),
    )
    return canvas


def write_csv(path: Path, frames: list[dict]) -> None:
    fieldnames = [
        "frame_index",
        "track_id",
        "tracking_status",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "detection_confidence",
        "top_emotion",
        "emotion_confidence",
        "group_top_emotion",
        "group_top_confidence",
        "group_reliability",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            group_top = max(frame["group_affect"], key=frame["group_affect"].get)
            for track in frame["tracks"]:
                x1, y1, x2, y2 = track["bbox_xyxy"]
                writer.writerow(
                    {
                        "frame_index": frame["frame_index"],
                        "track_id": track["track_id"],
                        "tracking_status": track["tracking_status"],
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "detection_confidence": track["detection_confidence"],
                        "top_emotion": track["top_emotion"],
                        "emotion_confidence": track["emotion_confidence"],
                        "group_top_emotion": group_top,
                        "group_top_confidence": frame["group_affect"][group_top],
                        "group_reliability": frame["group_reliability"],
                    }
                )


def build_summary(
    track_json: Path,
    frames_npz: Path,
    frames: list[dict],
) -> dict:
    per_track: dict[int, dict[str, object]] = defaultdict(
        lambda: {
            "observed_frames": [],
            "top_emotions": Counter(),
            "emotion_confidences": [],
            "detection_confidences": [],
        }
    )
    confirmed_observations = 0
    unconfirmed_observations = 0
    for frame in frames:
        for track in frame["tracks"]:
            track_id = track["track_id"]
            if track_id is None:
                unconfirmed_observations += 1
                continue
            confirmed_observations += 1
            record = per_track[int(track_id)]
            record["observed_frames"].append(int(frame["frame_index"]))
            record["top_emotions"][track["top_emotion"]] += 1
            record["emotion_confidences"].append(float(track["emotion_confidence"]))
            record["detection_confidences"].append(float(track["detection_confidence"]))

    track_summary = {}
    for track_id, record in sorted(per_track.items()):
        track_summary[str(track_id)] = {
            "observed_frames": record["observed_frames"],
            "observation_count": len(record["observed_frames"]),
            "top_emotion_counts": dict(record["top_emotions"]),
            "mean_emotion_confidence": float(np.mean(record["emotion_confidences"])),
            "mean_detection_confidence": float(np.mean(record["detection_confidences"])),
        }

    return {
        "sources": {
            "track_json": str(track_json.resolve()),
            "track_json_sha256": file_sha256(track_json),
            "frames_npz": str(frames_npz.resolve()),
            "frames_npz_sha256": file_sha256(frames_npz),
        },
        "frame_count": len(frames),
        "confirmed_track_ids": sorted(per_track),
        "confirmed_observations": confirmed_observations,
        "unconfirmed_observations": unconfirmed_observations,
        "tracks": track_summary,
        "interpretation_warning": (
            "Track IDs are anonymous and local to this video. ID changes may be "
            "tracker fragmentation, not different students. Facial-expression "
            "outputs are model estimates, not ground-truth emotions."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.video_fps <= 0 or args.seconds_per_frame <= 0:
        raise ValueError("--video_fps and --seconds_per_frame must be positive")
    track_json = args.track_json.expanduser().resolve()
    if not track_json.is_file():
        raise FileNotFoundError(track_json)

    with track_json.open(encoding="utf-8") as file:
        payload = json.load(file)
    details = payload.get("frames")
    if not isinstance(details, list) or not details:
        raise ValueError(f"No frame records found in {track_json}")

    frames_npz = (
        args.frames_npz.expanduser().resolve()
        if args.frames_npz
        else Path(payload["video"]).expanduser().resolve()
    )
    if not frames_npz.is_file():
        raise FileNotFoundError(
            f"Saved frame archive not found: {frames_npz}. Use --frames_npz to override it."
        )
    with np.load(frames_npz) as archive:
        frames_rgb = archive["frames"]
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape in {frames_npz}: {frames_rgb.shape}")
    if len(details) != len(frames_rgb):
        raise ValueError(
            f"JSON has {len(details)} frames but archive has {len(frames_rgb)} frames"
        )
    expected_indices = list(range(len(details)))
    actual_indices = [int(frame["frame_index"]) for frame in details]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Frame records must be ordered {expected_indices}; got {actual_indices}"
        )

    max_tracks = max(len(frame["tracks"]) for frame in details)
    card_height = max(760, 190 + max_tracks * 112 + 48)
    video_name = track_json.stem
    cards = [
        render_card(frame, frame_details, video_name, len(details), card_height)
        for frame, frame_details in zip(frames_rgb, details, strict=True)
    ]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{video_name}_affect_audit.png"
    mp4_path = output_dir / f"{video_name}_affect_audit.mp4"
    csv_path = output_dir / f"{video_name}_affect_audit.csv"
    summary_path = output_dir / f"{video_name}_affect_audit_summary.json"

    columns = 2
    rows = math.ceil(len(cards) / columns)
    gap = 8
    card_height, card_width = cards[0].shape[:2]
    contact_sheet = np.full(
        (
            rows * card_height + (rows - 1) * gap,
            columns * card_width + (columns - 1) * gap,
            3,
        ),
        (12, 14, 17),
        dtype=np.uint8,
    )
    for index, card in enumerate(cards):
        row, column = divmod(index, columns)
        top = row * (card_height + gap)
        left = column * (card_width + gap)
        contact_sheet[top : top + card_height, left : left + card_width] = card
    if not cv2.imwrite(str(png_path), contact_sheet):
        raise RuntimeError(f"Could not write {png_path}")

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.video_fps,
        (card_width, card_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not initialize MP4 writer for {mp4_path}")
    repeats = max(1, round(args.video_fps * args.seconds_per_frame))
    for card in cards:
        for _ in range(repeats):
            writer.write(card)
    writer.release()

    write_csv(csv_path, details)
    summary = build_summary(track_json, frames_npz, details)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Affect audit created from saved data (no inference was rerun):")
    print(f"  Contact sheet: {png_path}")
    print(f"  Audit video:   {mp4_path}")
    print(f"  Track table:   {csv_path}")
    print(f"  Provenance:    {summary_path}")


if __name__ == "__main__":
    main()

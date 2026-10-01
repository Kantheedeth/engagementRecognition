"""Create a blinded visual audit comparing H04 with the other High groups.

The audit renders the exact eight stored frames and saved YOLO-pose/Hungarian
tracking records used by V3. It does not run inference again, modify features,
or tune thresholds. Group identities are hidden in the normal review view and
can be revealed only for the post-annotation comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_schema import INTERACTION_COLUMNS


ZONE = (0.0, 0.27, 0.0, 1.0)
POSE_THRESHOLD = 0.30
SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)
TRACK_COLORS = (
    (60, 190, 90), (210, 130, 55), (180, 90, 210), (55, 165, 220),
    (210, 90, 135), (80, 190, 190), (135, 180, 70), (190, 135, 80),
)
SPEAKER_COLOR = (0, 175, 255)
UNKNOWN_COLOR = (150, 150, 150)
SELECTED_FEATURES = (
    "raised_head_ratio", "lowered_head_ratio", "mean_orientation_to_target",
    "orientation_consensus_ratio", "upright_ratio",
    "very_low_head_proxy_ratio", "student_count_norm", "mean_face_visibility",
    "posture_observation_ratio", "matched_center_displacement_mean_x10",
    "matched_stillness_ratio", "matched_high_motion_ratio", "speaker_in_zone",
    "orientation_alignment",
)
UF_DATALESS = 0x40000000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def view_number(key: str) -> int:
    match = re.fullmatch(r"high/view(\d+)\.mp4", key)
    if not match:
        raise ValueError(f"Expected a High clip key, got {key!r}")
    return int(match.group(1))


def load_high_episodes(protocol_path: Path) -> list[dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    episodes = [episode for episode in protocol["episodes"] if episode["class"] == "high"]
    ids = {episode["episode_id"] for episode in episodes}
    if ids != {"H01", "H02", "H03", "H04"}:
        raise ValueError(f"Expected H01--H04 in protocol, got {sorted(ids)}")
    return episodes


def episode_for(number: int, episodes: list[dict]) -> str:
    matches = [episode["episode_id"] for episode in episodes
               if episode["start_view"] <= number <= episode["end_view"]]
    if len(matches) != 1:
        raise ValueError(f"View {number} maps to {matches}, expected one High group")
    return matches[0]


def track_color(track_id: int | None, speaker: bool) -> tuple[int, int, int]:
    if speaker:
        return SPEAKER_COLOR
    if track_id is None:
        return UNKNOWN_COLOR
    return TRACK_COLORS[(int(track_id) - 1) % len(TRACK_COLORS)]


def boxed_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(np.clip(origin[0], 0, image.shape[1] - width - 7))
    y = int(np.clip(origin[1], height + 5, image.shape[0] - baseline - 3))
    cv2.rectangle(image, (x, y - height - 5), (x + width + 7, y + baseline + 3), color, -1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    foreground = (20, 20, 20) if luminance > 145 else (255, 255, 255)
    cv2.putText(image, text, (x + 3, y), font, scale, foreground, thickness, cv2.LINE_AA)


def orientation_vector(person: dict) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(person["keypoints_xy"], dtype=np.float32)
    confidence = np.asarray(person["keypoint_confidence"], dtype=np.float32)
    nose, left_eye, right_eye = points[0], points[1], points[2]
    left_shoulder, right_shoulder = points[5], points[6]
    has_shoulders = confidence[5] > POSE_THRESHOLD and confidence[6] > POSE_THRESHOLD

    box = np.asarray(person["box_xyxy"], dtype=np.float32)
    height = max(float(box[3] - box[1]), 1e-4)
    elevation_observed = confidence[0] > POSE_THRESHOLD and has_shoulders
    if elevation_observed:
        shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
        elevation = (shoulder_y - float(nose[1])) / height
        if elevation <= 0.06:
            return None

    if confidence[0] > POSE_THRESHOLD and (
        confidence[1] > POSE_THRESHOLD or confidence[2] > POSE_THRESHOLD
    ):
        if confidence[1] > POSE_THRESHOLD and confidence[2] > POSE_THRESHOLD:
            eye_mid = (left_eye + right_eye) / 2.0
        else:
            eye_mid = left_eye if confidence[1] > POSE_THRESHOLD else right_eye
        vector = nose - eye_mid
        origin = nose
    elif has_shoulders:
        shoulder_axis = right_shoulder - left_shoulder
        vector = np.array([-shoulder_axis[1], shoulder_axis[0]], dtype=np.float32)
        origin = (left_shoulder + right_shoulder) / 2.0
    else:
        return None
    norm = float(np.linalg.norm(vector))
    return (origin, vector / norm) if norm > 1e-4 else None


def render_overlay(frame_rgb: np.ndarray, sample: dict, known_speakers: set[int]) -> np.ndarray:
    if frame_rgb.shape != (640, 640, 3) or frame_rgb.dtype != np.uint8:
        raise ValueError(f"Expected a 640x640 uint8 frame, got {frame_rgb.shape}/{frame_rgb.dtype}")
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    x0, x1, y0, y1 = ZONE
    cv2.rectangle(canvas, (int(x0 * 640), int(y0 * 640)),
                  (int(x1 * 640), int(y1 * 640) - 1), SPEAKER_COLOR, 2)
    target = np.asarray(sample["target_xy"], dtype=np.float32) * 640.0
    cv2.drawMarker(canvas, tuple(np.rint(target).astype(int)), (0, 255, 255),
                   cv2.MARKER_CROSS, 18, 2)

    for person in sample["people"]:
        track_id = person.get("track_id")
        speaker = person["role"] in ("zone_person", "known_speaker") or track_id in known_speakers
        color = track_color(track_id, speaker)
        box = np.asarray(person["box_xyxy"], dtype=np.float32)
        x1b, y1b, x2b, y2b = np.rint(box).astype(int)
        cv2.rectangle(canvas, (x1b, y1b), (x2b, y2b), color, 2)
        role = "S" if speaker else "L"
        boxed_text(canvas, f"{role} ID {track_id}  {float(person['confidence']):.0%}",
                   (x1b, max(16, y1b - 5)), color)

        points = np.asarray(person["keypoints_xy"], dtype=np.float32)
        confidence = np.asarray(person["keypoint_confidence"], dtype=np.float32)
        for left, right in SKELETON_EDGES:
            if confidence[left] > POSE_THRESHOLD and confidence[right] > POSE_THRESHOLD:
                cv2.line(canvas, tuple(np.rint(points[left]).astype(int)),
                         tuple(np.rint(points[right]).astype(int)), color, 1, cv2.LINE_AA)
        for index, point in enumerate(points):
            if confidence[index] > POSE_THRESHOLD:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, color, -1, cv2.LINE_AA)

        if not speaker:
            facing = orientation_vector(person)
            if facing is not None:
                origin, direction = facing
                endpoint = origin + direction * 42.0
                cv2.arrowedLine(canvas, tuple(np.rint(origin).astype(int)),
                                tuple(np.rint(endpoint).astype(int)), (220, 80, 220),
                                2, cv2.LINE_AA, tipLength=0.25)
    return canvas


def summarize_matrix(path: Path) -> dict[str, float]:
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (8, 40) or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected behavioral matrix {path}: {matrix.shape}")
    means = matrix[:, :32].mean(axis=0)
    by_name = dict(zip(INTERACTION_COLUMNS, means, strict=True))
    return {name: float(by_name[name]) for name in SELECTED_FEATURES}


def source_paths(record: dict, frames_root: Path, tracks_root: Path) -> tuple[Path, Path, Path]:
    stem = Path(record["key"]).stem
    split = record["split"]
    return (
        frames_root / split / "high" / f"{stem}.npz",
        tracks_root / split / "high" / f"{stem}.tracks.json",
        Path(record["source_matrix"]).expanduser().resolve(),
    )


def unavailable_reason(paths: tuple[Path, ...]) -> str | None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return "missing: " + "; ".join(missing)
    dataless = [str(path) for path in paths
                if getattr(path.stat(), "st_flags", 0) & UF_DATALESS]
    if dataless:
        return "cloud placeholder not downloaded: " + "; ".join(dataless)
    return None


def create_audit(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    episodes = load_high_episodes(args.protocol)
    records = [record for record in manifest["records"] if int(record["label"]) == 2]
    if len(records) != 85:
        raise ValueError(f"Expected 85 selected High clips, found {len(records)}")

    prepared = []
    for record in records:
        number = view_number(record["key"])
        prepared.append({**record, "view_number": number,
                         "episode_id": episode_for(number, episodes)})
    if args.max_clips:
        by_group = {group: [] for group in ("H01", "H02", "H03", "H04")}
        for record in prepared:
            by_group[record["episode_id"]].append(record)
        prepared = []
        for group, group_records in by_group.items():
            prepared.extend(sorted(group_records, key=lambda row: row["view_number"])[:args.max_clips])

    available, unavailable = [], []
    for record in prepared:
        paths = source_paths(record, args.frames_root, args.tracks_root)
        reason = unavailable_reason(paths)
        if reason:
            unavailable.append({"key": record["key"], "episode_id": record["episode_id"],
                                "reason": reason})
        else:
            available.append(record)
    if args.require_all and unavailable:
        raise RuntimeError(
            f"{len(unavailable)} selected High clips are not locally available; "
            "download the cloud placeholders or omit --require-all"
        )
    prepared = available
    if not prepared:
        raise RuntimeError("No High clips have locally available frame, track, and matrix files")
    random.Random(args.blind_seed).shuffle(prepared)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output_dir}")
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True)

    clips = []
    for index, record in enumerate(prepared, start=1):
        code = f"C{index:03d}"
        split = record["split"]
        frames_path, tracks_path, matrix_path = source_paths(
            record, args.frames_root, args.tracks_root
        )

        with np.load(frames_path, allow_pickle=False) as archive:
            frames = archive["frames"]
        tracking = json.loads(tracks_path.read_text(encoding="utf-8"))
        samples = tracking.get("samples", [])
        if frames.shape != (8, 640, 640, 3) or len(samples) != 8:
            raise ValueError(f"Expected eight aligned frames/tracking samples for {record['key']}")
        if [int(sample["sample_index"]) for sample in samples] != list(range(8)):
            raise ValueError(f"Unexpected sample order for {record['key']}")

        known_speakers = {int(value) for value in tracking.get("known_speaker_track_ids", [])}
        frame_images = []
        for frame_index, (frame, sample) in enumerate(zip(frames, samples, strict=True)):
            raw_relative = Path("images") / f"{code}_f{frame_index}_raw.jpg"
            overlay_relative = Path("images") / f"{code}_f{frame_index}_overlay.jpg"
            raw_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay = render_overlay(frame, sample, known_speakers)
            parameters = [cv2.IMWRITE_JPEG_QUALITY, 88]
            if not cv2.imwrite(str(args.output_dir / raw_relative), raw_bgr, parameters):
                raise OSError(f"Could not write {raw_relative}")
            if not cv2.imwrite(str(args.output_dir / overlay_relative), overlay, parameters):
                raise OSError(f"Could not write {overlay_relative}")
            frame_images.append({"frame_index": frame_index,
                                 "raw": raw_relative.as_posix(),
                                 "overlay": overlay_relative.as_posix(),
                                 "target_source": sample["target_source"],
                                 "people": len(sample["people"])})

        clips.append({
            "audit_code": code,
            "key": record["key"],
            "episode_id": record["episode_id"],
            "comparison_group": "H04" if record["episode_id"] == "H04" else "Other High",
            "source_split": split,
            "matrix_name": record["matrix_name"],
            "frames": frame_images,
            "feature_means": summarize_matrix(matrix_path),
            "tracking_summary": {
                "known_speaker_tracks": sorted(known_speakers),
                "unique_tracks": len(tracking.get("tracks", [])),
                "target_speaker_frames": sum(sample["target_source"] == "speaker" for sample in samples),
            },
        })
        print(f"Rendered {index}/{len(prepared)}: {code}", flush=True)

    payload = {
        "metadata": {
            "audit_id": args.output_dir.name,
            "purpose": "blinded H04 versus other-High extraction and label audit",
            "status": "unannotated independent visual audit",
            "blind_seed": args.blind_seed,
            "clips": len(clips),
            "selected_high_clips": len(records),
            "unavailable_clips": len(unavailable),
            "unavailable": unavailable,
            "frames_per_clip": 8,
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_manifest_sha256": sha256(args.dataset_manifest),
            "protocol": str(args.protocol),
            "protocol_sha256": sha256(args.protocol),
            "rendering": "saved preprocessed frames plus saved YOLO-pose/Hungarian records; no inference rerun",
            "limitations": [
                "All audited clips already carry the dataset High label.",
                "Pose arrows are coarse body/head-orientation proxies, not eye gaze or attention probabilities.",
                "Visible activity does not establish whether a student is learning or internally attentive.",
            ],
        },
        "clips": clips,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    template = args.template.read_text(encoding="utf-8")
    embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    (args.output_dir / "index.html").write_text(
        template.replace("__AUDIT_DATA__", embedded), encoding="utf-8"
    )
    guide = args.guide.read_text(encoding="utf-8")
    guide += (
        "\n## This generated package\n\n"
        f"Included **{len(clips)} of {len(records)}** selected High clips. "
        f"**{len(unavailable)}** clips were skipped because at least one required "
        "frame, tracking, or matrix file was a cloud placeholder not downloaded "
        "locally (or was missing). The exact list is stored in `audit.json`. "
        "Download those files and rerun into a new output directory for the full "
        "85-clip audit.\n"
    )
    (args.output_dir / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    print(f"Created visual audit at {args.output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_only/dataset/selection_manifest.json")
    parser.add_argument("--protocol", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_temporal_group_cv/temporal_group_protocol.json")
    parser.add_argument("--frames-root", type=Path,
                        default=PROJECT_ROOT / "preprocessed_data/yolov5_640x640")
    parser.add_argument("--tracks-root", type=Path,
                        default=PROJECT_ROOT / "preprocessed_features/interaction_features")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "audit_outputs/high_group_visual_audit_20260927")
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).with_name("high_group_visual_audit.html"))
    parser.add_argument("--guide", type=Path,
                        default=Path(__file__).with_name("high_group_visual_audit.md"))
    parser.add_argument("--blind-seed", type=int, default=20260927)
    parser.add_argument("--max-clips", type=int,
                        help="Pilot only: maximum clips per High group; default audits all 85")
    parser.add_argument("--require-all", action="store_true",
                        help="Fail instead of skipping missing or cloud-placeholder source files")
    args = parser.parse_args()
    for name in ("dataset_manifest", "protocol", "frames_root", "tracks_root",
                 "output_dir", "template", "guide"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    return args


def main() -> None:
    create_audit(parse_args())


if __name__ == "__main__":
    main()

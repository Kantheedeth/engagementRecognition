"""Extract the branch's 32-dimensional YOLO interaction descriptor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO

from src.data.feature_schema import INTERACTION_FEATURE_SCHEMA, INTERACTION_COLUMNS
from src.data.person_tracking import file_sha256, matched_motion, track_sampled_frames


DEFAULT_INPUT_DIR = PROJECT_ROOT / "preprocessed_data" / "yolov5_640x640"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "preprocessed_features" / "interaction_features"

# VFOA "instructional zone" geometric boundary.
ZONE_X_MIN = 0.0
ZONE_X_MAX = 0.27
ZONE_Y_MIN = 0.0
ZONE_Y_MAX = 1.0
MAX_STUDENTS_TRACKED = 5
CALIBRATION_PATH = SCRIPT_DIR / "listener_proxy_calibration_v1.json"
CALIBRATION = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
THRESHOLDS = CALIBRATION["thresholds"]


def normalize_model_reference(value: str) -> str:
    """Normalize local model paths before comparing extraction provenance."""
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    script_relative = SCRIPT_DIR / candidate
    if not candidate.is_absolute() and script_relative.is_file():
        return str(script_relative.resolve())
    return value


def interaction_provenance(model: str, confidence: float, tracker_settings=None) -> dict:
    """Return the generation settings that must match for reusable features."""
    return {
        "format_version": 3,
        "feature_schema": INTERACTION_FEATURE_SCHEMA,
        "shape_per_video": [8, 32],
        "model": normalize_model_reference(model),
        "model_sha256": file_sha256(normalize_model_reference(model)),
        "extractor_sha256": file_sha256(__file__),
        "tracking_code_sha256": file_sha256(Path(__file__).with_name("person_tracking.py")),
        "behavior_calibration": CALIBRATION,
        "behavior_calibration_sha256": file_sha256(CALIBRATION_PATH),
        "columns": list(INTERACTION_COLUMNS),
        "tracker": {"algorithm": "hungarian-box-pose", "ultralytics_version": ultralytics.__version__,
                    "sampling": "eight_preprocessed_frames", "id_reset": "every_clip",
                    "settings": tracker_settings or {}, "unmatched_motion": "excluded"},
        "confidence": confidence,
        "input_color": "RGB",
        "ultralytics_numpy_color": "BGR",
        "instruction_zone": {
            "x_min": ZONE_X_MIN,
            "x_max": ZONE_X_MAX,
            "y_min": ZONE_Y_MIN,
            "y_max": ZONE_Y_MAX,
        },
        "max_person_states": None,
    }


def validate_reusable_provenance(existing: dict | None, requested: dict) -> None:
    """Reject existing arrays whose recorded generation settings are unknown."""
    if existing is None:
        raise RuntimeError(
            "Existing interaction features have no provenance manifest. Re-run with "
            "--overwrite instead of silently reusing unverifiable arrays."
        )

    comparable_existing = dict(existing)
    comparable_existing.pop("summary", None)
    comparable_existing["model"] = normalize_model_reference(
        str(comparable_existing.get("model", ""))
    )
    if comparable_existing != requested:
        raise RuntimeError(
            "Existing interaction features were generated with different or "
            "incomplete settings. Re-run with --overwrite to replace the full set; "
            "mixed-provenance features are not allowed."
        )


DEFAULT_TARGET = np.array([0.135, 0.50], dtype=np.float32)


def extract_video_interaction_descriptors(results, track_ids=None, listener_exclusions=None) -> np.ndarray:
    """Extract 32-dim collective posture, gaze, face, and motion features across all frames.

    Zero-coordinate schema: Evaluates ALL students in the lecture room without leaking
    spatial bounding-box coordinates (cx, cy, width, height).
    """
    descriptors = []
    prev_students = None
    prev_speaker_pos = None
    prev_speaker_id = None
    prev_tracks = {}
    if track_ids is not None and len(track_ids) != len(results):
        raise ValueError("Track IDs must align with feature frames")

    for t, result in enumerate(results):
        ids = track_ids[t] if track_ids is not None else None
        excluded = listener_exclusions[t] if listener_exclusions is not None else set()
        if ids is not None and len(ids) != len(result.boxes):
            raise ValueError("Track IDs must align with detection/keypoint rows")
        if len(result.boxes) == 0:
            descriptors.append(np.zeros(32, dtype=np.float32))
            prev_students = None
            prev_tracks = {}
            prev_speaker_pos = None
            prev_speaker_id = None
            continue

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        kpts = result.keypoints.xy.detach().cpu().numpy() if result.keypoints is not None else None
        kpts_conf = (
            result.keypoints.conf.detach().cpu().numpy()
            if result.keypoints is not None and result.keypoints.conf is not None
            else None
        )

        # 1. Identify teacher / speaker in instruction zone (x <= 0.27)
        speaker_pos = None
        speaker_id = None
        for i, box in enumerate(xyxy):
            if int(classes[i]) != 0:
                continue
            cx = ((box[0] + box[2]) / 2.0) / 640.0
            cy = ((box[1] + box[3]) / 2.0) / 640.0
            if ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX:
                speaker_pos = np.array([cx, cy], dtype=np.float32)
                speaker_id = ids[i] if ids is not None else None
                break

        target_pos = speaker_pos if speaker_pos is not None else DEFAULT_TARGET
        speaker_active = 1.0 if speaker_pos is not None else 0.0
        speaker_motion = 0.0
        same_speaker = ids is None or (speaker_id is not None and speaker_id == prev_speaker_id)
        if prev_speaker_pos is not None and speaker_pos is not None and same_speaker:
            speaker_motion = float(np.linalg.norm(speaker_pos - prev_speaker_pos))
        prev_speaker_pos = speaker_pos
        prev_speaker_id = speaker_id

        # 2. Extract per-student behavioral features for ALL students outside the zone
        student_gazes = []
        student_head_pitch = []    # calibrated nose-to-shoulder elevation; NaN = unobserved
        student_head_yaw = []      # 0 = frontal, large = turned sideways
        student_posture = []       # head elevation ratio: high = upright, low = slumped
        student_face_vis = []      # face keypoint confidence
        student_both_eyes = []     # 1 if both eyes visible, else 0
        student_single_eye = []    # 1 if only 1 eye visible, else 0
        student_shoulder_tilt = [] # shoulder asymmetry
        student_positions = []
        current_tracks = {}
        is_slumped_list = []
        facing_vectors = []

        for i, box in enumerate(xyxy):
            if int(classes[i]) != 0:
                continue
            cx = ((box[0] + box[2]) / 2.0) / 640.0
            cy = ((box[1] + box[3]) / 2.0) / 640.0

            # Exclude teacher in the instruction zone
            if cx <= ZONE_X_MAX or i in excluded:
                continue

            h = max((box[3] - box[1]) / 640.0, 1e-4)

            # Target direction to speaker
            v_target = target_pos - np.array([cx, cy], dtype=np.float32)
            dist_t = float(np.linalg.norm(v_target))
            u_target = v_target / dist_t if dist_t > 1e-4 else np.zeros(2, dtype=np.float32)

            if kpts is not None and i < len(kpts):
                p_kpts = kpts[i] / 640.0
                p_conf = kpts_conf[i] if kpts_conf is not None else np.ones(17, dtype=np.float32)

                nose = p_kpts[0]
                l_eye, r_eye = p_kpts[1], p_kpts[2]
                l_sh, r_sh = p_kpts[5], p_kpts[6]

                conf_nose = float(p_conf[0])
                conf_l_eye = float(p_conf[1])
                conf_r_eye = float(p_conf[2])
                conf_l_sh = float(p_conf[5])
                conf_r_sh = float(p_conf[6])

                face_vis = (conf_nose + conf_l_eye + conf_r_eye) / 3.0
                student_face_vis.append(face_vis)

                both_eyes = 1.0 if (conf_l_eye > 0.3 and conf_r_eye > 0.3) else 0.0
                student_both_eyes.append(both_eyes)

                single_eye = 1.0 if (both_eyes == 0.0 and (conf_l_eye > 0.3 or conf_r_eye > 0.3)) else 0.0
                student_single_eye.append(single_eye)

                # Posture: Head elevation relative to shoulders
                has_shoulders = (conf_l_sh > 0.3 and conf_r_sh > 0.3)
                sh_y = (l_sh[1] + r_sh[1]) / 2.0 if has_shoulders else cy
                elevation_observed = conf_nose > 0.3 and has_shoulders
                elevation = (sh_y - nose[1]) / h if elevation_observed else np.nan
                student_posture.append(float(elevation))

                # Conservative low-head proxy. It is not labelled as sleep: the
                # audit showed that an invisible face alone is ambiguous.
                is_slumped = bool(elevation_observed and elevation <= THRESHOLDS["very_low_head_max"])
                is_slumped_list.append(1.0 if is_slumped else 0.0)

                # Shoulder tilt
                if has_shoulders:
                    tilt = abs(l_sh[1] - r_sh[1]) / max(abs(l_sh[0] - r_sh[0]), 1e-4)
                else:
                    tilt = 0.0
                student_shoulder_tilt.append(float(np.clip(tilt, 0.0, 1.0)))

                # The reviewed CSVs showed nose-to-eye vertical distance does
                # not separate up/down in this camera. Use observed normalized
                # nose-to-shoulder elevation, with an explicit unknown state.
                student_head_pitch.append(float(elevation))

                # Head Yaw: looking forward vs turned sideways
                if conf_nose > 0.3 and conf_l_eye > 0.3 and conf_r_eye > 0.3:
                    eye_mid_x = (l_eye[0] + r_eye[0]) / 2.0
                    yaw = abs(nose[0] - eye_mid_x) / max(abs(l_eye[0] - r_eye[0]), 1e-4)
                    student_head_yaw.append(float(yaw))
                else:
                    student_head_yaw.append(0.5 if single_eye else 0.0)

                # Gaze direction vector
                v_facing = None
                if not is_slumped:
                    if conf_nose > 0.3 and (conf_l_eye > 0.3 or conf_r_eye > 0.3):
                        eye_mid = (
                            (l_eye + r_eye) / 2.0
                            if (conf_l_eye > 0.3 and conf_r_eye > 0.3)
                            else (l_eye if conf_l_eye > 0.3 else r_eye)
                        )
                        v_head = nose - eye_mid
                        norm_h = float(np.linalg.norm(v_head))
                        if norm_h > 1e-4:
                            v_facing = v_head / norm_h
                    elif has_shoulders:
                        v_sh = r_sh - l_sh
                        v_torso = np.array([-v_sh[1], v_sh[0]], dtype=np.float32)
                        norm_t = float(np.linalg.norm(v_torso))
                        if norm_t > 1e-4:
                            v_facing = v_torso / norm_t

                if v_facing is not None and dist_t > 1e-4:
                    cos_sim = float(np.dot(v_facing, u_target))
                    gaze_att = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
                    facing_vectors.append(v_facing)
                else:
                    gaze_att = 0.5

                student_gazes.append(gaze_att)
                student_positions.append(np.array([cx, cy], dtype=np.float32))
                if ids is not None and ids[i] is not None:
                    if ids[i] in current_tracks:
                        raise ValueError("Duplicate listener track ID within a frame")
                    current_tracks[ids[i]] = student_positions[-1]

        N = len(student_gazes)
        if N == 0:
            descriptors.append(np.zeros(32, dtype=np.float32))
            prev_students = None
            prev_tracks = {}
            continue

        gazes = np.array(student_gazes, dtype=np.float32)
        pitches = np.array(student_head_pitch, dtype=np.float32)
        yaws = np.array(student_head_yaw, dtype=np.float32)
        postures = np.array(student_posture, dtype=np.float32)
        face_vis = np.array(student_face_vis, dtype=np.float32)
        tilts = np.array(student_shoulder_tilt, dtype=np.float32)
        slumped = np.array(is_slumped_list, dtype=np.float32)

        # Calculate Frame-to-Frame Motion
        head_motion = 0.0
        body_motion = 0.0
        still_count = 0
        active_count = 0
        if ids is None and prev_students is not None and len(prev_students) > 0:
            dists = []
            for curr_pos in student_positions:
                min_d = min(float(np.linalg.norm(curr_pos - p_pos)) for p_pos in prev_students)
                dists.append(min_d)
                if min_d < 0.01:
                    still_count += 1
                elif min_d > 0.04:
                    active_count += 1
            head_motion = float(np.mean(dists))
            body_motion = float(np.std(dists))

        prev_students = student_positions
        stillness_ratio = still_count / float(N)
        high_motion_ratio = active_count / float(N)
        if ids is not None:
            head_motion, body_motion, stillness_ratio, high_motion_ratio = matched_motion(prev_tracks, current_tracks)
            prev_tracks = current_tracks

        # Mutual Gaze Alignment (Are students looking in the same direction?)
        if len(facing_vectors) > 1:
            dot_prods = []
            for a in range(len(facing_vectors)):
                for b in range(a + 1, len(facing_vectors)):
                    dot_prods.append(float(np.dot(facing_vectors[a], facing_vectors[b])))
            mutual_gaze = float(np.clip((np.mean(dot_prods) + 1.0) / 2.0, 0.0, 1.0))
        else:
            mutual_gaze = 0.5

        valid_head = np.isfinite(pitches)
        valid_posture = np.isfinite(postures)
        observed_heads = pitches[valid_head]
        observed_postures = postures[valid_posture]
        # Development thresholds selected from the submitted first-audit CSVs
        # (Kan 1-15, Pan 19-20, Bright 22-36; 16-18/21 excluded).
        head_up_ratio = float(np.mean(observed_heads >= THRESHOLDS["raised_head_min"])) if len(observed_heads) else 0.0
        head_down_ratio = float(np.mean(observed_heads < THRESHOLDS["lowered_head_max"])) if len(observed_heads) else 0.0
        upright_ratio = float(np.mean(observed_postures >= THRESHOLDS["upright_min"])) if len(observed_postures) else 0.0
        forward_lean_ratio = float(np.mean((observed_postures >= THRESHOLDS["very_low_head_max"])
                                           & (observed_postures < THRESHOLDS["upright_min"]))) if len(observed_postures) else 0.0
        mean_posture = float(np.mean(observed_postures)) if len(observed_postures) else 0.0
        min_posture = float(np.min(observed_postures)) if len(observed_postures) else 0.0
        std_posture = float(np.std(observed_postures)) if len(observed_postures) else 0.0
        head_dispersion = float(np.std(observed_heads)) if len(observed_heads) else 0.0

        # Assemble the 32 Features (Zero Spatial Coordinates)
        features = [
            # 1. Gaze & Head Orientation (8)
            head_up_ratio,                           # 0: raised_head_ratio
            head_down_ratio,                         # 1: lowered_head_ratio
            float(np.mean(yaws > 0.3)),              # 2: head_turned_ratio
            float(np.mean(gazes)),                   # 3: mean_gaze_to_teacher
            float(np.mean(gazes >= THRESHOLDS["orientation_consensus_min"])),
            float(np.std(gazes)),                    # 5: std_gaze_angle
            float(np.min(gazes)),                    # 6: min_gaze_attention
            float(np.max(gazes)),                    # 7: max_gaze_attention

            # 2. Postural Alertness & Fatigue (8)
            upright_ratio,                           # 8: upright_ratio
            forward_lean_ratio,                      # 9: forward_lean_proxy_ratio
            float(np.mean(slumped)),                 # 10: very_low_head_proxy_ratio (not sleep)
            mean_posture,                            # 11: mean observed elevation
            min_posture,                             # 12: min observed elevation
            std_posture,                             # 13: observed elevation spread
            float(np.mean(tilts)),                   # 14: shoulder_tilt_mean
            float(np.mean(tilts < 0.08)),            # 15: posture_symmetry

            # 3. Face Visibility & Eye Alertness (8)
            float(min(N / 15.0, 1.0)),               # 16: student_count_norm
            float(np.mean(face_vis)),                # 17: mean_face_visibility
            float(np.mean(face_vis < 0.25)),         # 18: hidden_face_ratio
            float(np.mean(student_both_eyes)),       # 19: both_eyes_visible_ratio
            float(np.mean(student_single_eye)),      # 20: single_eye_ratio
            float(np.median(face_vis)),              # 21: median_face_vis
            float(np.percentile(face_vis, 75) - np.percentile(face_vis, 25)), # 22: face_vis_iqr
            float(valid_posture.mean()),             # 23: posture_observation_ratio

            # 4. Motion Dynamics & Class Coordination (8)
            float(head_motion * 10.0),               # 24: matched center displacement mean
            float(body_motion * 10.0),               # 25: matched center displacement std
            float(stillness_ratio),                  # 26: stillness_ratio
            float(high_motion_ratio),                # 27: high_motion_ratio
            float(speaker_active),                   # 28: speaker_active
            float(speaker_motion * 10.0),            # 29: speaker_motion
            float(mutual_gaze),                      # 30: mutual_gaze_alignment
            head_dispersion,                         # 31: head_elevation_dispersion
        ]

        descriptors.append(np.array(features, dtype=np.float32))

    matrix = np.stack(descriptors).astype(np.float32)
    if matrix.shape != (len(results), 32) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid collective interaction matrix: {matrix.shape}")
    return matrix


def extract_frame_interaction_descriptor(boxes, keypoints=None) -> np.ndarray:
    """Backwards-compatible single-frame wrapper."""
    class MockResult:
        def __init__(self, b, k):
            self.boxes = b
            self.keypoints = k
    return extract_video_interaction_descriptors([MockResult(boxes, keypoints)])[0]


def select_device(requested: str) -> str:
    if requested != "auto":
        return "0" if requested == "cuda" else requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_inputs(input_dir: Path) -> list[tuple[str, str, Path]]:
    tasks = []
    for split in ("train", "val", "test"):
        for category in ("low", "mid", "high"):
            directory = input_dir / split / category
            if directory.is_dir():
                tasks.extend(
                    (split, category, path) for path in sorted(directory.glob("*.npz"))
                )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index")
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--video_names", nargs="+", help="Diagnostic subset, e.g. view12 view2228")
    parser.add_argument("--save_track_video", action="store_true", help="Save eight-frame ID-overlay MP4 previews")
    parser.add_argument("--track_max_center_distance", type=float, default=.22)
    parser.add_argument("--track_max_gap", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    tasks = collect_inputs(input_dir)
    if args.video_names:
        names = {Path(name).stem for name in args.video_names}
        tasks = [task for task in tasks if task[2].stem in names]
        absent = names - {task[2].stem for task in tasks}
        if absent:
            raise FileNotFoundError(f"Requested clips not present: {sorted(absent)}")
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError("--max_videos must be positive")
        tasks = tasks[: args.max_videos]
    if not tasks:
        raise FileNotFoundError(f"No preprocessed .npz files found under {input_dir}")
    subset = args.max_videos is not None or bool(args.video_names)
    if not np.isfinite(args.track_max_center_distance) or args.track_max_center_distance <= 0:
        raise ValueError("--track_max_center_distance must be finite and positive")
    if args.track_max_gap < 0:
        raise ValueError("--track_max_gap cannot be negative")
    settings = {"max_center_distance": args.track_max_center_distance,
                "max_gap": args.track_max_gap,
                "center_weight": .55, "iou_weight": .30, "pose_weight": .15}

    device = select_device(args.device)
    print("=" * 72)
    print("Extracting 32-Dimensional Interaction Features with clip-local person tracking")
    print(f"YOLO model: {args.model} | device: {device} | confidence: {args.confidence}")
    print(f"Instruction zone: x=[{ZONE_X_MIN}, {ZONE_X_MAX}], y=[{ZONE_Y_MIN}, {ZONE_Y_MAX}]")
    print(f"Found {len(tasks)} videos")
    print("One batched pose pass over 8 stored frames; Hungarian IDs reset per clip.")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extraction_manifest.json"
    config_path = output_dir / "extraction_config.json"
    requested_provenance = interaction_provenance(args.model, args.confidence, settings)
    requested_provenance.update(input_dir=str(input_dir), device=device)
    existing_manifest = None
    previous_path = manifest_path if manifest_path.is_file() else config_path
    if previous_path.is_file():
        with previous_path.open(encoding="utf-8") as file:
            existing_manifest = json.load(file)
    if subset and manifest_path.exists():
        raise ValueError("Use a separate --output_dir for diagnostic subsets; a complete manifest exists here.")
    existing_arrays = any(output_dir.glob("*/*/*.npy"))
    if existing_arrays and (not args.overwrite or subset):
        validate_reusable_provenance(existing_manifest, requested_provenance)
    model = YOLO(args.model)
    if model.task != "pose":
        raise ValueError("Person tracking requires a YOLO pose checkpoint")
    # A partially replaced dataset must not keep a valid buildable manifest.
    manifest_path.unlink(missing_ok=True)
    config_path.write_text(json.dumps(requested_provenance, indent=2), encoding="utf-8")
    processed = 0
    skipped = 0
    failures = []

    for split, category, source_path in tqdm(tasks, desc="Extracting interaction"):
        destination_dir = output_dir / split / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{source_path.stem}.npy"
        track_path = destination_path.with_suffix(".tracks.json")
        preview_path = destination_path.with_suffix(".tracks.mp4") if args.save_track_video else None
        if destination_path.is_file() and not args.overwrite:
            existing = np.load(destination_path, allow_pickle=False)
            if existing.shape == (8, 32) and np.isfinite(existing).all() and track_path.is_file():
                saved = json.loads(track_path.read_text())
                if (saved.get("provenance") == requested_provenance
                    and saved.get("input_sha256") == file_sha256(source_path)
                    and saved.get("feature_sha256") == file_sha256(destination_path)
                    and (preview_path is None or preview_path.is_file())):
                    skipped += 1
                    continue
            print(f"Replacing incompatible existing feature: {destination_path}")

        try:
            with np.load(source_path, allow_pickle=False) as archive:
                frames_rgb = archive["frames"]
            if frames_rgb.shape != (8, 640, 640, 3) or frames_rgb.dtype != np.uint8:
                raise ValueError(
                    f"frames are {frames_rgb.shape}/{frames_rgb.dtype}; expected "
                    "(8, 640, 640, 3)/uint8"
                )
            results, ids, exclusions, details = track_sampled_frames(
                model, frames_rgb, device, args.confidence, settings,
                (ZONE_X_MIN, ZONE_X_MAX, ZONE_Y_MIN, ZONE_Y_MAX), DEFAULT_TARGET, preview_path)
            descriptors = extract_video_interaction_descriptors(results, ids, exclusions)
            if descriptors.shape != (8, 32) or not np.isfinite(descriptors).all():
                raise ValueError(f"Invalid output matrix {descriptors.shape}")
            np.save(destination_path, descriptors)
            details.update(provenance=requested_provenance, input_sha256=file_sha256(source_path),
                           feature_sha256=file_sha256(destination_path), feature_columns=list(INTERACTION_COLUMNS))
            previous_listener_ids = set()
            for sample, features in zip(details["samples"], descriptors):
                current_ids = {p["track_id"] for p in sample["people"]
                               if p["role"] == "listener" and p["track_id"] is not None}
                sample["matched_listener_count"] = len(current_ids & previous_listener_ids)
                sample["unmatched_listener_count"] = sum(p["role"] == "listener" for p in sample["people"]) - sample["matched_listener_count"]
                sample["features"] = features.tolist()
                previous_listener_ids = current_ids
            track_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
            processed += 1
        except Exception as exc:
            failures.append(f"{source_path}: {type(exc).__name__}: {exc}")

    summary = {
        "input_videos": len(tasks),
        "processed_videos": processed,
        "skipped_existing_videos": skipped,
        "failed_videos": len(failures),
    }
    print(json.dumps(summary, indent=2))
    if failures:
        manifest_path.unlink(missing_ok=True)
        for failure in failures[:20]:
            print(f"  - {failure}")
        raise RuntimeError(
            f"Interaction extraction failed for {len(failures)} videos; no "
            "descriptors were silently fabricated."
        )

    if not subset:
        manifest = {**requested_provenance, "summary": summary}
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        print(f"Saved interaction provenance manifest: {manifest_path}")
    else:
        print("Diagnostic subset run: full-dataset manifest was not replaced.")


if __name__ == "__main__":
    main()

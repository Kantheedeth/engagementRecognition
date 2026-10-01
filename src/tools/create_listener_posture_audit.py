"""Create a local, manually annotated audit of the CURRENT interaction rules.

Uses the same 640x640 RGB inputs and extractor as production; never modifies
features, checkpoints, source data, or human labels. See listener_posture_audit.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import extract_interaction_features as interaction

# Sampling strata ONLY: user-supplied ranges, not verified recording identities.
RANGES = [(12, 203), (543, 1059), (1172, 1230), (2105, 2130),
          (2227, 2303), (2348, 2644)]
MANUAL_FIELDS = ["annotator", "manual_role", "manual_head", "manual_posture",
                 "manual_facing", "manual_activity", "notes"]
PRED_COLUMNS = {0: "head_up_flag", 1: "head_down_flag", 2: "head_turned_flag",
                3: "orientation_score", 8: "upright_flag", 9: "slouching_flag",
                10: "slumped_flag", 11: "posture_elevation_used",
                14: "shoulder_tilt_used", 17: "face_visibility"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_number(path):
    match = re.fullmatch(r"view(\d+)", Path(path).stem)
    if not match:
        raise ValueError("Expected a view<number> filename: {}".format(path))
    return int(match.group(1))


def select_inputs(input_dir, csv_dir, per_range):
    """Exclude train IDs appearing in held-out CSVs, even with zero padding."""
    held_out = set()
    for split in ("val", "test"):
        for line in (csv_dir / (split + ".csv")).read_text().splitlines():
            if line.strip():
                held_out.add(clip_number(line.rsplit(maxsplit=1)[0]))
    available = {}
    excluded = set()
    ambiguous = set()
    for line in (csv_dir / "train.csv").read_text().splitlines():
        if not line.strip():
            continue
        name, label = line.rsplit(maxsplit=1)
        number = clip_number(name)
        if number in held_out:
            excluded.add(number)
            continue
        if number in ambiguous:
            continue
        category = Path(name).parent.name.lower()
        if {"low": "0", "mid": "1", "high": "2"}.get(category) != label:
            raise ValueError("CSV label/path mismatch: " + line)
        source = input_dir / "train" / category / (Path(name).stem + ".npz")
        if not source.is_file():
            raise FileNotFoundError(source)
        if number in available and available[number] != source:
            ambiguous.add(number)
            del available[number]
            continue
        available[number] = source
    selected = []
    strata = []
    for low, high in RANGES:
        candidates = sorted(n for n in available if low <= n <= high)
        count = min(per_range, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, count, dtype=int) if count else []
        chosen = [candidates[int(i)] for i in indices]
        selected.extend(available[n] for n in chosen)
        strata.append({"range": [low, high], "available": len(candidates), "selected_ids": chosen})
    if not selected:
        raise ValueError("No eligible training clips in the sampling ranges")
    return selected, {"strata": strata, "excluded_train_ids_in_val_test": sorted(excluded),
                      "excluded_ambiguous_train_ids": sorted(ambiguous),
                      "range_interpretation": "provisional numeric sampling strata, not verified sessions"}


def inspect_result(result):
    """Obtain listener predictions directly from the production extractor.

    Single-listener sub-results retain the actual speaker so posture/orientation
    columns exactly match production; temporal/motion columns are NOT audited.
    """
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    centers = (boxes[:, :2] + boxes[:, 2:]) / (2 * 640.0)
    speaker = next((i for i, (c, p) in enumerate(zip(classes, centers))
                    if int(c) == 0 and interaction.ZONE_X_MIN <= p[0] <= interaction.ZONE_X_MAX
                    and interaction.ZONE_Y_MIN <= p[1] <= interaction.ZONE_Y_MAX), None)
    target = centers[speaker] if speaker is not None else interaction.DEFAULT_TARGET
    rows = []
    for i, (c, center) in enumerate(zip(classes, centers)):
        if int(c) != 0 or center[0] <= interaction.ZONE_X_MAX:
            continue
        has_pose = result.keypoints is not None and i < len(result.keypoints.xy)
        row = {"listener_id": "P{}".format(i + 1), "detection_index": i,
               "box_xyxy": boxes[i].tolist(), "det_confidence": float(result.boxes.conf[i]),
               "pose_available": has_pose, "flags": [], "keypoints": [], "confidence": []}
        if has_pose:
            subset = [speaker, i] if speaker is not None else [i]
            values = interaction.extract_video_interaction_descriptors([result[subset]])[0]
            row.update({name: float(values[column]) for column, name in PRED_COLUMNS.items()})
            points = result.keypoints.xy[i].cpu().numpy() / 640.0
            confidence = (result.keypoints.conf[i].cpu().numpy()
                          if result.keypoints.conf is not None else np.ones(17))
            row["keypoints"] = (points * 640).tolist()
            row["confidence"] = confidence.tolist()
            shoulders = bool(confidence[5] > .3 and confidence[6] > .3)
            elevation_valid = bool(shoulders and confidence[0] > .3)
            pitch_valid = bool(confidence[0] > .3 and (confidence[1] > .3 or confidence[2] > .3))
            row.update(elevation_observed=elevation_valid, pitch_observed=pitch_valid)
            row["pitch_used"] = row["posture_elevation_used"] if elevation_valid else None
            if not elevation_valid:
                row["flags"].append("Head elevation unknown; excluded from head/posture ratios")
            if not elevation_valid:
                row["flags"].append("Missing face/shoulders is not interpreted as sleep")
            if shoulders and row["posture_elevation_used"] <= .06:
                row["flags"].append("Very low observed head elevation proxy; not a sleep label")
            if not shoulders:
                row["flags"].append("Shoulder tilt default zero; not measured symmetry")
        else:
            row.update({name: None for name in PRED_COLUMNS.values()})
            row["flags"].append("No pose: omitted from production listener aggregates")
        rows.append(row)
    # Confirm singleton diagnostics match the actual full-frame aggregates.
    actual = interaction.extract_video_interaction_descriptors([result])[0]
    included = [r for r in rows if r["pose_available"]]
    if included:
        for column, key in PRED_COLUMNS.items():
            if not np.isclose(np.mean([r[key] for r in included]), actual[column], atol=1e-6):
                raise AssertionError("Audit differs from extractor for " + key)
    return rows, target, speaker


def render(frame_bgr, result, rows, target, speaker, diagnostics):
    canvas = frame_bgr.copy()
    cv2.rectangle(canvas, (int(interaction.ZONE_X_MIN * 640), int(interaction.ZONE_Y_MIN * 640)),
                  (int(interaction.ZONE_X_MAX * 640), int(interaction.ZONE_Y_MAX * 640)), (220, 180, 0), 2)
    if diagnostics:
        cv2.drawMarker(canvas, tuple((target * 640).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    if speaker is not None:
        x1, y1, x2, y2 = result.boxes.xyxy[speaker].cpu().numpy().astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 180, 0), 2)
        cv2.putText(canvas, "Zone candidate", (max(0, x1), max(16, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 180, 0), 1)
    for row in rows:
        x1, y1, x2, y2 = np.asarray(row["box_xyxy"], dtype=int)
        color = (70, 220, 100)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = row["listener_id"]
        if diagnostics and row["pose_available"]:
            label += " slump={:.0f}".format(row["slumped_flag"])
            for idx in (0, 1, 2, 5, 6):
                if row["confidence"][idx] > .3:
                    cv2.circle(canvas, tuple(np.asarray(row["keypoints"][idx], dtype=int)), 3, (0, 210, 255), -1)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0] + 5
        cv2.rectangle(canvas, (max(0, x1), max(0, y1 - 20)), (min(639, x1 + text_width), max(20, y1)), (25, 25, 25), -1)
        cv2.putText(canvas, label, (max(0, x1 + 2), max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=interaction.DEFAULT_INPUT_DIR)
    parser.add_argument("--csv-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8n-pose.pt")
    parser.add_argument("--clips-per-range", type=int, default=2)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=[1, 4, 6])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    if args.clips_per_range < 1 or not args.frame_indices or any(i < 0 or i > 7 for i in args.frame_indices):
        parser.error("Positive clips-per-range and frame indices 0..7 required")
    if len(set(args.frame_indices)) != len(args.frame_indices) or not 0 < args.confidence <= 1:
        parser.error("Frame indices must be unique and confidence must be in (0,1]")
    if not args.model.is_file():
        raise FileNotFoundError("A local pose checkpoint is required: " + str(args.model))
    selected, sampling = select_inputs(args.input_dir, args.csv_dir, args.clips_per_range)
    output = args.output_dir or ROOT / "audit_outputs" / ("listener_posture_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    model = YOLO(str(args.model))
    frames = []
    sources = []
    for clip_index, path in enumerate(selected):
        with np.load(path, allow_pickle=False) as archive:
            rgb = archive["frames"]
        if rgb.shape != (8, 640, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError("Unexpected preprocessed frame format: " + str(path))
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
        for frame_index in args.frame_indices:
            key = "clip{:02d}_f{}".format(clip_index + 1, frame_index)
            bgr = cv2.cvtColor(rgb[frame_index], cv2.COLOR_RGB2BGR)
            result = model.predict(source=bgr, conf=args.confidence, device=args.device, verbose=False)[0]
            rows, target, speaker = inspect_result(result)
            for row in rows:
                row["record_id"] = key + "_" + row["listener_id"]
                x1, y1, x2, y2 = np.asarray(row["box_xyxy"], dtype=int)
                crop = bgr[max(0, y1 - 8):min(640, y2 + 8), max(0, x1 - 8):min(640, x2 + 8)]
                row["crop_image"] = "images/" + row["record_id"] + ".jpg"
                if not crop.size or not cv2.imwrite(str(output / row["crop_image"]), crop):
                    raise OSError("Failed to save listener crop")
            for mode in ("clean", "diagnostic"):
                rendered = render(bgr, result, rows, target, speaker, mode == "diagnostic")
                if not cv2.imwrite(str(output / "images" / (key + "_" + mode + ".jpg")), rendered):
                    raise OSError("Failed to save audit image")
            if not cv2.imwrite(str(output / "images" / (key + "_raw.jpg")), bgr):
                raise OSError("Failed to save raw frame")
            frames.append({"id": key, "source": str(path.relative_to(args.input_dir)),
                           "frame_index": frame_index, "target_source": "speaker" if speaker is not None else "zone_center",
                           "target_xy": target.tolist(), "speaker_detection_index": speaker,
                           "listeners": rows})
        print("Audited {}/{}: {}".format(clip_index + 1, len(selected), path.name), flush=True)
    metadata = {"format_version": 1, "audit_id": output.name, "status": "unannotated development audit",
                "sampling": sampling, "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
                "extractor_sha256": sha256(interaction.__file__), "sources": sources,
                "device": args.device, "confidence": args.confidence,
                "frame_indices_zero_based": args.frame_indices,
                "csv_sha256": {s: sha256(args.csv_dir / (s + ".csv")) for s in ("train", "val", "test")},
                "frame_count": len(frames), "listener_count": sum(len(f["listeners"]) for f in frames)}
    payload = {"metadata": metadata, "frames": frames}
    (output / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    template = Path(__file__).with_name("listener_posture_audit.html").read_text(encoding="utf-8")
    # Escaping '<' prevents an embedded path/string from closing the script tag.
    (output / "index.html").write_text(template.replace("__AUDIT_DATA__", json.dumps(payload).replace("<", "\\u003c")), encoding="utf-8")
    with (output / "annotations_blank.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "frame_id", "listener_id"] + MANUAL_FIELDS)
        writer.writeheader()
        for frame in frames:
            for row in frame["listeners"]:
                writer.writerow({"record_id": row["record_id"], "frame_id": frame["id"], "listener_id": row["listener_id"]})
    (output / "ANNOTATION_GUIDE.md").write_text(Path(__file__).with_name("listener_posture_audit.md").read_text(), encoding="utf-8")
    print("Created {} frames / {} listener observations at {}".format(metadata["frame_count"], metadata["listener_count"], output.resolve()))


if __name__ == "__main__":
    main()

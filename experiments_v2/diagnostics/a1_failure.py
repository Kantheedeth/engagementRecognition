"""Diagnose legacy A1 failures without changing or publishing feature outputs.

The controller reads the immutable partial A1 validation record, selects a
small representative set, and launches one isolated worker per sample.  A
worker calls the unchanged A1 module and observes its stage boundaries.  An
unhandled-exception hook records the complete traceback and traceback-frame
locals before delegating to Python's normal exception printer.

This module deliberately has no recovery path: it never writes ``.npy``
features and never catches an A1 exception in the worker.
"""

from __future__ import annotations

import argparse
import json
import linecache
import re
import sys
import traceback
from collections.abc import Mapping as MappingABC
from collections import Counter
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments_v2.adapters.base import run_logged
from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    new_id,
    read_json,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CATEGORIES = ("low", "mid", "high")
EXPECTED_LABELS = {"low": 0, "mid": 1, "high": 2}

_ACTIVE_CONTEXT: dict[str, Any] | None = None
_FAILURE_PATH: Path | None = None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def describe(value: Any, *, depth: int = 0) -> dict[str, Any]:
    """Return bounded, JSON-safe structural metadata without changing a value."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    result: dict[str, Any] = {"type": type_name}
    if isinstance(value, np.ndarray):
        return {
            **result,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "ndim": value.ndim,
            "size": value.size,
            "finite": bool(np.isfinite(value).all()) if value.size else True,
        }
    if isinstance(value, torch.Tensor):
        finite = None
        try:
            finite = bool(torch.isfinite(value).all().item()) if value.numel() else True
        except (RuntimeError, TypeError):
            pass
        return {
            **result,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "ndim": value.ndim,
            "size": value.numel(),
            "finite": finite,
        }
    if isinstance(value, MappingABC):
        try:
            keys = list(value.keys())
        except Exception as exc:
            result["inspection_error"] = f"{type(exc).__name__}: {exc}"
            return result
        result["length"] = len(keys)
        result["keys"] = [str(key) for key in keys[:30]]
        if depth < 1:
            items = {}
            for key in keys[:15]:
                try:
                    items[str(key)] = describe(value[key], depth=depth + 1)
                except Exception as exc:
                    items[str(key)] = {
                        "inspection_error": f"{type(exc).__name__}: {exc}"
                    }
            result["items"] = items
        return result
    if isinstance(value, (list, tuple)):
        result["length"] = len(value)
        if depth < 1:
            result["items"] = [describe(item, depth=depth + 1) for item in value[:15]]
        return result

    # Ultralytics Boxes and similar containers expose useful structural fields.
    for attribute in ("data", "conf", "cls", "xyxy", "xywh"):
        try:
            member = getattr(value, attribute)
        except Exception:
            continue
        if isinstance(member, (np.ndarray, torch.Tensor)):
            result[attribute] = describe(member, depth=depth + 1)
    try:
        result["length"] = len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass
    if not any(key in result for key in ("data", "conf", "cls", "xyxy", "xywh", "length")):
        rendered = str(_json_scalar(value))
        result["value"] = rendered[:500]
    return result


def _traceback_frames(tb: TracebackType | None) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while tb is not None:
        frame = tb.tb_frame
        line_number = tb.tb_lineno
        frames.append(
            {
                "file": frame.f_code.co_filename,
                "function": frame.f_code.co_name,
                "line": line_number,
                "expression": linecache.getline(frame.f_code.co_filename, line_number).strip(),
                "locals": {
                    name: describe(value)
                    for name, value in sorted(frame.f_locals.items())
                    if not name.startswith("__")
                },
            }
        )
        tb = tb.tb_next
    return frames


def _index_attempt(message: str) -> dict[str, Any] | None:
    match = re.search(
        r"index\s+(-?\d+)\s+is out of bounds for axis\s+(\d+)\s+with size\s+(\d+)",
        message,
    )
    if match is None:
        return None
    return {
        "attempted_index": int(match.group(1)),
        "axis": int(match.group(2)),
        "actual_axis_size": int(match.group(3)),
    }


def _diagnostic_excepthook(
    exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
) -> None:
    """Persist evidence, then leave the exception unhandled as normal."""
    if _FAILURE_PATH is not None:
        frames = _traceback_frames(tb)
        record = {
            "status": "failed",
            "failed_at": utc_now(),
            "exception_type": f"{exc_type.__module__}.{exc_type.__qualname__}",
            "exception": str(exc),
            "index_operation": _index_attempt(str(exc)),
            "failure_location": frames[-1] if frames else None,
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            "traceback_frames": frames,
            "diagnostic_context": _ACTIVE_CONTEXT,
        }
        try:
            write_json_exclusive(_FAILURE_PATH, record)
        except Exception as hook_error:
            print(f"Diagnostic hook could not persist failure evidence: {hook_error}", file=sys.stderr)
    sys.__excepthook__(exc_type, exc, tb)


def failure_distribution(missing: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split_category = Counter(
        (str(item["split"]), str(item["category"])) for item in missing
    )
    return {
        "total": len(missing),
        "by_split": dict(sorted(Counter(str(item["split"]) for item in missing).items())),
        "by_category": dict(
            sorted(Counter(str(item["category"]) for item in missing).items())
        ),
        "by_split_category": {
            split: {
                category: by_split_category.get((split, category), 0)
                for category in EXPECTED_CATEGORIES
            }
            for split in ("train", "val", "test")
        },
    }


def select_failures(missing: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select one sample per category, preferring a preserved exact error."""
    selected: list[dict[str, Any]] = []
    for category in EXPECTED_CATEGORIES:
        candidates = [dict(item) for item in missing if item.get("category") == category]
        candidates.sort(
            key=lambda item: (
                not bool(item.get("reported_error")),
                {"train": 0, "val": 1, "test": 2}.get(str(item.get("split")), 3),
                int(item.get("csv_line", 0)),
                str(item.get("key")),
            )
        )
        if candidates:
            selected.append(candidates[0])
    return selected


def _csv_samples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    split_files = config["dataset"]["split_files"]
    preprocessed_root = Path(config["dataset"]["preprocessed_input_dir"])
    for split in ("train", "val", "test"):
        csv_path = Path(split_files[split])
        for line_number, raw_line in enumerate(
            csv_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = raw_line.strip()
            if not stripped:
                continue
            video_path, label_text = stripped.rsplit(maxsplit=1)
            source = Path(video_path)
            csv_category = source.parent.name
            category = csv_category.lower()
            stem = source.stem
            samples.append(
                {
                    "split": split,
                    "category": category,
                    "csv_category": csv_category,
                    "csv_line": line_number,
                    "csv_video_path": video_path,
                    "label": int(label_text),
                    "key": f"{split}/{category}/{stem}",
                    "preprocessed_path": str(
                        preprocessed_root / split / category / f"{stem}.npz"
                    ),
                }
            )
    return samples


def select_control(
    config: Mapping[str, Any], partial_directory: Path, missing_keys: set[str]
) -> dict[str, Any]:
    """Choose a validated successful Low sample from the preserved artifact."""
    candidates = sorted(
        _csv_samples(config),
        key=lambda item: (
            item["category"] != "low",
            item["split"] != "train",
            item["csv_line"],
        ),
    )
    for sample in candidates:
        if sample["key"] in missing_keys:
            continue
        feature_path = partial_directory / "data" / f"{sample['key']}.npy"
        if feature_path.is_file():
            sample["preserved_feature_path"] = str(feature_path.resolve())
            sample["control"] = True
            return sample
    raise FileNotFoundError("No preserved successful A1 output is available as a control")


def _shape_of_sequence(values: Sequence[Any]) -> list[int] | None:
    if not values:
        return [0]
    shapes = [tuple(getattr(value, "shape", ())) for value in values]
    if len(set(shapes)) == 1:
        return [len(values), *shapes[0]]
    return None


class StageObserver:
    """Attach behavior-neutral observation hooks around unchanged A1 components."""

    def __init__(self, module: torch.nn.Module, context: dict[str, Any]) -> None:
        self.module = module
        self.context = context
        self.current: dict[str, Any] | None = None
        self._handles: list[Any] = []
        self._patched_attributes: list[tuple[Any, str, Any]] = []
        self._install()

    def _install(self) -> None:
        detector = self.module.detector
        original_detect = detector.detect

        def observed_detect(frame_rgb_np: np.ndarray):
            assert self.current is not None
            stage = {
                "status": "running",
                "frame_input": describe(frame_rgb_np),
            }
            self.current["retinaface"] = stage
            self.current["active_stage"] = "retinaface"
            detections = original_detect(frame_rgb_np)
            bboxes = [det.bbox for det in detections]
            landmarks = [det.landmarks for det in detections if det.landmarks is not None]
            scores = np.asarray([det.score for det in detections], dtype=np.float64)
            stage.update(
                {
                    "status": "complete",
                    "number_of_faces": len(detections),
                    "bbox_batch_shape": _shape_of_sequence(bboxes),
                    "bbox_shapes": [list(value.shape) for value in bboxes],
                    "bboxes_xyxy": [value.tolist() for value in bboxes],
                    "keypoint_batch_shape": _shape_of_sequence(landmarks),
                    "keypoint_shapes": [list(value.shape) for value in landmarks],
                    "missing_keypoints": len(detections) - len(landmarks),
                    "detection_score_shape": list(scores.shape),
                    "detection_scores": scores.tolist(),
                }
            )
            return detections

        self._patched_attributes.append((detector, "detect", original_detect))
        detector.detect = observed_detect

        tracker = self.module.tracker
        if tracker is not None:
            original_tracker_update = tracker.update
            original_byte_update = tracker._tracker.update

            def observed_byte_update(boxes, *args, **kwargs):
                assert self.current is not None
                tracker_stage = self.current["bytetrack"]
                tracker_stage["boxes_object"] = describe(boxes)
                confidence = boxes.conf
                high_threshold = float(tracker._tracker.args.track_high_thresh)
                tracker_stage["high_confidence_selection"] = {
                    "threshold": high_threshold,
                    "scores": confidence.detach().cpu().tolist(),
                    "mask_type": f"{type(confidence >= high_threshold).__module__}."
                    f"{type(confidence >= high_threshold).__qualname__}",
                    "mask": (confidence >= high_threshold).detach().cpu().tolist(),
                }
                raw = original_byte_update(boxes, *args, **kwargs)
                tracker_stage["raw_tracker_output"] = describe(raw)
                tracker_stage["active_tracks_after_update"] = len(
                    tracker._tracker.tracked_stracks
                )
                return raw

            self._patched_attributes.append(
                (tracker._tracker, "update", original_byte_update)
            )
            tracker._tracker.update = observed_byte_update

            def observed_tracker_update(detections, frame_shape):
                assert self.current is not None
                stage = {
                    "status": "running",
                    "detection_count": len(detections),
                    "tracker_input_shape": [len(detections), 6],
                    "frame_shape": list(frame_shape),
                    "active_tracks_before_update": len(tracker._tracker.tracked_stracks),
                }
                self.current["bytetrack"] = stage
                self.current["active_stage"] = "bytetrack"
                track_ids = original_tracker_update(detections, frame_shape)
                stage.update(
                    {
                        "status": "complete",
                        "tracker_output_mapping": describe(track_ids),
                        "number_of_tracks": len(track_ids),
                    }
                )
                return track_ids

            self._patched_attributes.append((tracker, "update", original_tracker_update))
            tracker.update = observed_tracker_update

        import src.data.affect_module as affect_module_source

        original_align_face = affect_module_source.align_face

        def observed_align_face(frame_rgb_np, detection, output_size=224):
            assert self.current is not None
            self.current["active_stage"] = "face_crop"
            crops = self.current.setdefault("face_crops", [])
            entry = {
                "status": "running",
                "bbox_xyxy": detection.bbox.tolist(),
                "bbox_shape": list(detection.bbox.shape),
                "keypoint_shape": (
                    list(detection.landmarks.shape)
                    if detection.landmarks is not None
                    else None
                ),
                "requested_output_size": output_size,
            }
            crops.append(entry)
            crop = original_align_face(frame_rgb_np, detection, output_size=output_size)
            entry.update({"status": "complete", "crop": describe(crop)})
            return crop

        self._patched_attributes.append(
            (affect_module_source, "align_face", original_align_face)
        )
        affect_module_source.align_face = observed_align_face

        fer_model = self.module.fer_model

        def fer_pre_hook(_module, inputs):
            assert self.current is not None
            faces = inputs[0]
            self.current["active_stage"] = "fer"
            self.current["fer"] = {
                "status": "running",
                "number_of_faces": len(faces),
                "aligned_face_shapes": [list(face.shape) for face in faces],
                "batch_input_shape": _shape_of_sequence(faces),
            }

        def fer_output_hook(_module, _inputs, output):
            assert self.current is not None
            self.current["fer"].update(
                {
                    "status": "complete",
                    "probabilities": describe(output),
                }
            )

        self._handles.append(fer_model.register_forward_pre_hook(fer_pre_hook))
        self._handles.append(fer_model.register_forward_hook(fer_output_hook))

        inner_model = getattr(fer_model, "model", None)
        if isinstance(inner_model, torch.nn.Module):
            def model_pre_hook(_module, _args, kwargs):
                assert self.current is not None
                pixel_values = kwargs.get("pixel_values")
                if pixel_values is None and _args:
                    pixel_values = _args[0]
                self.current["fer"]["model_input"] = describe(pixel_values)

            def model_output_hook(_module, _args, kwargs, output):
                assert self.current is not None
                logits = getattr(output, "logits", output)
                self.current["fer"]["logits"] = describe(logits)

            self._handles.append(
                inner_model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
            )
            self._handles.append(
                inner_model.register_forward_hook(model_output_hook, with_kwargs=True)
            )

        def affect_output_hook(_module, _inputs, output):
            assert self.current is not None
            affect, reliability = output
            valid_faces = len(self.module.last_observations)
            self.current["active_stage"] = "aggregation"
            self.current["aggregation"] = {
                "status": "complete",
                "number_of_valid_faces": valid_faces,
                "smoothed_vectors_shape": [valid_faces, 7],
                "weight_vector_shape": [valid_faces],
                "final_affect_vector": describe(affect),
                "reliability": describe(reliability),
            }

        self._handles.append(self.module.register_forward_hook(affect_output_hook))

    def begin_frame(self, frame_index: int, frame: np.ndarray) -> dict[str, Any]:
        self.current = {
            "frame_index": frame_index,
            "status": "running",
            "active_stage": "input",
            "input": describe(frame),
        }
        self.context["frames"].append(self.current)
        self.context["current_frame_index"] = frame_index
        return self.current

    def close(self) -> None:
        """Remove observation hooks and restore every wrapped callable."""
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for owner, name, original in reversed(self._patched_attributes):
            setattr(owner, name, original)
        self._patched_attributes.clear()



def _a1_args(parameters: Mapping[str, Any]) -> argparse.Namespace:
    from src.data.affect_module import EMOTION_NAMES

    values = dict(parameters)
    values.setdefault("device", "auto")
    values.setdefault("retinaface_model", "buffalo_s")
    values.setdefault("insightface_root", "~/.insightface")
    values.setdefault("det_size", 640)
    values.setdefault("det_threshold", 0.45)
    values.setdefault("max_faces", 64)
    values.setdefault("fer_backend", "huggingface")
    values.setdefault("fer_model_id", "abhilash88/face-emotion-detection")
    values.setdefault("local_files_only", False)
    values.setdefault("fer_checkpoint", None)
    values.setdefault("fer_input_size", 224)
    values.setdefault("fer_source_emotions", list(EMOTION_NAMES))
    values.setdefault("tracking", True)
    values.setdefault("track_high_threshold", 0.45)
    values.setdefault("track_low_threshold", 0.10)
    values.setdefault("new_track_threshold", 0.45)
    values.setdefault("track_match_threshold", 0.80)
    values.setdefault("track_buffer", 8)
    values.setdefault("expected_faces", 8)
    values.setdefault("emotion_momentum", 0.60)
    values.setdefault("missed_detection_decay", 0.35)
    values.setdefault("max_feature_age", 1)
    return argparse.Namespace(**values)


def _affect_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    for method in config["methods"]["affect"]:
        if method.get("code") == "A1" and method.get("enabled", True):
            return dict(method.get("parameters", {}))
    raise ValueError("The diagnostic config does not contain enabled METHOD_A1")


def diagnose_sample(
    *, config_path: Path, sample_path: Path, output_dir: Path
) -> None:
    """Worker entry point; an A1 failure intentionally escapes this function."""
    global _ACTIVE_CONTEXT, _FAILURE_PATH

    config = load_config(config_path, project_root=PROJECT_ROOT)
    sample = read_json(sample_path)
    parameters = _affect_parameters(config)
    preprocessed_path = Path(sample["preprocessed_path"])

    _FAILURE_PATH = output_dir / "failure.json"
    _ACTIVE_CONTEXT = {
        "run_kind": "A1_read_only_diagnostic",
        "sample": sample,
        "a1_parameters": parameters,
        "preprocessed_sha256": sha256_file(preprocessed_path),
        "frames": [],
        "current_frame_index": None,
    }
    sys.excepthook = _diagnostic_excepthook

    print(f"Initializing unchanged METHOD_A1 for {sample['key']}", flush=True)
    from src.data.extract_affect_features import build_affect_module

    module = build_affect_module(_a1_args(parameters))
    observer = StageObserver(module, _ACTIVE_CONTEXT)

    with np.load(preprocessed_path, allow_pickle=False) as archive:
        frames = archive["frames"]
        npz_label = int(np.asarray(archive["label"]).item())
        npz_video_name = str(np.asarray(archive["video_name"]).item())
        npz_category = str(np.asarray(archive["category"]).item()).lower()
        keys = list(archive.files)
    contract = {
        "keys": keys,
        "frames": describe(frames),
        "label": npz_label,
        "video_name": npz_video_name,
        "category": npz_category,
    }
    _ACTIVE_CONTEXT["npz_contract"] = contract
    if frames.shape != (8, 640, 640, 3) or frames.dtype != np.uint8:
        raise ValueError(f"Invalid diagnostic NPZ frames contract: {frames.shape}/{frames.dtype}")
    if npz_label != int(sample["label"]):
        raise ValueError(f"NPZ label {npz_label} differs from CSV label {sample['label']}")
    if npz_category != str(sample["category"]).lower():
        raise ValueError(
            f"NPZ category {npz_category!r} differs from CSV category {sample['category']!r}"
        )
    if npz_video_name != Path(str(sample["csv_video_path"])).stem:
        raise ValueError("NPZ video identity differs from the CSV video identity")

    module.reset()
    rows = []
    for frame_index, frame_rgb in enumerate(frames):
        frame_record = observer.begin_frame(frame_index, frame_rgb)
        affect, reliability = module(frame_rgb)
        row = torch.cat(
            [
                affect.detach().float().cpu(),
                reliability.detach().float().cpu().reshape(1),
            ]
        ).numpy()
        rows.append(row)
        frame_record["feature_row"] = describe(row)
        frame_record["status"] = "complete"

    matrix = np.stack(rows).astype(np.float32)
    result = {
        "status": "success_control" if sample.get("control") else "unexpected_success",
        "completed_at": utc_now(),
        "sample": sample,
        "npz_contract": contract,
        "feature_matrix": describe(matrix),
        "diagnostic_context": _ACTIVE_CONTEXT,
    }
    write_json_exclusive(output_dir / "result.json", result)
    print(f"A1 diagnostic completed without exception for {sample['key']}", flush=True)


def _sample_directory_name(sample: Mapping[str, Any]) -> str:
    role = "control" if sample.get("control") else "failure"
    return f"{role}__{str(sample['key']).replace('/', '__')}"


def run_controller(config_path: Path, partial_validation_path: Path) -> Path:
    config = load_config(config_path, project_root=PROJECT_ROOT)
    partial = read_json(partial_validation_path)
    missing = partial.get("missing_outputs")
    if not isinstance(missing, list) or len(missing) != 100:
        raise ValueError(
            f"Expected the preserved 100-sample A1 failure inventory, got "
            f"{len(missing) if isinstance(missing, list) else 'invalid metadata'}"
        )
    if partial.get("valid_output_count") != 1095:
        raise ValueError("Preserved partial A1 metadata no longer reports 1095 valid outputs")

    partial_directory = partial_validation_path.parent
    selected = select_failures(missing)
    missing_keys = {str(item["key"]) for item in missing}
    control = select_control(config, partial_directory, missing_keys)
    samples = [*selected, control]

    artifacts_root = Path(config["experiment"]["artifacts_root"])
    run_id = new_id("RUN")
    run_dir = create_exclusive_dir(
        artifacts_root / "diagnostics" / "a1_failures" / run_id
    )
    write_json_exclusive(run_dir / "failure_inventory.json", missing)
    write_json_exclusive(
        run_dir / "request.json",
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "purpose": "read-only A1 failure root-cause diagnosis",
            "partial_feature_id": partial.get("feature_id"),
            "partial_validation": str(partial_validation_path.resolve()),
            "config": str(config_path.resolve()),
            "distribution": failure_distribution(missing),
            "selected_samples": samples,
        },
    )

    outcomes = []
    for sample in samples:
        sample_dir = create_exclusive_dir(run_dir / _sample_directory_name(sample))
        sample_json = sample_dir / "sample.json"
        write_json_exclusive(sample_json, sample)
        command = [
            sys.executable,
            "-m",
            "experiments_v2.diagnostics.a1_failure",
            "--worker",
            "--config",
            str(config_path.resolve()),
            "--sample",
            str(sample_json.resolve()),
            "--output-dir",
            str(sample_dir.resolve()),
        ]
        child_error = None
        try:
            run_logged(command, cwd=PROJECT_ROOT, log_path=sample_dir / "diagnostic.log")
        except RuntimeError as exc:
            child_error = str(exc)
        failure_path = sample_dir / "failure.json"
        result_path = sample_dir / "result.json"
        if failure_path.is_file():
            status = "reproduced_failure"
        elif result_path.is_file():
            status = read_json(result_path)["status"]
        else:
            status = "worker_failed_without_diagnostic_record"
        outcomes.append(
            {
                "sample": sample["key"],
                "control": bool(sample.get("control")),
                "status": status,
                "directory": str(sample_dir.resolve()),
                "controller_error": child_error,
            }
        )

    write_json_exclusive(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "completed_at": utc_now(),
            "partial_feature_id": partial.get("feature_id"),
            "distribution": failure_distribution(missing),
            "outcomes": outcomes,
        },
    )
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only stage and traceback diagnostic for partial METHOD_A1"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partial-validation", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.sample is None or args.output_dir is None:
            parser.error("worker mode requires --sample and --output-dir")
    elif args.partial_validation is None:
        parser.error("controller mode requires --partial-validation")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        diagnose_sample(
            config_path=args.config.resolve(),
            sample_path=args.sample.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        return
    run_dir = run_controller(
        args.config.resolve(), args.partial_validation.resolve()
    )
    print(f"Diagnostic artifact: {run_dir}")


if __name__ == "__main__":
    main()

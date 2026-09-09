"""Recover only the known missing METHOD_A1 singleton-detection outputs.

The workflow is intentionally gated:

1. compare unchanged A1 with the V2 shim on successful controls;
2. recover and stage the three diagnosed cross-class failures;
3. copy the 1095 validated parent arrays into a new immutable FEATURE artifact;
4. extract only the remaining missing identities and publish after full validation.

No legacy source, site-package file, model, or dataset file is modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence, TextIO

import numpy as np
import torch

from experiments_v2.certification.environment import inspect_environment
from experiments_v2.compatibility.bytetrack_singleton import (
    COMPATIBILITY_FIX_ID,
    apply_bytetrack_singleton_numpy_mask_v1,
)
from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    dataset_identity,
    fingerprint,
    new_id,
    read_json,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.config import load_config
from experiments_v2.diagnostics.a1_failure import (
    StageObserver,
    _a1_args,
    _affect_parameters,
    _csv_samples,
    describe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PARENT_FEATURE_ID = "FEATURE_20260908T110815068960Z_8E01D785"
EXPECTED_MODEL_ID = "MODEL_20260908T110815045056Z_176DFBA6"
EXPECTED_I1_FEATURE_ID = "FEATURE_20260908T113809721154Z_474E2629"
REPRESENTATIVE_KEYS = (
    "train/low/view2024",
    "train/mid/view1230",
    "train/high/view2396",
)
CONTROL_KEYS = (
    "train/low/view642",
    "train/mid/view55",
    "train/high/view2392",
)
STRICT_ATOL = 1e-7


class RecoveryLog:
    def __init__(self, path: Path) -> None:
        self.handle: TextIO = path.open("x", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        self.handle.write(line + "\n")
        self.handle.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def _git_identity() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _command() -> list[str]:
    return [sys.executable, "-m", "experiments_v2.recovery.a1_singleton", *sys.argv[1:]]


def _sample_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    samples = _csv_samples(config)
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = str(sample["key"])
        if key in result:
            raise ValueError(f"Duplicate CSV sample identity: {key}")
        expected_label = {"low": 0, "mid": 1, "high": 2}.get(sample["category"])
        if expected_label is None or sample["label"] != expected_label:
            raise ValueError(
                f"CSV category/label mismatch for {key}: {sample['label']}"
            )
        result[key] = sample
    if len(result) != 1195:
        raise ValueError(f"Expected 1195 unique CSV samples, found {len(result)}")
    return result


def _feature_relative(sample: Mapping[str, Any]) -> Path:
    return Path(f"{sample['key']}.npy")


def _load_npz_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(sample["preprocessed_path"])
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        label = int(np.asarray(archive["label"]).item())
        category = str(np.asarray(archive["category"]).item()).lower()
        video_name = str(np.asarray(archive["video_name"]).item())
    expected_name = Path(str(sample["csv_video_path"])).stem
    if label != int(sample["label"]):
        raise ValueError(f"NPZ label mismatch for {sample['key']}")
    if category != str(sample["category"]):
        raise ValueError(f"NPZ category mismatch for {sample['key']}")
    if video_name != expected_name:
        raise ValueError(f"NPZ identity mismatch for {sample['key']}")
    return {
        "path": str(path.resolve()),
        "keys": keys,
        "label": label,
        "category": category,
        "video_name": video_name,
    }


def _validate_matrix(matrix: np.ndarray, key: str) -> None:
    if matrix.shape != (8, 8):
        raise ValueError(f"{key} has feature shape {matrix.shape}; expected (8, 8)")
    if matrix.dtype != np.float32:
        raise ValueError(f"{key} has feature dtype {matrix.dtype}; expected float32")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{key} contains NaN or Inf")


def _track_signature(frame_details: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [observation.get("track_id") for observation in frame["tracks"]]
        for frame in frame_details
    ]


def _run_equivalence(
    module: Any,
    controls: Sequence[Mapping[str, Any]],
    parent_data: Path,
    log: RecoveryLog,
) -> tuple[dict[str, Any], Any]:
    from src.data.extract_affect_features import extract_video

    original_results: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    original_seconds = 0.0
    for sample in controls:
        source = Path(sample["preprocessed_path"])
        started = perf_counter()
        matrix, _reliability, details = extract_video(module, source)
        original_seconds += perf_counter() - started
        _validate_matrix(matrix, str(sample["key"]))
        original_results[str(sample["key"])] = (matrix, details)
        log.write(f"equivalence original complete: {sample['key']}")

    shim = apply_bytetrack_singleton_numpy_mask_v1(module)
    shim_results: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    shim_seconds = 0.0
    comparisons = []
    all_face_counts: list[int] = []
    for sample in controls:
        key = str(sample["key"])
        source = Path(sample["preprocessed_path"])
        started = perf_counter()
        matrix, _reliability, details = extract_video(module, source)
        shim_seconds += perf_counter() - started
        _validate_matrix(matrix, key)
        shim_results[key] = (matrix, details)
        original_matrix, original_details = original_results[key]
        difference = np.abs(original_matrix - matrix)
        max_difference = float(difference.max()) if difference.size else 0.0
        exact_equal = bool(np.array_equal(original_matrix, matrix))
        within_tolerance = bool(
            np.allclose(original_matrix, matrix, rtol=0.0, atol=STRICT_ATOL)
        )
        tracks_equal = _track_signature(original_details) == _track_signature(details)
        preserved = np.load(parent_data / _feature_relative(sample), allow_pickle=False)
        _validate_matrix(preserved, key)
        all_face_counts.extend(len(frame["tracks"]) for frame in details)
        comparisons.append(
            {
                "sample": key,
                "shape": list(matrix.shape),
                "original_vs_shim_exact_equal": exact_equal,
                "original_vs_shim_max_absolute_difference": max_difference,
                "original_vs_shim_within_atol": within_tolerance,
                "track_ids_equal": tracks_equal,
                "face_counts": [len(frame["tracks"]) for frame in details],
                "original_vs_preserved_exact_equal": bool(
                    np.array_equal(original_matrix, preserved)
                ),
                "shim_vs_preserved_exact_equal": bool(np.array_equal(matrix, preserved)),
            }
        )
        log.write(
            f"equivalence shim complete: {key}; exact={exact_equal}; "
            f"max_abs={max_difference:.9g}; tracks_equal={tracks_equal}"
        )

    passed = (
        all(item["original_vs_shim_within_atol"] for item in comparisons)
        and all(item["track_ids_equal"] for item in comparisons)
        and 2 in all_face_counts
        and any(count > 2 for count in all_face_counts)
        and shim.trigger_count == 0
    )
    report = {
        "status": "passed" if passed else "failed",
        "sample_count": len(controls),
        "strict_atol": STRICT_ATOL,
        "all_exact_equal": all(
            item["original_vs_shim_exact_equal"] for item in comparisons
        ),
        "max_absolute_difference": max(
            item["original_vs_shim_max_absolute_difference"] for item in comparisons
        ),
        "all_track_ids_equal": all(item["track_ids_equal"] for item in comparisons),
        "contains_two_face_frame": 2 in all_face_counts,
        "contains_multi_face_frame": any(count > 2 for count in all_face_counts),
        "shim_trigger_count": shim.trigger_count,
        "original_seconds": original_seconds,
        "shim_seconds": shim_seconds,
        "comparisons": comparisons,
    }
    if not passed:
        raise RuntimeError("Successful-control A1 equivalence gate failed")
    return report, shim


def _extract_observed(module: Any, sample: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(Path(sample["preprocessed_path"]), allow_pickle=False) as archive:
        frames = archive["frames"]
    if frames.shape != (8, 640, 640, 3) or frames.dtype != np.uint8:
        raise ValueError(f"Invalid preprocessed frame contract for {sample['key']}")

    context: dict[str, Any] = {"sample": dict(sample), "frames": []}
    observer = StageObserver(module, context)
    try:
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
    finally:
        observer.close()
    _validate_matrix(matrix, str(sample["key"]))
    return matrix, context


def _representative_gate(
    module: Any,
    shim: Any,
    representatives: Sequence[Mapping[str, Any]],
    log: RecoveryLog,
) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    matrices: dict[str, np.ndarray] = {}
    reports = []
    total_seconds = 0.0
    for sample in representatives:
        key = str(sample["key"])
        _load_npz_metadata(sample)
        triggers_before = shim.trigger_count
        started = perf_counter()
        matrix, context = _extract_observed(module, sample)
        elapsed = perf_counter() - started
        total_seconds += elapsed
        trigger_delta = shim.trigger_count - triggers_before
        singleton_frames = []
        for frame in context["frames"]:
            retinaface = frame.get("retinaface", {})
            if retinaface.get("number_of_faces") != 1:
                continue
            stage_valid = (
                frame.get("bytetrack", {}).get("status") == "complete"
                and len(frame.get("face_crops", [])) == 1
                and frame.get("face_crops", [{}])[0].get("crop", {}).get("shape")
                == [224, 224, 3]
                and frame.get("fer", {}).get("status") == "complete"
                and frame.get("fer", {}).get("probabilities", {}).get("shape") == [1, 7]
                and frame.get("aggregation", {}).get("status") == "complete"
                and frame.get("aggregation", {})
                .get("final_affect_vector", {})
                .get("shape")
                == [7]
            )
            singleton_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "stage_chain_complete": stage_valid,
                    "retinaface": retinaface,
                    "bytetrack": frame.get("bytetrack"),
                    "face_crops": frame.get("face_crops"),
                    "fer": frame.get("fer"),
                    "aggregation": frame.get("aggregation"),
                }
            )
        passed = (
            trigger_delta > 0
            and bool(singleton_frames)
            and all(frame["stage_chain_complete"] for frame in singleton_frames)
        )
        reports.append(
            {
                "sample": key,
                "label": sample["label"],
                "category": sample["category"],
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "finite": bool(np.isfinite(matrix).all()),
                "shim_trigger_count": trigger_delta,
                "extraction_seconds": elapsed,
                "singleton_frames": singleton_frames,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"Representative singleton stage gate failed: {key}")
        matrices[key] = matrix
        log.write(
            f"representative recovered: {key}; singleton_triggers={trigger_delta}; "
            f"seconds={elapsed:.3f}"
        )
    return {"status": "passed", "samples": reports}, matrices, total_seconds


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _save_array_exclusive(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, matrix, allow_pickle=False)


def validate_complete_a1(
    data_dir: Path, samples: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected_paths = {
        _feature_relative(sample).as_posix(): sample for sample in samples.values()
    }
    actual_paths = {
        path.relative_to(data_dir).as_posix(): path
        for path in data_dir.glob("*/*/*.npy")
    }
    missing = sorted(set(expected_paths) - set(actual_paths))
    extras = sorted(set(actual_paths) - set(expected_paths))
    malformed = []
    identity_mismatches = []
    duplicate_paths = len(expected_paths) != len(samples)
    counts = {split: 0 for split in ("train", "val", "test")}
    for relative, sample in expected_paths.items():
        path = actual_paths.get(relative)
        if path is None:
            continue
        try:
            matrix = np.load(path, allow_pickle=False)
            _validate_matrix(matrix, str(sample["key"]))
        except Exception as exc:
            malformed.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        try:
            _load_npz_metadata(sample)
        except Exception as exc:
            identity_mismatches.append(
                {"sample": sample["key"], "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        counts[str(sample["split"])] += 1

    ready = (
        not missing
        and not extras
        and not malformed
        and not identity_mismatches
        and not duplicate_paths
        and counts == {"train": 939, "val": 124, "test": 132}
    )
    return {
        "status": "complete" if ready else "failed",
        "ready": ready,
        "expected_count": len(samples),
        "validated_count": sum(counts.values()),
        "counts": counts,
        "missing_count": len(missing),
        "missing": missing[:100],
        "extra_count": len(extras),
        "extras": extras[:100],
        "duplicate_paths": duplicate_paths,
        "malformed_count": len(malformed),
        "malformed": malformed[:100],
        "identity_mismatch_count": len(identity_mismatches),
        "identity_mismatches": identity_mismatches[:100],
        "contract": {"shape": [8, 8], "dtype": "float32", "finite": True},
    }


def _legacy_manifest(parameters: Mapping[str, Any]) -> dict[str, Any]:
    from src.data.affect_module import EMOTION_NAMES
    from src.data.feature_schema import AFFECT_COLUMNS, AFFECT_FEATURE_SCHEMA

    return {
        "format_version": 2,
        "feature_schema": AFFECT_FEATURE_SCHEMA,
        "shape_per_video": [8, 8],
        "columns": list(AFFECT_COLUMNS),
        "detector": {
            "library": "insightface",
            "model": parameters.get("retinaface_model", "buffalo_s"),
            "threshold": parameters.get("det_threshold", 0.45),
            "input_size": parameters.get("det_size", 640),
        },
        "tracker": {
            "enabled": parameters.get("tracking", True),
            "library": "ultralytics-bytetrack",
            "high_threshold": parameters.get("track_high_threshold", 0.45),
            "low_threshold": parameters.get("track_low_threshold", 0.10),
            "new_track_threshold": parameters.get("new_track_threshold", 0.45),
            "match_threshold": parameters.get("track_match_threshold", 0.80),
            "track_buffer": parameters.get("track_buffer", 8),
        },
        "fer": {
            "backend": parameters.get("fer_backend", "huggingface"),
            "model": parameters.get(
                "fer_model_id", "abhilash88/face-emotion-detection"
            ),
            "source_emotions": list(parameters.get("fer_source_emotions", EMOTION_NAMES)),
        },
        "aggregation": {
            "expected_faces": parameters.get("expected_faces", 8),
            "emotion_momentum": parameters.get("emotion_momentum", 0.60),
            "missed_detection_decay": parameters.get("missed_detection_decay", 0.35),
            "max_feature_age": parameters.get("max_feature_age", 1),
        },
        "compatibility_fix": {
            "id": COMPATIBILITY_FIX_ID,
            "scope": "V2 only",
            "trigger": "exactly one detection",
            "semantics": "representation normalization only",
        },
        "summary": {
            "input_videos": 1195,
            "processed_videos": 100,
            "inherited_videos": 1095,
            "failed_videos": 0,
        },
    }


def _environment_record(module: Any) -> dict[str, Any]:
    report = inspect_environment("3.10")
    return {
        "python": report["python"],
        "platform": report["platform"],
        "hardware": report["hardware"],
        "packages": {
            name: item["version"] for name, item in report["packages"].items()
        },
        "fer_device": str(module.fer_model.device),
    }


def recover(
    *, config_path: Path, partial_validation_path: Path
) -> tuple[Path, Path]:
    config = load_config(config_path, PROJECT_ROOT)
    parameters = _affect_parameters(config)
    partial = read_json(partial_validation_path)
    parent_dir = partial_validation_path.parent
    parent_data = parent_dir / "data"
    if partial.get("feature_id") != EXPECTED_PARENT_FEATURE_ID:
        raise ValueError("Recovery was not pointed at the approved parent FEATURE artifact")
    if partial.get("valid_output_count") != 1095 or partial.get("missing_count") != 100:
        raise ValueError("Approved parent no longer reports 1095 valid + 100 missing")
    parent_request = read_json(parent_dir / "request.json")
    if parent_request["request"]["model_id"] != EXPECTED_MODEL_ID:
        raise ValueError("Approved parent model ID changed")

    samples = _sample_map(config)
    missing_records = partial.get("missing_outputs")
    if not isinstance(missing_records, list):
        raise ValueError("Parent failure inventory is missing")
    missing_keys = [str(item["key"]) for item in missing_records]
    if len(set(missing_keys)) != 100 or any(key not in samples for key in missing_keys):
        raise ValueError("Parent failure inventory is not 100 unique CSV identities")
    controls = [samples[key] for key in CONTROL_KEYS]
    representatives = [samples[key] for key in REPRESENTATIVE_KEYS]
    if any(key not in missing_keys for key in REPRESENTATIVE_KEYS):
        raise ValueError("Approved representative is not in the parent missing set")

    artifacts_root = Path(config["experiment"]["artifacts_root"])
    recovery_run_id = new_id("RUN")
    run_dir = create_exclusive_dir(
        artifacts_root / "recovery" / "a1_singleton" / recovery_run_id
    )
    log = RecoveryLog(run_dir / "recovery.log")
    feature_dir: Path | None = None
    feature_id: str | None = None
    command = _command()
    write_json_exclusive(
        run_dir / "request.json",
        {
            "run_id": recovery_run_id,
            "created_at": utc_now(),
            "command": command,
            "config": str(config_path.resolve()),
            "partial_validation": str(partial_validation_path.resolve()),
            "parent_feature_id": EXPECTED_PARENT_FEATURE_ID,
            "model_id": EXPECTED_MODEL_ID,
            "compatibility_fix_id": COMPATIBILITY_FIX_ID,
            "controls": list(CONTROL_KEYS),
            "representatives": list(REPRESENTATIVE_KEYS),
            "missing_count": len(missing_keys),
        },
    )

    try:
        log.write("building unchanged METHOD_A1 components")
        from src.data.extract_affect_features import build_affect_module, extract_video
        from src.data.feature_schema import AFFECT_FEATURE_SCHEMA

        module = build_affect_module(_a1_args(parameters))
        equivalence, shim = _run_equivalence(module, controls, parent_data, log)
        write_json_exclusive(run_dir / "equivalence.json", equivalence)
        log.write(
            f"successful-control gate passed: samples={equivalence['sample_count']}; "
            f"max_abs={equivalence['max_absolute_difference']:.9g}"
        )

        representative_report, staged, representative_seconds = _representative_gate(
            module, shim, representatives, log
        )
        write_json_exclusive(
            run_dir / "representative_recovery.json", representative_report
        )
        log.write("representative failed-sample gate passed: 3/3")

        split_paths = {
            split: Path(path) for split, path in config["dataset"]["split_files"].items()
        }
        dataset_record = dataset_identity(
            split_paths,
            Path(config["dataset"]["preprocessed_input_dir"]),
            config["dataset"]["preprocessing"],
        )
        if (
            dataset_record["fingerprint"]
            != parent_request["request"]["dataset_fingerprint"]
        ):
            raise ValueError("Current split/preprocessing fingerprint differs from parent")

        feature_id = new_id("FEATURE")
        feature_dir = create_exclusive_dir(
            artifacts_root
            / "features"
            / "affect"
            / "METHOD_A1"
            / EXPECTED_MODEL_ID
            / feature_id
        )
        data_dir = feature_dir / "data"
        data_dir.mkdir()
        recovery_provenance = {
            "compatibility_fix_id": COMPATIBILITY_FIX_ID,
            "parent_feature_id": EXPECTED_PARENT_FEATURE_ID,
            "recovery_scope": "100 parent-missing samples only",
        }
        write_json_exclusive(
            feature_dir / "request.json",
            {
                "feature_id": feature_id,
                "created_at": utc_now(),
                "fingerprint": parent_request["fingerprint"],
                "recovery_fingerprint": fingerprint(recovery_provenance),
                "request": parent_request["request"],
                "recovery": recovery_provenance,
            },
        )

        checksum_entries: dict[str, dict[str, Any]] = {}
        inherited = [sample for key, sample in samples.items() if key not in set(missing_keys)]
        if len(inherited) != 1095:
            raise ValueError(f"Expected 1095 inherited identities, got {len(inherited)}")
        for index, sample in enumerate(inherited, start=1):
            relative = _feature_relative(sample)
            source = parent_data / relative
            destination = data_dir / relative
            matrix = np.load(source, allow_pickle=False)
            _validate_matrix(matrix, str(sample["key"]))
            source_checksum = sha256_file(source)
            _copy_exclusive(source, destination)
            destination_checksum = sha256_file(destination)
            if source_checksum != destination_checksum:
                raise RuntimeError(f"Physical copy checksum mismatch: {relative}")
            checksum_entries[relative.as_posix()] = {
                "origin": "inherited",
                "label": sample["label"],
                "source_feature_id": EXPECTED_PARENT_FEATURE_ID,
                "source_sha256": source_checksum,
                "feature_sha256": destination_checksum,
            }
            if index % 250 == 0:
                log.write(f"physically copied validated parent features: {index}/1095")
        log.write("physically copied validated parent features: 1095/1095")

        recovery_timings = []
        recovery_failures = []
        recovery_seconds = representative_seconds
        for key, matrix in staged.items():
            sample = samples[key]
            relative = _feature_relative(sample)
            destination = data_dir / relative
            _save_array_exclusive(destination, matrix)
            checksum_entries[relative.as_posix()] = {
                "origin": "recovered",
                "label": sample["label"],
                "feature_sha256": sha256_file(destination),
                "representative_gate": True,
            }
            sample_report = next(
                item
                for item in representative_report["samples"]
                if item["sample"] == key
            )
            recovery_timings.append(
                {
                    "sample": key,
                    "seconds": sample_report["extraction_seconds"],
                    "shim_trigger_count": sample_report["shim_trigger_count"],
                }
            )

        remaining_keys = [key for key in missing_keys if key not in staged]
        for index, key in enumerate(remaining_keys, start=1):
            sample = samples[key]
            destination = data_dir / _feature_relative(sample)
            try:
                _load_npz_metadata(sample)
                triggers_before = shim.trigger_count
                started = perf_counter()
                matrix, _reliability, _details = extract_video(
                    module, Path(sample["preprocessed_path"])
                )
                elapsed = perf_counter() - started
                recovery_seconds += elapsed
                _validate_matrix(matrix, key)
                _save_array_exclusive(destination, matrix)
                checksum_entries[_feature_relative(sample).as_posix()] = {
                    "origin": "recovered",
                    "label": sample["label"],
                    "feature_sha256": sha256_file(destination),
                    "representative_gate": False,
                }
                recovery_timings.append(
                    {
                        "sample": key,
                        "seconds": elapsed,
                        "shim_trigger_count": shim.trigger_count - triggers_before,
                    }
                )
            except Exception as exc:
                recovery_failures.append(
                    {
                        "sample": key,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            if index % 10 == 0 or index == len(remaining_keys):
                log.write(
                    f"missing-sample recovery progress: {index + len(staged)}/100; "
                    f"failures={len(recovery_failures)}"
                )

        if recovery_failures:
            write_json_exclusive(
                feature_dir / "failure.json",
                {
                    "status": "failed",
                    "feature_id": feature_id,
                    "recovered_count": len(recovery_timings),
                    "failures": recovery_failures,
                    "failed_at": utc_now(),
                },
            )
            raise RuntimeError(
                f"A1 recovery retained {len(recovery_failures)} failed samples"
            )
        if len(recovery_timings) != 100:
            raise RuntimeError(
                f"Expected 100 recovered timings, got {len(recovery_timings)}"
            )

        validation = validate_complete_a1(data_dir, samples)
        write_json_exclusive(feature_dir / "validation.json", validation)
        if not validation["ready"]:
            raise RuntimeError("New A1 feature artifact failed full validation")

        checksums = {
            "algorithm": "sha256",
            "parent_feature_id": EXPECTED_PARENT_FEATURE_ID,
            "inherited_file_count": sum(
                value["origin"] == "inherited" for value in checksum_entries.values()
            ),
            "recovered_file_count": sum(
                value["origin"] == "recovered" for value in checksum_entries.values()
            ),
            "files": dict(sorted(checksum_entries.items())),
        }
        write_json_exclusive(feature_dir / "checksums.json", checksums)
        write_json_exclusive(feature_dir / "recovery_timings.json", recovery_timings)

        legacy_manifest = _legacy_manifest(parameters)
        write_json_exclusive(data_dir / "extraction_manifest.json", legacy_manifest)
        environment = _environment_record(module)
        git = _git_identity()
        manifest = {
            "status": "complete",
            "feature_id": feature_id,
            "method_id": "METHOD_A1",
            "method_code": "A1",
            "method_name": "legacy_affect",
            "method_version": "1",
            "model_id": EXPECTED_MODEL_ID,
            "category": "affect",
            "feature_schema": AFFECT_FEATURE_SCHEMA,
            "shape_per_video": [8, 8],
            "feature_dim": 8,
            "fingerprint": parent_request["fingerprint"],
            "recovery_fingerprint": fingerprint(recovery_provenance),
            "dataset_identity": dataset_record,
            "preprocessing_fingerprint": fingerprint(
                config["dataset"]["preprocessing"]
            ),
            "parameters": parameters,
            "parent_feature_id": EXPECTED_PARENT_FEATURE_ID,
            "inherited_file_count": 1095,
            "recovered_file_count": 100,
            "validated_files": 1195,
            "compatibility_fix_id": COMPATIBILITY_FIX_ID,
            "compatibility_fix": shim.metadata,
            "equivalence": equivalence,
            "representative_recovery": representative_report,
            "extraction_seconds": recovery_seconds,
            "recovery_command": command,
            "checksums_file": "checksums.json",
            "validation_file": "validation.json",
            "environment": environment,
            "git_commit": git["commit"],
            "git": git,
            "legacy_manifest": legacy_manifest,
            "created_at": utc_now(),
        }
        write_json_exclusive(feature_dir / "manifest.json", manifest)
        write_json_exclusive(
            run_dir / "manifest.json",
            {
                "status": "complete",
                "run_id": recovery_run_id,
                "feature_id": feature_id,
                "parent_feature_id": EXPECTED_PARENT_FEATURE_ID,
                "model_id": EXPECTED_MODEL_ID,
                "compatibility_fix_id": COMPATIBILITY_FIX_ID,
                "inherited_file_count": 1095,
                "recovered_file_count": 100,
                "validation": validation,
                "completed_at": utc_now(),
            },
        )
        log.write(f"published complete immutable A1 artifact: {feature_id}")
        return run_dir, feature_dir
    except Exception as exc:
        failure = {
            "status": "failed",
            "run_id": recovery_run_id,
            "feature_id": feature_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
        }
        if not (run_dir / "failure.json").exists():
            write_json_exclusive(run_dir / "failure.json", failure)
        if feature_dir is not None and not (feature_dir / "failure.json").exists():
            write_json_exclusive(feature_dir / "failure.json", failure)
        log.write(f"recovery stopped without publishing: {type(exc).__name__}: {exc}")
        raise
    finally:
        log.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover the approved 100 METHOD_A1 singleton failures"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partial-validation", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir, feature_dir = recover(
        config_path=args.config.resolve(),
        partial_validation_path=args.partial_validation.resolve(),
    )
    print(json.dumps({"run_dir": str(run_dir), "feature_dir": str(feature_dir)}, indent=2))


if __name__ == "__main__":
    main()

"""Data, artifact, and route readiness for A1 + I1 certification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments_v2.certification.environment import inspect_environment


Record = tuple[str, str, int, str, str]


def _read_records(split_files: Mapping[str, str]) -> tuple[list[Record], dict[str, Any]]:
    records: list[Record] = []
    checks = {}
    for split, value in split_files.items():
        path = Path(value)
        split_records = []
        malformed = []
        if path.is_file():
            for line_number, raw_line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    malformed.append(line_number)
                    continue
                video_text, label_text = parts
                try:
                    label = int(label_text)
                except ValueError:
                    malformed.append(line_number)
                    continue
                video = Path(video_text)
                split_records.append(
                    (split, video_text, label, video.stem, video.parent.name.lower())
                )
        records.extend(split_records)
        checks[split] = {
            "path": str(path),
            "exists": path.is_file(),
            "record_count": len(split_records),
            "malformed_line_numbers": malformed[:20],
            "ready": path.is_file() and bool(split_records) and not malformed,
        }
    return records, checks


def _expected_artifact_check(
    root: Path,
    records: list[Record],
    *,
    kind: str,
    required_manifest: str | None = None,
) -> dict[str, Any]:
    missing = []
    missing_count = 0
    matched = 0
    for split, _video_text, label, stem, category in records:
        if kind == "matrix":
            path = root / split / f"{stem}_label{label}.npy"
        else:
            suffix = ".npz" if kind == "preprocessed" else ".npy"
            path = root / split / category / f"{stem}{suffix}"
        if path.is_file():
            matched += 1
        else:
            missing_count += 1
            if len(missing) < 20:
                missing.append(str(path))
    manifest_ready = (
        (root / required_manifest).is_file() if required_manifest is not None else True
    )
    ready = (
        root.is_dir()
        and bool(records)
        and matched == len(records)
        and manifest_ready
    )
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "expected_file_count": len(records),
        "matched_file_count": matched,
        "missing_count": missing_count,
        "missing_examples": missing,
        "required_manifest": required_manifest,
        "manifest_ready": manifest_ready,
        "ready": ready,
    }


def _raw_dataset_check(root: Path, records: list[Record]) -> dict[str, Any]:
    missing = []
    matched = 0
    for _split, video_text, _label, _stem, _category in records:
        path = root / video_text
        if path.is_file():
            matched += 1
        elif len(missing) < 20:
            missing.append(str(path))
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "expected_video_count": len(records),
        "matched_video_count": matched,
        "missing_count": len(records) - matched,
        "missing_examples": missing,
        "ready": root.is_dir() and bool(records) and matched == len(records),
    }


def _checkpoint_check(root: Path) -> dict[str, Any]:
    checkpoint = root / "best_model_behavioral.pth"
    return {
        "root": str(root),
        "checkpoint": str(checkpoint),
        "ready": checkpoint.is_file(),
        "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
    }


def inspect_data_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    certification_paths = config["certification"]["paths"]
    records, split_files = _read_records(dataset["split_files"])
    splits_ready = all(item["ready"] for item in split_files.values())

    method_entries = {
        category: next(
            entry
            for entry in config["methods"][category]
            if entry.get("enabled", True)
        )
        for category in ("affect", "interaction")
    }
    affect_cache = _expected_artifact_check(
        Path(method_entries["affect"]["legacy_feature_dir"]),
        records,
        kind="feature",
        required_manifest="extraction_manifest.json",
    )
    interaction_cache = _expected_artifact_check(
        Path(method_entries["interaction"]["legacy_feature_dir"]),
        records,
        kind="feature",
        required_manifest="extraction_manifest.json",
    )
    preprocessed = _expected_artifact_check(
        Path(dataset["preprocessed_input_dir"]),
        records,
        kind="preprocessed",
    )
    raw_dataset = _raw_dataset_check(
        Path(certification_paths["dataset_root"]), records
    )
    matrices = _expected_artifact_check(
        Path(certification_paths["legacy_matrix_root"]),
        records,
        kind="matrix",
        required_manifest="build_manifest.json",
    )
    checkpoint = _checkpoint_check(Path(certification_paths["legacy_checkpoint_root"]))

    reuse_data_ready = (
        splits_ready
        and affect_cache["ready"]
        and interaction_cache["ready"]
        and matrices["ready"]
        and checkpoint["ready"]
    )
    rebuild_preprocessed_data_ready = splits_ready and preprocessed["ready"]
    raw_preparation_available = splits_ready and raw_dataset["ready"]
    return {
        "status": "READY"
        if reuse_data_ready or rebuild_preprocessed_data_ready
        else "NOT_READY",
        "ready": reuse_data_ready or rebuild_preprocessed_data_ready,
        "split_files": split_files,
        "splits_ready": splits_ready,
        "raw_dataset": raw_dataset,
        "preprocessed_frames": preprocessed,
        "legacy_affect_cache": affect_cache,
        "legacy_interaction_cache": interaction_cache,
        "legacy_behavioral_matrices": matrices,
        "legacy_behavioral_checkpoint": checkpoint,
        "route_data_readiness": {
            "reuse_legacy_artifacts": reuse_data_ready,
            "rebuild_from_preprocessed": rebuild_preprocessed_data_ready,
            "raw_data_requires_preprocessing": raw_preparation_available,
        },
    }


def build_preflight_report(config: Mapping[str, Any]) -> dict[str, Any]:
    environment = inspect_environment(
        str(config["certification"].get("required_python", "3.10"))
    )
    data = inspect_data_readiness(config)
    reuse_ready = (
        data["route_data_readiness"]["reuse_legacy_artifacts"]
        and environment["profiles"]["reuse_and_training"]["ready"]
    )
    rebuild_ready = (
        data["route_data_readiness"]["rebuild_from_preprocessed"]
        and environment["profiles"]["feature_extraction_and_training"]["ready"]
    )
    selected_path = (
        "reuse_legacy_artifacts"
        if reuse_ready
        else ("rebuild_from_preprocessed" if rebuild_ready else None)
    )
    raw_available = data["route_data_readiness"]["raw_data_requires_preprocessing"]
    explanations = []
    if selected_path is None:
        explanations.append(
            "No executable certification route currently satisfies both data and "
            "environment requirements."
        )
    if raw_available and not data["preprocessed_frames"]["ready"]:
        explanations.append(
            "Raw data is present, but preprocessing must complete before certification."
        )
    return {
        "status": "READY" if selected_path is not None else "NOT_READY",
        "ready": selected_path is not None,
        "selected_path": selected_path,
        "routes": {
            "reuse_legacy_artifacts": {
                "ready": reuse_ready,
                "environment_profile": "reuse_and_training",
                "requires": [
                    "split CSVs",
                    "legacy Affect cache",
                    "legacy Interaction cache",
                    "legacy behavioral matrices",
                    "legacy behavioral checkpoint",
                ],
            },
            "rebuild_from_preprocessed": {
                "ready": rebuild_ready,
                "environment_profile": "feature_extraction_and_training",
                "requires": ["split CSVs", "preprocessed frames"],
            },
            "raw_data_preparation": {
                "ready": False,
                "available": raw_available,
                "requires_next": "run documented preprocessing to create frames",
            },
        },
        "environment": environment,
        "data": data,
        "explanations": explanations,
    }

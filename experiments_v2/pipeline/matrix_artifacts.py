"""Immutable, content-validated behavioral matrices for a V2 method pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    find_manifest_by_fingerprint,
    fingerprint,
    new_id,
    read_json,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.contracts import FeatureArtifact, PairDefinition
from experiments_v2.pipeline.matrix_builder import (
    build_pair_matrices,
    pair_matrix_contract,
    parse_csv_record,
)


DEFAULT_SMOKE_SAMPLES = (
    "train/low/view642",
    "train/mid/view55",
    "train/high/view2514",
)


@dataclass(frozen=True)
class MatrixArtifact:
    matrix_id: str
    pair_id: str
    fingerprint: str
    directory: Path
    data_dir: Path
    manifest: Mapping[str, Any]
    reused: bool = False


def _records(split_files: Mapping[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        csv_path = split_files[split]
        for line_number, raw in enumerate(
            csv_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line:
                continue
            video_path, label, stem, category = parse_csv_record(
                line, csv_path, line_number
            )
            records.append(
                {
                    "split": split,
                    "category": category,
                    "label": label,
                    "video_path": video_path,
                    "video_name": stem,
                    "sample": f"{split}/{category}/{stem}",
                }
            )
    return records


def smoke_test_pair_matrices(
    *,
    pair: PairDefinition,
    features: Mapping[str, FeatureArtifact],
    split_files: Mapping[str, Path],
    samples: Iterable[str] = DEFAULT_SMOKE_SAMPLES,
) -> dict[str, Any]:
    """Construct selected matrices in memory and prove lossless segment placement."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for matrix smoke validation") from exc

    by_sample = {record["sample"].lower(): record for record in _records(split_files)}
    results = []
    for requested in samples:
        record = by_sample.get(str(requested).lower())
        if record is None:
            raise ValueError(f"Matrix smoke sample is not in the split CSVs: {requested}")
        source_arrays: dict[str, Any] = {}
        source_paths: dict[str, str] = {}
        for entry in pair.feature_layout:
            path = (
                features[entry.category].data_dir
                / record["split"]
                / record["category"]
                / f"{record['video_name']}.npy"
            )
            array = np.load(path, allow_pickle=False)
            expected_shape = (pair.temporal_frames, entry.feature_dim)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Smoke source {path} has shape {array.shape}; expected {expected_shape}"
                )
            if array.dtype != np.float32:
                raise ValueError(
                    f"Smoke source {path} has dtype {array.dtype}; expected float32"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"Smoke source contains NaN/Inf: {path}")
            source_arrays[entry.category] = array
            source_paths[entry.category] = str(path.resolve())

        matrix = np.concatenate(
            [source_arrays[entry.category] for entry in pair.feature_layout], axis=1
        ).astype(np.float32, copy=False)
        segments = {}
        for entry in pair.feature_layout:
            actual = matrix[:, entry.start : entry.end]
            expected = source_arrays[entry.category]
            exact = bool(np.array_equal(actual, expected))
            difference = float(np.max(np.abs(actual - expected)))
            segments[entry.category] = {
                "start": entry.start,
                "end": entry.end,
                "exact_equal": exact,
                "max_absolute_difference": difference,
            }
            if not exact or difference != 0.0:
                raise ValueError(
                    f"Matrix smoke segment mismatch for {requested}/{entry.category}: "
                    f"max absolute difference {difference}"
                )
        result = {
            **record,
            "source_paths": source_paths,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "finite": bool(np.isfinite(matrix).all()),
            "temporal_order": list(range(pair.temporal_frames)),
            "temporal_alignment_preserved": True,
            "segments": segments,
            "passed": matrix.shape == (pair.temporal_frames, pair.matrix_dim),
        }
        if not result["passed"] or result["dtype"] != "float32" or not result["finite"]:
            raise ValueError(f"Matrix smoke contract failed for {requested}: {result}")
        results.append(result)

    return {
        "status": "passed",
        "pair_id": pair.pair_id,
        "sample_count": len(results),
        "samples": results,
        "all_segments_exact": True,
        "maximum_absolute_difference": 0.0,
        "completed_at": utc_now(),
    }


def validate_matrix_artifact(
    *,
    pair: PairDefinition,
    features: Mapping[str, FeatureArtifact],
    split_files: Mapping[str, Path],
    data_dir: Path,
    equivalence_samples: Iterable[str] = DEFAULT_SMOKE_SAMPLES,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate every expected matrix plus exact source-segment equivalence samples."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for matrix validation") from exc

    records = _records(split_files)
    expected_paths: dict[Path, dict[str, Any]] = {}
    duplicate_paths: list[str] = []
    seen_video_paths: set[str] = set()
    for record in records:
        video_identity = str(record["video_path"])
        if video_identity in seen_video_paths:
            duplicate_paths.append(video_identity)
        seen_video_paths.add(video_identity)
        relative = Path(record["split"]) / (
            f"{record['video_name']}_label{record['label']}.npy"
        )
        if relative in expected_paths:
            duplicate_paths.append(relative.as_posix())
        expected_paths[relative] = record

    actual_paths = {
        path.relative_to(data_dir)
        for path in data_dir.glob("*/*.npy")
        if path.is_file()
    }
    missing = sorted(path.as_posix() for path in expected_paths.keys() - actual_paths)
    extras = sorted(path.as_posix() for path in actual_paths - expected_paths.keys())
    malformed: list[dict[str, str]] = []
    shape_mismatches: list[dict[str, Any]] = []
    dtype_mismatches: list[dict[str, str]] = []
    nan_files: list[str] = []
    inf_files: list[str] = []
    label_mismatches: list[dict[str, Any]] = []
    split_mismatches: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}

    actual_by_stem: dict[str, list[Path]] = {}
    for relative in actual_paths:
        stem = relative.stem.rsplit("_label", 1)[0]
        actual_by_stem.setdefault(stem, []).append(relative)

    for relative, record in expected_paths.items():
        path = data_dir / relative
        if not path.is_file():
            alternatives = actual_by_stem.get(str(record["video_name"]), [])
            for alternative in alternatives:
                if alternative.parent.name != record["split"]:
                    split_mismatches.append(
                        {
                            "sample": record["sample"],
                            "expected": record["split"],
                            "actual": alternative.parent.name,
                        }
                    )
                if alternative.stem != relative.stem:
                    label_mismatches.append(
                        {
                            "sample": record["sample"],
                            "expected_label": record["label"],
                            "actual_file": alternative.as_posix(),
                        }
                    )
            continue
        try:
            matrix = np.load(path, allow_pickle=False)
        except Exception as exc:
            malformed.append(
                {"path": relative.as_posix(), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if matrix.shape != (pair.temporal_frames, pair.matrix_dim):
            shape_mismatches.append(
                {
                    "path": relative.as_posix(),
                    "expected": [pair.temporal_frames, pair.matrix_dim],
                    "actual": list(matrix.shape),
                }
            )
        if matrix.dtype != np.float32:
            dtype_mismatches.append(
                {
                    "path": relative.as_posix(),
                    "expected": "float32",
                    "actual": str(matrix.dtype),
                }
            )
        if np.isnan(matrix).any():
            nan_files.append(relative.as_posix())
        if np.isinf(matrix).any():
            inf_files.append(relative.as_posix())
        checksums[relative.as_posix()] = sha256_file(path)
        split_counts[record["split"]] += 1

    requested = {str(sample).lower() for sample in equivalence_samples}
    equivalence = []
    for relative, record in expected_paths.items():
        if record["sample"].lower() not in requested or not (data_dir / relative).is_file():
            continue
        matrix = np.load(data_dir / relative, allow_pickle=False)
        segments = {}
        for entry in pair.feature_layout:
            source_path = (
                features[entry.category].data_dir
                / record["split"]
                / record["category"]
                / f"{record['video_name']}.npy"
            )
            source = np.load(source_path, allow_pickle=False)
            segment = matrix[:, entry.start : entry.end]
            exact = bool(np.array_equal(segment, source))
            maximum = float(np.max(np.abs(segment - source)))
            segments[entry.category] = {
                "exact_equal": exact,
                "max_absolute_difference": maximum,
                "source_path": str(source_path.resolve()),
            }
            if not exact or maximum != 0.0:
                identity_mismatches.append(
                    {
                        "sample": record["sample"],
                        "category": entry.category,
                        "max_absolute_difference": maximum,
                    }
                )
        equivalence.append({"sample": record["sample"], "segments": segments})

    counters = {
        "missing": len(missing),
        "extras": len(extras),
        "duplicates": len(duplicate_paths),
        "malformed": len(malformed),
        "nan": len(nan_files),
        "inf": len(inf_files),
        "identity_mismatches": len(identity_mismatches),
        "label_mismatches": len(label_mismatches),
        "split_mismatches": len(split_mismatches),
        "shape_mismatches": len(shape_mismatches),
        "dtype_mismatches": len(dtype_mismatches),
    }
    expected_split_counts = {
        split: sum(1 for record in records if record["split"] == split)
        for split in ("train", "val", "test")
    }
    passed = (
        not any(counters.values())
        and split_counts == expected_split_counts
        and len(checksums) == len(records)
        and len(equivalence) == len(requested)
    )
    report = {
        "status": "passed" if passed else "failed",
        "pair_id": pair.pair_id,
        "expected_files": len(records),
        "validated_files": len(checksums),
        "expected_split_counts": expected_split_counts,
        "validated_split_counts": split_counts,
        "counters": counters,
        "details": {
            "missing": missing,
            "extras": extras,
            "duplicates": duplicate_paths,
            "malformed": malformed,
            "nan": nan_files,
            "inf": inf_files,
            "identity_mismatches": identity_mismatches,
            "label_mismatches": label_mismatches,
            "split_mismatches": split_mismatches,
            "shape_mismatches": shape_mismatches,
            "dtype_mismatches": dtype_mismatches,
        },
        "equivalence_samples": equivalence,
        "shape_per_video": [pair.temporal_frames, pair.matrix_dim],
        "dtype": "float32",
        "matrix_order": list(pair.matrix_order),
        "completed_at": utc_now(),
    }
    return report, checksums


class MatrixStore:
    """Create or reuse an immutable matrix artifact after mandatory validation gates."""

    def __init__(self, artifacts_root: Path) -> None:
        self.root = artifacts_root / "matrices"
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_or_build(
        self,
        *,
        pair: PairDefinition,
        features: Mapping[str, FeatureArtifact],
        split_files: Mapping[str, Path],
        dataset_identity: Mapping[str, Any],
        git: Mapping[str, Any],
        environment: Mapping[str, Any] | None,
        dataset_name: str,
        preprocessing_provenance: str,
        smoke_samples: Iterable[str] = DEFAULT_SMOKE_SAMPLES,
    ) -> MatrixArtifact:
        samples = tuple(smoke_samples)
        contract = pair_matrix_contract(pair)
        identity = {
            "kind": "behavioral_matrix_artifact",
            "pair_id": pair.pair_id,
            "pair_fingerprint": pair.manifest.get("fingerprint"),
            "source_features": {
                category: {
                    "method_id": feature.method_id,
                    "model_id": feature.model_id,
                    "feature_id": feature.feature_id,
                    "fingerprint": feature.fingerprint,
                }
                for category, feature in sorted(features.items())
            },
            "contract": contract,
            "dataset_fingerprint": dataset_identity["fingerprint"],
            "split_identity": dataset_identity["splits"],
        }
        matrix_fingerprint = fingerprint(identity)
        found = find_manifest_by_fingerprint(self.root, matrix_fingerprint)
        if found is not None:
            manifest_path, manifest = found
            directory = manifest_path.parent
            validation, checksums = validate_matrix_artifact(
                pair=pair,
                features=features,
                split_files=split_files,
                data_dir=directory / "data",
                equivalence_samples=samples,
            )
            if validation["status"] != "passed":
                raise RuntimeError(
                    f"Cached matrix artifact failed validation: {directory}"
                )
            recorded = read_json(directory / str(manifest["checksums_file"]))
            if recorded != checksums:
                raise RuntimeError(
                    f"Cached matrix artifact checksum mismatch: {directory}"
                )
            return self._artifact(directory, manifest, reused=True)

        smoke = smoke_test_pair_matrices(
            pair=pair,
            features=features,
            split_files=split_files,
            samples=samples,
        )
        matrix_id = new_id("MATRIX")
        directory = create_exclusive_dir(self.root / pair.pair_id / matrix_id)
        data_dir = directory / "data"
        write_json_exclusive(directory / "smoke_validation.json", smoke)
        try:
            build_manifest = build_pair_matrices(
                pair=pair,
                features=features,
                split_files=split_files,
                output_dir=data_dir,
            )
            validation, checksums = validate_matrix_artifact(
                pair=pair,
                features=features,
                split_files=split_files,
                data_dir=data_dir,
                equivalence_samples=samples,
            )
            write_json_exclusive(directory / "validation.json", validation)
            write_json_exclusive(directory / "checksums.json", checksums)
            if validation["status"] != "passed":
                raise RuntimeError(
                    f"New matrix artifact failed validation: {directory}"
                )
            compatibility = {
                category: feature.manifest.get("compatibility_fix")
                for category, feature in sorted(features.items())
                if feature.manifest.get("compatibility_fix") is not None
            }
            manifest = {
                "status": "complete",
                "matrix_id": matrix_id,
                "pair_id": pair.pair_id,
                "fingerprint": matrix_fingerprint,
                **contract,
                "source_features": identity["source_features"],
                "compatibility_fixes": compatibility,
                "dataset": {
                    "name": dataset_name,
                    "fingerprint": dataset_identity["fingerprint"],
                    "split_identity": dataset_identity["splits"],
                    "preprocessed_inputs": dataset_identity.get(
                        "preprocessed_inputs"
                    ),
                    "preprocessing": dataset_identity.get("preprocessing"),
                    "preprocessing_provenance": preprocessing_provenance,
                },
                "labels": {"Low": 0, "mid": 1, "high": 2},
                "split_counts": validation["validated_split_counts"],
                "validated_files": validation["validated_files"],
                "build_manifest": build_manifest,
                "smoke_validation_file": "smoke_validation.json",
                "validation_file": "validation.json",
                "checksums_file": "checksums.json",
                "git": dict(git),
                "environment_fingerprint": fingerprint(environment or {}),
                "created_at": utc_now(),
            }
            write_json_exclusive(directory / "manifest.json", manifest)
            return self._artifact(directory, manifest, reused=False)
        except Exception as exc:
            write_json_exclusive(
                directory / "failure.json",
                {
                    "status": "failed",
                    "matrix_id": matrix_id,
                    "pair_id": pair.pair_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": utc_now(),
                },
            )
            raise

    @staticmethod
    def _artifact(
        directory: Path, manifest: Mapping[str, Any], *, reused: bool
    ) -> MatrixArtifact:
        return MatrixArtifact(
            matrix_id=str(manifest["matrix_id"]),
            pair_id=str(manifest["pair_id"]),
            fingerprint=str(manifest["fingerprint"]),
            directory=directory,
            data_dir=directory / "data",
            manifest=manifest,
            reused=reused,
        )

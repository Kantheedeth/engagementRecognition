"""Build run-local Interaction + Affect matrices for a versioned pair."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments_v2.core.artifacts import utc_now, write_json_exclusive
from experiments_v2.core.contracts import FeatureArtifact, PairDefinition


LABEL_BY_CATEGORY = {"low": 0, "mid": 1, "high": 2}


def pair_matrix_contract(pair: PairDefinition) -> dict[str, Any]:
    """Describe the matrix shape and segments without loading numerical data."""

    return {
        "shape_per_video": [pair.temporal_frames, pair.matrix_dim],
        "matrix_order": list(pair.matrix_order),
        "feature_layout": [entry.as_manifest() for entry in pair.feature_layout],
        "segments": {
            entry.category: entry.as_manifest() for entry in pair.feature_layout
        },
    }


def parse_csv_record(line: str, path: Path, line_number: int) -> tuple[str, int, str, str]:
    parts = line.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Malformed record at {path}:{line_number}: {line!r}")
    video_path, label_text = parts
    try:
        label = int(label_text)
    except ValueError as exc:
        raise ValueError(f"Invalid label at {path}:{line_number}: {label_text!r}") from exc
    video = Path(video_path)
    category = video.parent.name.lower()
    expected = LABEL_BY_CATEGORY.get(category)
    if expected is None or label != expected:
        raise ValueError(
            f"Category/label mismatch at {path}:{line_number}: {category!r}/{label}"
        )
    return video_path, label, video.stem, category


def build_pair_matrices(
    *,
    pair: PairDefinition,
    features: Mapping[str, FeatureArtifact],
    split_files: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to build pair feature matrices") from exc

    required_categories = set(pair.matrix_order)
    if set(features) != required_categories:
        raise ValueError(
            f"Pair requires feature categories {sorted(required_categories)}, "
            f"got {sorted(features)}"
        )
    for entry in pair.feature_layout:
        artifact = features[entry.category]
        if (
            artifact.method_id != entry.method_id
            or artifact.model_id != entry.model_id
            or artifact.feature_id != entry.feature_id
            or artifact.feature_dim != entry.feature_dim
        ):
            raise ValueError(
                f"{entry.category} artifact does not match pair layout metadata"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        csv_path = split_files[split]
        lines = [line.strip() for line in csv_path.read_text().splitlines() if line.strip()]
        split_dir = output_dir / split
        split_dir.mkdir()
        seen: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            video_path, label, video_name, category = parse_csv_record(
                line, csv_path, line_number
            )
            if video_name in seen:
                raise ValueError(f"Duplicate video stem in {csv_path}: {video_name}")
            seen.add(video_name)
            arrays = []
            for entry in pair.feature_layout:
                feature_path = (
                    features[entry.category].data_dir
                    / split
                    / category
                    / f"{video_name}.npy"
                )
                if not feature_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {entry.category} feature for {video_path}: "
                        f"{feature_path}"
                    )
                array = np.load(feature_path, allow_pickle=False)
                expected_shape = (pair.temporal_frames, entry.feature_dim)
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{feature_path} has shape {array.shape}; expected {expected_shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite feature found in {feature_path}")
                arrays.append(array)
            matrix = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
            if matrix.shape != (pair.temporal_frames, pair.matrix_dim):
                raise AssertionError(f"Internal pair matrix shape error: {matrix.shape}")
            destination = split_dir / f"{video_name}_label{label}.npy"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite matrix: {destination}")
            np.save(destination, matrix)
        split_counts[split] = len(lines)

    manifest = {
        "status": "complete",
        "format_version": 1,
        "feature_schema": "experiments_v2_affect_interaction_pair_v1",
        "pair_id": pair.pair_id,
        **pair_matrix_contract(pair),
        "split_counts": split_counts,
        "created_at": utc_now(),
    }
    write_json_exclusive(output_dir / "build_manifest.json", manifest)
    return manifest

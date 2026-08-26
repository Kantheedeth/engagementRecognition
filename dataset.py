"""Dataset loading with feature-schema and shape validation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_schema import AFFECT_FEATURE_SCHEMA


LEGACY_AFFECT_TRACKER_DEFAULTS = {
    "new_track_threshold": 0.45,
    "track_buffer": 8,
}


def normalize_feature_manifest_for_comparison(manifest: dict) -> dict:
    """Fill only recorded legacy defaults before provenance comparison."""
    normalized = deepcopy(manifest)
    affect = normalized.get("affect_extraction")
    if not isinstance(affect, dict):
        return normalized
    affect_schema = affect.get("feature_schema") or normalized.get("affect_schema")
    tracker = affect.get("tracker")
    if (
        affect_schema == AFFECT_FEATURE_SCHEMA
        and isinstance(tracker, dict)
        and tracker.get("enabled") is True
        and tracker.get("library") == "ultralytics-bytetrack"
    ):
        for key, value in LEGACY_AFFECT_TRACKER_DEFAULTS.items():
            tracker.setdefault(key, value)
    return normalized


def feature_manifests_compatible(checkpoint_manifest: dict, data_manifest: dict) -> bool:
    """Compare provenance while supporting the original track-aware manifest."""
    if not isinstance(checkpoint_manifest, dict) or not isinstance(data_manifest, dict):
        return False
    return normalize_feature_manifest_for_comparison(
        checkpoint_manifest
    ) == normalize_feature_manifest_for_comparison(data_manifest)


def load_feature_manifest(
    data_dir: str | Path,
    *,
    expected_schema: str,
    expected_shape: Sequence[int],
) -> dict:
    data_dir = Path(data_dir).expanduser().resolve()
    manifest_path = data_dir / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Feature manifest not found: {manifest_path}. Rebuild matrices with "
            "the track-aware affect pipeline; legacy matrices are not accepted."
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("feature_schema") != expected_schema:
        raise ValueError(
            f"Unsupported feature schema {manifest.get('feature_schema')!r}; "
            f"expected {expected_schema!r}"
        )
    expected_shape = list(expected_shape)
    if manifest.get("shape_per_video") != expected_shape:
        raise ValueError(
            f"Manifest shape {manifest.get('shape_per_video')} does not match "
            f"expected shape {expected_shape}"
        )
    return manifest


class EngagementDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        *,
        expected_shape: Sequence[int] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.file_paths = sorted(self.split_dir.glob("*.npy"))
        self.expected_shape = tuple(expected_shape) if expected_shape is not None else None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[index]
        label_text = path.stem.rsplit("_label", maxsplit=1)
        if len(label_text) != 2:
            raise ValueError(f"Cannot parse class label from {path.name}")
        try:
            label = int(label_text[1])
        except ValueError as exc:
            raise ValueError(f"Cannot parse class label from {path.name}") from exc
        if label not in (0, 1, 2):
            raise ValueError(f"Class label must be 0, 1, or 2 in {path.name}")

        matrix = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if self.expected_shape is not None and matrix.shape != self.expected_shape:
            raise ValueError(
                f"{path} has shape {matrix.shape}; expected {self.expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite feature value found in {path}")
        return torch.from_numpy(matrix), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "feature_matrices"
    dataset = EngagementDataset(root / "train", expected_shape=(8, 616))
    print(f"Dataset loaded: {len(dataset)} items")
    if len(dataset):
        features, target = dataset[0]
        print(f"Sample tensor shape: {features.shape}, label: {target}")

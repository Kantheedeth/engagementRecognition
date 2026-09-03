"""Automatic Cartesian Affect x Interaction pair generation."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    find_manifest_by_fingerprint,
    fingerprint,
    new_id,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.contracts import (
    FeatureArtifact,
    FeatureLayoutEntry,
    ModelArtifact,
    PairDefinition,
)


ResolvedMethod = tuple[ModelArtifact, FeatureArtifact]


class PairStore:
    def __init__(
        self,
        artifacts_root: Path,
        *,
        feature_order: Iterable[str],
        temporal_frames: int,
    ) -> None:
        self.root = artifacts_root / "pairs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.feature_order = tuple(feature_order)
        if len(self.feature_order) != 2 or set(self.feature_order) != {
            "affect",
            "interaction",
        }:
            raise ValueError(
                "feature_order must contain interaction and affect exactly once"
            )
        self.temporal_frames = int(temporal_frames)
        if self.temporal_frames <= 0:
            raise ValueError("temporal_frames must be positive")

    def generate_cartesian(
        self,
        *,
        affect: Iterable[ResolvedMethod],
        interaction: Iterable[ResolvedMethod],
        git_commit: str | None,
    ) -> list[PairDefinition]:
        pairs = []
        for affect_resolved, interaction_resolved in product(affect, interaction):
            pairs.append(
                self.resolve_or_create(
                    affect=affect_resolved,
                    interaction=interaction_resolved,
                    git_commit=git_commit,
                )
            )
        if not pairs:
            raise ValueError("Cartesian pairing produced no Affect x Interaction pairs")
        return pairs

    def resolve_or_create(
        self,
        *,
        affect: ResolvedMethod,
        interaction: ResolvedMethod,
        git_commit: str | None,
    ) -> PairDefinition:
        affect_model, affect_feature = affect
        interaction_model, interaction_feature = interaction
        resolved = {
            "affect": (affect_model, affect_feature),
            "interaction": (interaction_model, interaction_feature),
        }
        for category, (model, feature) in resolved.items():
            if feature.category != category or model.category != category:
                raise ValueError(f"Resolved {category} artifact has the wrong category")
            if feature.model_id != model.model_id:
                raise ValueError(f"Resolved {category} feature/model IDs do not match")

        feature_layout = []
        offset = 0
        for category in self.feature_order:
            model, feature = resolved[category]
            entry = FeatureLayoutEntry(
                category=category,
                method_id=feature.method_id,
                model_id=model.model_id,
                feature_id=feature.feature_id,
                feature_dim=feature.feature_dim,
                start=offset,
                end=offset + feature.feature_dim,
            )
            feature_layout.append(entry)
            offset = entry.end
        identity = {
            "affect": {
                "method_id": affect_feature.method_id,
                "model_id": affect_model.model_id,
                "feature_id": affect_feature.feature_id,
                "feature_fingerprint": affect_feature.fingerprint,
                "feature_dim": affect_feature.feature_dim,
            },
            "interaction": {
                "method_id": interaction_feature.method_id,
                "model_id": interaction_model.model_id,
                "feature_id": interaction_feature.feature_id,
                "feature_fingerprint": interaction_feature.fingerprint,
                "feature_dim": interaction_feature.feature_dim,
            },
            "feature_layout": [entry.as_manifest() for entry in feature_layout],
            "temporal_frames": self.temporal_frames,
        }
        pair_fingerprint = fingerprint(identity)
        found = find_manifest_by_fingerprint(self.root, pair_fingerprint)
        if found is not None:
            path, manifest = found
            return self._definition(path.parent, manifest)

        pair_id = new_id("PAIR")
        directory = create_exclusive_dir(self.root / pair_id)
        manifest = {
            "status": "complete",
            "pair_id": pair_id,
            "fingerprint": pair_fingerprint,
            **identity,
            "matrix_order": list(self.feature_order),
            "matrix_dim": offset,
            "created_at": utc_now(),
            "git_commit": git_commit,
        }
        write_json_exclusive(directory / "manifest.json", manifest)
        return self._definition(directory, manifest)

    @staticmethod
    def _definition(directory: Path, manifest: Mapping[str, Any]) -> PairDefinition:
        raw_layout = manifest.get("feature_layout")
        if not isinstance(raw_layout, list) or not raw_layout:
            raise ValueError(f"Pair manifest has no feature_layout: {directory}")
        feature_layout = tuple(
            FeatureLayoutEntry(
                category=str(entry["category"]),
                method_id=str(entry["method_id"]),
                model_id=str(entry["model_id"]),
                feature_id=str(entry["feature_id"]),
                feature_dim=int(entry["feature_dim"]),
                start=int(entry["start"]),
                end=int(entry["end"]),
            )
            for entry in raw_layout
        )
        return PairDefinition(
            pair_id=str(manifest["pair_id"]),
            feature_layout=feature_layout,
            temporal_frames=int(manifest["temporal_frames"]),
            directory=directory,
            manifest=manifest,
        )

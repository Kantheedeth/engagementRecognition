"""Immutable registry for pretrained bundles and trained engagement checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    file_size_mb,
    find_manifest_by_fingerprint,
    fingerprint,
    new_id,
    read_json,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.contracts import MethodSpec, ModelArtifact


class ModelRegistry:
    def __init__(self, artifacts_root: Path) -> None:
        self.root = artifacts_root / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_pretrained(
        self,
        *,
        spec: MethodSpec,
        identity: Mapping[str, Any],
        force_train: bool,
        git_commit: str | None,
    ) -> ModelArtifact:
        if force_train:
            raise ValueError(
                f"{spec.code}/{spec.name} is a pretrained legacy method; "
                "force_train is not applicable"
            )
        request = {
            "kind": "pretrained_method_bundle",
            "method_id": spec.method_id,
            "method_version": spec.version,
            "category": spec.category,
            "identity": dict(identity),
        }
        model_fingerprint = fingerprint(request)
        found = find_manifest_by_fingerprint(self.root, model_fingerprint)
        if found is not None:
            path, manifest = found
            return self._artifact(path.parent, manifest)

        model_id = new_id("MODEL")
        directory = create_exclusive_dir(self.root / model_id)
        component_sizes = [
            component.get("size_mb")
            for component in identity.get("components", [])
            if isinstance(component, dict) and component.get("size_mb") is not None
        ]
        all_components_sized = len(component_sizes) == len(identity.get("components", []))
        manifest = {
            "status": "complete",
            "model_id": model_id,
            "method_id": spec.method_id,
            "method_code": spec.code,
            "category": spec.category,
            "architecture": identity.get("architecture"),
            "pretrained": True,
            "training_required": False,
            "identity": dict(identity),
            "fingerprint": model_fingerprint,
            "parameter_count": None,
            "known_component_size_mb": sum(component_sizes),
            "checkpoint_size_mb": sum(component_sizes) if all_components_sized else None,
            "created_at": utc_now(),
            "git_commit": git_commit,
        }
        write_json_exclusive(directory / "manifest.json", manifest)
        return self._artifact(directory, manifest)

    def register_engagement_checkpoint(
        self,
        *,
        model_id: str,
        pair_id: str,
        run_id: str,
        checkpoint_path: Path,
        model_config: Mapping[str, Any],
        parameter_count: int,
        validation_metric: Mapping[str, Any],
        git_commit: str | None,
    ) -> ModelArtifact:
        if not model_id.startswith("MODEL_"):
            raise ValueError("Engagement model IDs must start with MODEL_")
        directory = create_exclusive_dir(self.root / model_id)
        identity = {
            "kind": "engagement_checkpoint",
            "pair_id": pair_id,
            "run_id": run_id,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_config": dict(model_config),
        }
        model_fingerprint = fingerprint(identity)
        manifest = {
            "status": "complete",
            "model_id": model_id,
            "method_id": "METHOD_ENGAGEMENT",
            "category": "engagement",
            "architecture": "legacy_pure_behavioral_attention",
            "pretrained": False,
            "training_required": True,
            "pair_id": pair_id,
            "run_id": run_id,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_size_mb": file_size_mb(checkpoint_path),
            "parameter_count": parameter_count,
            "validation_metric": dict(validation_metric),
            "identity": identity,
            "fingerprint": model_fingerprint,
            "created_at": utc_now(),
            "git_commit": git_commit,
        }
        write_json_exclusive(directory / "manifest.json", manifest)
        return self._artifact(directory, manifest)

    def get(self, model_id: str) -> ModelArtifact:
        directory = self.root / model_id
        return self._artifact(directory, read_json(directory / "manifest.json"))

    @staticmethod
    def _artifact(directory: Path, manifest: Mapping[str, Any]) -> ModelArtifact:
        return ModelArtifact(
            model_id=str(manifest["model_id"]),
            method_id=str(manifest["method_id"]),
            category=str(manifest["category"]),
            fingerprint=str(manifest["fingerprint"]),
            directory=directory,
            manifest=manifest,
        )

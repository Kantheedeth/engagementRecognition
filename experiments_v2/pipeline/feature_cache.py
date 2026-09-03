"""Immutable per-method feature extraction and cache reuse."""

from __future__ import annotations

import traceback
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    find_manifest_by_fingerprint,
    fingerprint,
    new_id,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.contracts import FeatureArtifact, MethodAdapter, ModelArtifact


class FeatureCache:
    def __init__(self, artifacts_root: Path, project_root: Path) -> None:
        self.root = artifacts_root / "features"
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root

    def resolve_or_extract(
        self,
        *,
        adapter: MethodAdapter,
        model: ModelArtifact,
        dataset_identity: Mapping[str, Any],
        input_dir: Path,
        parameters: Mapping[str, Any],
        force_extract: bool,
        legacy_feature_dir: Path | None,
        git_commit: str | None,
    ) -> FeatureArtifact:
        spec = adapter.spec
        request = {
            "method_id": spec.method_id,
            "method_version": spec.version,
            "feature_schema": spec.feature_schema,
            "feature_dim": spec.feature_dim,
            "model_id": model.model_id,
            "model_fingerprint": model.fingerprint,
            "dataset_fingerprint": dataset_identity["fingerprint"],
            "parameters": dict(parameters),
        }
        cache_fingerprint = fingerprint(request)
        preprocessing = dataset_identity.get("preprocessing")
        if not isinstance(preprocessing, Mapping) or "num_frames" not in preprocessing:
            raise ValueError("dataset_identity must record preprocessing.num_frames")
        temporal_frames = int(preprocessing["num_frames"])
        method_root = self.root / spec.category / spec.method_id / model.model_id
        if not force_extract:
            found = find_manifest_by_fingerprint(method_root, cache_fingerprint)
            if found is not None:
                path, manifest = found
                data_dir = path.parent / "data"
                if data_dir.is_dir():
                    try:
                        validation = adapter.validate_features(data_dir)
                    except (OSError, RuntimeError, ValueError):
                        validation = None
                    if validation is not None and validation.get("validated_files") == manifest.get(
                        "validated_files"
                    ):
                        return self._artifact(path.parent, manifest, reused=True)

        legacy_validation = None
        if not force_extract and legacy_feature_dir is not None and legacy_feature_dir.is_dir():
            try:
                legacy_validation = adapter.validate_features(legacy_feature_dir)
            except (OSError, RuntimeError, ValueError):
                legacy_validation = None
        if legacy_validation is None and not input_dir.is_dir():
            raise FileNotFoundError(
                f"No reusable {spec.code} feature cache was found and legacy "
                f"preprocessed inputs are unavailable: {input_dir}"
            )

        feature_id = new_id("FEATURE")
        directory = create_exclusive_dir(method_root / feature_id)
        data_dir = directory / "data"
        data_dir.mkdir()
        write_json_exclusive(
            directory / "request.json",
            {
                "feature_id": feature_id,
                "fingerprint": cache_fingerprint,
                "request": request,
                "created_at": utc_now(),
            },
        )

        started = perf_counter()
        try:
            if legacy_validation is not None:
                shutil.copytree(legacy_feature_dir, data_dir, dirs_exist_ok=True)
                copied_validation = adapter.validate_features(data_dir)
                extraction = {
                    **copied_validation,
                    "legacy_manifest": copied_validation.get("legacy_manifest"),
                    "command": None,
                }
                extraction_seconds = None
                adoption_seconds = perf_counter() - started
                adopted_from_legacy = str(legacy_feature_dir.resolve())
            else:
                extraction = adapter.extract_features(
                    project_root=self.project_root,
                    input_dir=input_dir,
                    output_dir=data_dir,
                    parameters=parameters,
                    log_path=directory / "extraction.log",
                )
                extraction_seconds = perf_counter() - started
                adoption_seconds = None
                adopted_from_legacy = None
            manifest = {
                "status": "complete",
                "feature_id": feature_id,
                "method_id": spec.method_id,
                "method_code": spec.code,
                "method_name": spec.name,
                "method_version": spec.version,
                "model_id": model.model_id,
                "category": spec.category,
                "feature_schema": spec.feature_schema,
                "shape_per_video": [temporal_frames, spec.feature_dim],
                "feature_dim": spec.feature_dim,
                "fingerprint": cache_fingerprint,
                "dataset_identity": dict(dataset_identity),
                "parameters": dict(parameters),
                "extraction_seconds": extraction_seconds,
                "adoption_seconds": adoption_seconds,
                "adopted_from_legacy": adopted_from_legacy,
                "validated_files": extraction.get("validated_files"),
                "legacy_manifest": extraction.get("legacy_manifest"),
                "command": extraction.get("command"),
                "created_at": utc_now(),
                "git_commit": git_commit,
            }
            write_json_exclusive(directory / "manifest.json", manifest)
            return self._artifact(
                directory, manifest, reused=adopted_from_legacy is not None
            )
        except Exception as exc:
            write_json_exclusive(
                directory / "failure.json",
                {
                    "status": "failed",
                    "feature_id": feature_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": utc_now(),
                },
            )
            raise

    @staticmethod
    def _artifact(
        directory: Path, manifest: Mapping[str, Any], *, reused: bool
    ) -> FeatureArtifact:
        return FeatureArtifact(
            feature_id=str(manifest["feature_id"]),
            method_id=str(manifest["method_id"]),
            model_id=str(manifest["model_id"]),
            category=str(manifest["category"]),
            fingerprint=str(manifest["fingerprint"]),
            directory=directory,
            data_dir=directory / "data",
            feature_dim=int(manifest["feature_dim"]),
            manifest=manifest,
            reused=reused,
        )

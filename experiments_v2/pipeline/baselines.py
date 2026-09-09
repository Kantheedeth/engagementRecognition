"""Immutable publication and discovery of the official A1 + I1 baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    fingerprint,
    new_id,
    read_json,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.pipeline.comparison import comparison_values


class BaselineStore:
    def __init__(self, artifacts_root: Path) -> None:
        self.root = artifacts_root / "baselines"
        self.root.mkdir(parents=True, exist_ok=True)

    def official(self) -> dict[str, Any] | None:
        records = []
        for path in sorted(self.root.glob("BASELINE_*/baseline.json")):
            record = read_json(path)
            if record.get("status") == "complete" and record.get("official") is True:
                records.append(record)
        if len(records) > 1:
            raise RuntimeError("Multiple official V2 baselines were found")
        return records[0] if records else None

    def publish_official(
        self,
        *,
        metrics: Mapping[str, Any],
        metrics_path: Path,
        affect_code: str,
        interaction_code: str,
    ) -> dict[str, Any]:
        if self.official() is not None:
            raise FileExistsError("The official V2 baseline already exists")
        if metrics.get("status") != "complete":
            raise ValueError("Only a complete run can become the official baseline")
        environment = metrics.get("environment")
        if (
            not isinstance(environment, Mapping)
            or environment.get("certification_ready") is not True
        ):
            raise ValueError(
                "Official publication requires a successful certification preflight"
            )

        identity = metrics.get("identity", {})
        performance = metrics.get("performance", {})
        model_cost = metrics.get("model_cost", {})
        matrix_manifest = metrics.get("matrix_manifest", {})
        training = metrics.get("training", {})
        if affect_code != "A1" or identity.get("affect_method_id") != "METHOD_A1":
            raise ValueError("The official baseline Affect method must be A1/METHOD_A1")
        if (
            interaction_code != "I1"
            or identity.get("interaction_method_id") != "METHOD_I1"
        ):
            raise ValueError(
                "The official baseline Interaction method must be I1/METHOD_I1"
            )
        if matrix_manifest.get("matrix_order") != ["interaction", "affect"]:
            raise ValueError("The official baseline matrix order must be interaction, affect")
        if matrix_manifest.get("shape_per_video") != [8, 40]:
            raise ValueError("The official A1 + I1 matrix shape must be [8, 40]")

        baseline_id = new_id("BASELINE")
        directory = create_exclusive_dir(self.root / baseline_id)
        record = {
            "status": "complete",
            "official": True,
            "baseline_id": baseline_id,
            "pair_id": metrics["pair_id"],
            "run_id": metrics["run_id"],
            "affect": {
                "code": affect_code,
                "method_id": identity["affect_method_id"],
                "model_id": identity["affect_model_id"],
                "feature_id": identity["affect_feature_id"],
                "compatibility_fix": matrix_manifest.get(
                    "compatibility_fixes", {}
                ).get("affect"),
            },
            "interaction": {
                "code": interaction_code,
                "method_id": identity["interaction_method_id"],
                "model_id": identity["interaction_model_id"],
                "feature_id": identity["interaction_feature_id"],
            },
            "engagement": {
                "model_id": identity["engagement_model_id"],
                "checkpoint_id": identity.get(
                    "engagement_checkpoint_id", identity["engagement_model_id"]
                ),
            },
            "dataset": {
                "name": matrix_manifest.get("dataset", {}).get("name"),
                "fingerprint": identity["dataset_fingerprint"],
                "splits": matrix_manifest.get("split_counts"),
                "total": sum((matrix_manifest.get("split_counts") or {}).values()),
                "split_identity": identity.get("split_identity"),
                "seed": identity["random_seed"],
                "preprocessing_provenance": matrix_manifest.get("dataset", {}).get(
                    "preprocessing_provenance"
                ),
            },
            "matrix": {
                "matrix_id": identity.get("matrix_id"),
                "shape_per_video": matrix_manifest.get("shape_per_video"),
                "temporal_dim": (matrix_manifest.get("shape_per_video") or [None])[0],
                "total_feature_dim": (
                    matrix_manifest.get("shape_per_video") or [None, None]
                )[1],
                "feature_order": matrix_manifest.get("matrix_order"),
                "feature_layout": matrix_manifest.get("feature_layout"),
                "checksums_file": matrix_manifest.get("checksums_file"),
                "validation_file": matrix_manifest.get("validation_file"),
            },
            "training": {
                "configuration": training.get("training_config"),
                "model_configuration": training.get("model_config"),
                "device": training.get("device"),
                "best_epoch": training.get("best_epoch"),
                "checkpoint_selection": training.get("checkpoint_selection"),
            },
            "metrics": {
                "accuracy": performance.get("accuracy"),
                "precision": performance.get("precision_macro"),
                "recall": performance.get("recall_macro"),
                "f1": performance.get("f1_macro"),
                "confusion_matrix": performance.get("confusion_matrix"),
            },
            "documented_legacy_reference": metrics.get(
                "documented_legacy_reference"
            ),
            "documented_reference_differences": metrics.get(
                "documented_reference_differences"
            ),
            "efficiency": {
                "parameter_count": model_cost.get("engagement_parameter_count"),
                "checkpoint_size_mb": model_cost.get(
                    "engagement_checkpoint_size_mb"
                ),
                "training_seconds": metrics.get("training", {}).get(
                    "training_seconds"
                ),
                "inference_seconds": performance.get("inference_seconds"),
                "inference_ms_per_video": performance.get(
                    "inference_ms_per_video"
                ),
                "fps": performance.get("engagement_fps"),
                "feature_method_footprint": model_cost.get(
                    "feature_method_footprint"
                ),
                "combined_feature_method_size_mb": model_cost.get(
                    "combined_feature_method_size_mb"
                ),
                "extraction_seconds": metrics.get("extraction", {})
                .get("feature_generation", {})
                .get("total_seconds"),
            },
            "comparison_values": comparison_values(metrics),
            "environment": dict(environment),
            "environment_fingerprint": fingerprint(environment),
            "git": identity.get("git"),
            "provenance": {
                "preprocessing_source": matrix_manifest.get("dataset", {}).get(
                    "preprocessing_provenance"
                ),
                "source_affect_feature_id": identity["affect_feature_id"],
                "source_interaction_feature_id": identity[
                    "interaction_feature_id"
                ],
                "test_set_used_for_model_selection": training.get(
                    "checkpoint_selection", {}
                ).get("test_metrics_used"),
            },
            "source_metrics_path": str(metrics_path.resolve()),
            "created_at": utc_now(),
        }
        write_json_exclusive(directory / "baseline.json", record)
        return record

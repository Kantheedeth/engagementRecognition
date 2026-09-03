"""End-to-end orchestration for immutable Affect x Interaction runs."""

from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path
from typing import Any, Mapping

from experiments_v2.core.artifacts import (
    dataset_identity,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.config import load_config
from experiments_v2.core.contracts import FeatureArtifact, ModelArtifact
from experiments_v2.pipeline.baselines import BaselineStore
from experiments_v2.pipeline.comparison import baseline_deltas, comparison_values
from experiments_v2.pipeline.engagement import evaluate_engagement, train_engagement
from experiments_v2.pipeline.feature_cache import FeatureCache
from experiments_v2.pipeline.matrix_builder import build_pair_matrices
from experiments_v2.pipeline.pairs import PairStore
from experiments_v2.pipeline.runs import RunStore
from experiments_v2.registry.builtins import create_builtin_registry
from experiments_v2.registry.model_registry import ModelRegistry


class ExperimentRunner:
    def __init__(self, *, project_root: Path, config_path: Path) -> None:
        self.project_root = project_root.resolve()
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path, self.project_root)
        self.registry = create_builtin_registry()
        self.git = _git_identity(self.project_root)

    def plan(self) -> dict[str, Any]:
        dataset = self.config["dataset"]
        split_paths = {
            split: Path(path) for split, path in dataset["split_files"].items()
        }
        input_dir = Path(dataset["preprocessed_input_dir"])
        selected = {
            category: [
                {
                    "code": entry["code"],
                    "method_id": self.registry.spec(entry["code"]).method_id,
                    "name": self.registry.spec(entry["code"]).name,
                }
                for entry in self.config["methods"][category]
                if entry.get("enabled", True)
            ]
            for category in ("affect", "interaction")
        }
        return {
            "experiment": self.config["experiment"]["name"],
            "branch": self.git.get("branch"),
            "git_commit": self.git.get("commit"),
            "git_dirty": self.git.get("dirty"),
            "methods": selected,
            "pair_count": len(selected["affect"]) * len(selected["interaction"]),
            "dataset": {
                "preprocessed_input_dir": str(input_dir),
                "preprocessed_input_exists": input_dir.is_dir(),
                "splits": {
                    split: {"path": str(path), "exists": path.is_file()}
                    for split, path in split_paths.items()
                },
            },
            "artifacts_root": self.config["experiment"]["artifacts_root"],
        }

    def run(
        self,
        *,
        environment_info: Mapping[str, Any] | None = None,
        certify_official: bool = False,
    ) -> list[dict[str, Any]]:
        plan = self.plan()
        missing = [
            details["path"]
            for details in plan["dataset"]["splits"].values()
            if not details["exists"]
        ]
        artifacts_root = Path(self.config["experiment"]["artifacts_root"])
        enabled_entries = [
            entry
            for category in ("affect", "interaction")
            for entry in self.config["methods"][category]
            if entry.get("enabled", True)
        ]
        legacy_caches_exist = bool(enabled_entries) and all(
            entry.get("legacy_feature_dir")
            and Path(entry["legacy_feature_dir"]).is_dir()
            for entry in enabled_entries
        )
        v2_cache_exists = (artifacts_root / "features").is_dir()
        if (
            not plan["dataset"]["preprocessed_input_exists"]
            and not legacy_caches_exist
            and not v2_cache_exists
        ):
            missing.append(plan["dataset"]["preprocessed_input_dir"])
        if missing:
            raise FileNotFoundError(
                "Baseline inputs are unavailable; restore the existing project data first: "
                + ", ".join(missing)
            )

        artifacts_root.mkdir(parents=True, exist_ok=True)
        dataset_config = self.config["dataset"]
        model_registry = ModelRegistry(artifacts_root)
        feature_cache = FeatureCache(artifacts_root, self.project_root)
        pair_store = PairStore(
            artifacts_root,
            feature_order=self.config["pairing"]["feature_order"],
            temporal_frames=int(dataset_config["preprocessing"]["num_frames"]),
        )
        run_store = RunStore(artifacts_root)
        baseline_store = BaselineStore(artifacts_root)
        official_baseline = baseline_store.official()
        split_files = {
            split: Path(path) for split, path in dataset_config["split_files"].items()
        }
        input_dir = Path(dataset_config["preprocessed_input_dir"])
        data_identity = dataset_identity(
            split_files,
            input_dir,
            dataset_config["preprocessing"],
        )

        resolved: dict[str, list[tuple[ModelArtifact, FeatureArtifact]]] = {
            "affect": [],
            "interaction": [],
        }
        for category in ("affect", "interaction"):
            for entry in self.config["methods"][category]:
                if not entry.get("enabled", True):
                    continue
                adapter = self.registry.create(entry["code"])
                if adapter.spec.category != category:
                    raise ValueError(
                        f"Method {entry['code']} is registered as {adapter.spec.category}, "
                        f"not {category}"
                    )
                model = model_registry.resolve_pretrained(
                    spec=adapter.spec,
                    identity=adapter.model_identity(entry["parameters"]),
                    force_train=bool(entry["force_train"]),
                    git_commit=self.git.get("commit"),
                )
                features = feature_cache.resolve_or_extract(
                    adapter=adapter,
                    model=model,
                    dataset_identity=data_identity,
                    input_dir=input_dir,
                    parameters=entry["parameters"],
                    force_extract=bool(entry["force_extract"]),
                    legacy_feature_dir=Path(entry["legacy_feature_dir"])
                    if entry.get("legacy_feature_dir")
                    else None,
                    git_commit=self.git.get("commit"),
                )
                resolved[category].append((model, features))

        pairs = pair_store.generate_cartesian(
            affect=resolved["affect"],
            interaction=resolved["interaction"],
            git_commit=self.git.get("commit"),
        )
        features_by_id = {
            feature.feature_id: (model, feature)
            for category in resolved.values()
            for model, feature in category
        }
        results = []
        for pair in pairs:
            affect_model, affect_feature = features_by_id[pair.affect_feature_id]
            interaction_model, interaction_feature = features_by_id[
                pair.interaction_feature_id
            ]
            results.append(
                self._run_pair(
                    artifacts_root=artifacts_root,
                    model_registry=model_registry,
                    pair=pair,
                    affect_model=affect_model,
                    affect_feature=affect_feature,
                    interaction_model=interaction_model,
                    interaction_feature=interaction_feature,
                    split_files=split_files,
                    dataset_identity=data_identity,
                    run_store=run_store,
                    baseline_store=baseline_store,
                    official_baseline=official_baseline,
                    environment_info=environment_info,
                    certify_official=certify_official,
                )
            )
        return results

    def _run_pair(
        self,
        *,
        artifacts_root: Path,
        model_registry: ModelRegistry,
        pair: Any,
        affect_model: ModelArtifact,
        affect_feature: FeatureArtifact,
        interaction_model: ModelArtifact,
        interaction_feature: FeatureArtifact,
        split_files: Mapping[str, Path],
        dataset_identity: Mapping[str, Any],
        run_store: RunStore,
        baseline_store: BaselineStore,
        official_baseline: Mapping[str, Any] | None,
        environment_info: Mapping[str, Any] | None,
        certify_official: bool,
    ) -> dict[str, Any]:
        run_id, run_dir = run_store.create()
        original_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        write_json_exclusive(run_dir / "config.original.json", original_config)
        resolved_config = {
            **{key: value for key, value in self.config.items() if not key.startswith("_")},
            "resolved": {
                "run_id": run_id,
                "pair_id": pair.pair_id,
                "affect_model_id": affect_model.model_id,
                "affect_feature_id": affect_feature.feature_id,
                "interaction_model_id": interaction_model.model_id,
                "interaction_feature_id": interaction_feature.feature_id,
                "dataset_fingerprint": dataset_identity["fingerprint"],
                "git": self.git,
            },
        }
        write_json_exclusive(run_dir / "config.resolved.json", resolved_config)
        try:
            matrix_dir = run_dir / "feature_matrices"
            matrix_manifest = build_pair_matrices(
                pair=pair,
                features={
                    "affect": affect_feature,
                    "interaction": interaction_feature,
                },
                split_files=split_files,
                output_dir=matrix_dir,
            )
            engagement = self.config["engagement"]
            engagement_model, training = train_engagement(
                run_id=run_id,
                run_dir=run_dir,
                pair=pair,
                matrix_dir=matrix_dir,
                pair_manifest=pair.manifest,
                model_config=engagement["model"],
                training_config=engagement["training"],
                model_registry=model_registry,
                git_commit=self.git.get("commit"),
            )
            evaluation = evaluate_engagement(
                run_dir=run_dir,
                pair=pair,
                matrix_dir=matrix_dir,
                engagement_model=engagement_model,
                evaluation_config=engagement["evaluation"],
            )
            metrics = _compose_metrics(
                run_id=run_id,
                pair=pair,
                affect_model=affect_model,
                affect_feature=affect_feature,
                interaction_model=interaction_model,
                interaction_feature=interaction_feature,
                engagement_model=engagement_model,
                training=training,
                evaluation=evaluation,
                matrix_manifest=matrix_manifest,
                dataset_identity=dataset_identity,
                git=self.git,
                official_baseline=official_baseline,
                environment_info=environment_info,
                documented_reference=self.config["certification"][
                    "documented_legacy_reference"
                ],
            )
            metrics_path = run_dir / "metrics.json"
            write_json_exclusive(metrics_path, metrics)
            baseline_record = official_baseline
            baseline_config = self.config["baseline"]
            if (
                certify_official
                and baseline_config["publish_official"]
                and baseline_record is None
            ):
                baseline_record = baseline_store.publish_official(
                    metrics=metrics,
                    metrics_path=metrics_path,
                    affect_code=str(baseline_config["affect_code"]),
                    interaction_code=str(baseline_config["interaction_code"]),
                )
            baseline_id = (
                str(baseline_record["baseline_id"])
                if baseline_record is not None
                else None
            )
            if baseline_id is not None:
                write_json_exclusive(
                    run_dir / "baseline_reference.json",
                    {
                        "baseline_id": baseline_id,
                        "is_official_baseline_run": baseline_record.get("run_id")
                        == run_id,
                    },
                )
            write_json_exclusive(
                run_dir / "manifest.json",
                {
                    "status": "complete",
                    "run_id": run_id,
                    "pair_id": pair.pair_id,
                    "engagement_model_id": engagement_model.model_id,
                    "baseline_id": baseline_id,
                    "metrics_path": str(metrics_path.resolve()),
                    "completed_at": utc_now(),
                },
            )
            return metrics
        except Exception as exc:
            write_json_exclusive(
                run_dir / "failure.json",
                {
                    "status": "failed",
                    "run_id": run_id,
                    "pair_id": pair.pair_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": utc_now(),
                },
            )
            raise


def _compose_metrics(
    *,
    run_id: str,
    pair: Any,
    affect_model: ModelArtifact,
    affect_feature: FeatureArtifact,
    interaction_model: ModelArtifact,
    interaction_feature: FeatureArtifact,
    engagement_model: ModelArtifact,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    matrix_manifest: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    git: Mapping[str, Any],
    official_baseline: Mapping[str, Any] | None,
    environment_info: Mapping[str, Any] | None,
    documented_reference: Mapping[str, Any],
) -> dict[str, Any]:
    affect_size = affect_model.manifest.get("checkpoint_size_mb")
    interaction_size = interaction_model.manifest.get("checkpoint_size_mb")
    engagement_size = engagement_model.manifest.get("checkpoint_size_mb")
    known_method_sizes = [
        float(value) for value in (affect_size, interaction_size) if value is not None
    ]
    known_method_size = sum(known_method_sizes) if known_method_sizes else None
    full_method_size_known = affect_size is not None and interaction_size is not None
    extraction_original = {
        "affect_seconds": affect_feature.manifest.get("extraction_seconds"),
        "interaction_seconds": interaction_feature.manifest.get("extraction_seconds"),
    }
    extraction_current = {
        "affect_seconds": 0.0
        if affect_feature.reused
        else affect_feature.manifest.get("extraction_seconds"),
        "interaction_seconds": 0.0
        if interaction_feature.reused
        else interaction_feature.manifest.get("extraction_seconds"),
    }
    extraction_current["total_seconds"] = sum(
        float(value)
        for key, value in extraction_current.items()
        if key.endswith("_seconds") and value is not None
    )
    metrics = {
        "status": "complete",
        "run_id": run_id,
        "pair_id": pair.pair_id,
        "identity": {
            "affect_method_id": pair.affect_method_id,
            "affect_model_id": affect_model.model_id,
            "affect_feature_id": affect_feature.feature_id,
            "interaction_method_id": pair.interaction_method_id,
            "interaction_model_id": interaction_model.model_id,
            "interaction_feature_id": interaction_feature.feature_id,
            "engagement_model_id": engagement_model.model_id,
            "dataset_fingerprint": dataset_identity["fingerprint"],
            "split_identity": dict(dataset_identity["splits"]),
            "random_seed": training["seed"],
            "git": dict(git),
        },
        "feature_dimensions": {
            "affect": pair.affect_dim,
            "interaction": pair.interaction_dim,
            "matrix": pair.matrix_dim,
            "matrix_order": list(pair.matrix_order),
            "feature_layout": [
                entry.as_manifest() for entry in pair.feature_layout
            ],
        },
        "performance": dict(evaluation),
        "training": {
            key: training[key]
            for key in (
                "training_seconds",
                "epochs_requested",
                "epochs_completed",
                "best_epoch",
                "best_validation_loss",
                "best_validation_accuracy",
            )
        },
        "extraction": {
            "feature_generation": extraction_original,
            "this_run": extraction_current,
            "affect_reused": affect_feature.reused,
            "interaction_reused": interaction_feature.reused,
        },
        "model_cost": {
            "affect_checkpoint_size_mb": affect_size,
            "interaction_checkpoint_size_mb": interaction_size,
            "affect_parameter_count": affect_model.manifest.get("parameter_count"),
            "interaction_parameter_count": interaction_model.manifest.get(
                "parameter_count"
            ),
            "known_method_checkpoint_size_mb": known_method_size,
            "total_method_checkpoint_size_mb": sum(known_method_sizes)
            if full_method_size_known
            else None,
            "engagement_checkpoint_size_mb": engagement_size,
            "engagement_parameter_count": engagement_model.manifest.get(
                "parameter_count"
            ),
        },
        "matrix_manifest": dict(matrix_manifest),
        "environment": dict(environment_info) if environment_info is not None else None,
        "completed_at": utc_now(),
    }
    metrics["comparison"] = {
        "baseline_id": official_baseline.get("baseline_id")
        if official_baseline is not None
        else None,
        "values": comparison_values(metrics),
        "deltas": baseline_deltas(metrics, official_baseline)
        if official_baseline is not None
        else None,
    }
    metrics["documented_legacy_reference"] = dict(documented_reference)
    metrics["documented_reference_differences"] = {
        "accuracy_percentage_points": (
            float(evaluation["accuracy"])
            - float(documented_reference["accuracy"])
        )
        * 100.0,
        "f1_macro_percentage_points": (
            float(evaluation["f1_macro"])
            - float(documented_reference["f1_macro"])
        )
        * 100.0,
    }
    return metrics


def _git_identity(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
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

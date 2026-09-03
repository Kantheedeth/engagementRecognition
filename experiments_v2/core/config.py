"""Configuration loading and validation for the legacy golden-pair baseline."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


REQUIRED_SPLITS = ("train", "val", "test")


def load_config(path: str | Path, project_root: Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    resolved = validate_config(config, project_root=project_root)
    resolved["_config_path"] = str(config_path)
    return resolved


def _resolve(project_root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def validate_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    if resolved.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    experiment = resolved.get("experiment")
    dataset = resolved.get("dataset")
    methods = resolved.get("methods")
    engagement = resolved.get("engagement")
    if not all(isinstance(section, dict) for section in (experiment, dataset, methods, engagement)):
        raise ValueError("experiment, dataset, methods, and engagement must be objects")

    experiment.setdefault("name", "legacy_baseline")
    experiment["artifacts_root"] = _resolve(
        project_root, experiment.get("artifacts_root", "experiments_v2/artifacts")
    )

    split_files = dataset.get("split_files", {})
    if set(split_files) != set(REQUIRED_SPLITS):
        raise ValueError("dataset.split_files must define train, val, and test")
    dataset["split_files"] = {
        split: _resolve(project_root, split_files[split]) for split in REQUIRED_SPLITS
    }
    dataset["preprocessed_input_dir"] = _resolve(
        project_root,
        dataset.get("preprocessed_input_dir", "preprocessed_data/yolov5_640x640"),
    )
    preprocessing = dataset.setdefault("preprocessing", {})
    preprocessing.setdefault("num_frames", 8)
    preprocessing.setdefault("frame_size", [640, 640])
    preprocessing.setdefault("color", "RGB")
    if preprocessing["num_frames"] != 8:
        raise ValueError("The frozen legacy feature contracts require exactly 8 frames")

    for category in ("affect", "interaction"):
        entries = methods.get(category)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"methods.{category} must contain at least one method")
        enabled = [entry for entry in entries if entry.get("enabled", True)]
        if not enabled:
            raise ValueError(f"methods.{category} has no enabled method")
        for entry in enabled:
            if not isinstance(entry, dict) or not entry.get("code"):
                raise ValueError(f"Each {category} method requires a code")
            entry.setdefault("force_train", False)
            entry.setdefault("force_extract", False)
            entry.setdefault("parameters", {})
            if entry.get("legacy_feature_dir"):
                entry["legacy_feature_dir"] = _resolve(
                    project_root, entry["legacy_feature_dir"]
                )
    pairing = resolved.setdefault("pairing", {})
    if pairing.setdefault("strategy", "cartesian") != "cartesian":
        raise ValueError("Only cartesian pairing is supported")
    feature_order = pairing.setdefault("feature_order", ["interaction", "affect"])
    if not isinstance(feature_order, list) or len(feature_order) != 2:
        raise ValueError("pairing.feature_order must be a two-item list")
    if set(feature_order) != {"interaction", "affect"}:
        raise ValueError(
            "pairing.feature_order must contain interaction and affect exactly once"
        )

    baseline = resolved.setdefault("baseline", {})
    baseline.setdefault("publish_official", False)
    baseline.setdefault("affect_code", "A1")
    baseline.setdefault("interaction_code", "I1")
    if not isinstance(baseline["publish_official"], bool):
        raise ValueError("baseline.publish_official must be true or false")

    certification = resolved.setdefault("certification", {})
    certification.setdefault("required_python", "3.10")
    reference = certification.setdefault("documented_legacy_reference", {})
    reference.setdefault("label", "DOCUMENTED LEGACY REFERENCE")
    reference.setdefault("accuracy", 0.840909091)
    reference.setdefault("precision_macro", 0.814679884)
    reference.setdefault("recall_macro", 0.843492063)
    reference.setdefault("f1_macro", 0.826892110)
    paths = certification.setdefault("paths", {})
    feature_root_was_explicit = "legacy_feature_root" in paths
    for key, default in {
        "dataset_root": ".",
        "preprocessed_root": dataset["preprocessed_input_dir"],
        "legacy_feature_root": "preprocessed_features",
        "legacy_matrix_root": "feature_matrices_behavioral",
        "legacy_checkpoint_root": "checkpoints",
    }.items():
        paths[key] = _resolve(project_root, paths.get(key, default))

    dataset["preprocessed_input_dir"] = paths["preprocessed_root"]
    feature_subdirectories = {
        "affect": "affect_track_features",
        "interaction": "interaction_features",
    }
    if feature_root_was_explicit:
        for category, subdirectory in feature_subdirectories.items():
            for entry in methods[category]:
                if entry.get("code") in {"A1", "I1"}:
                    entry["legacy_feature_dir"] = str(
                        (Path(paths["legacy_feature_root"]) / subdirectory).resolve()
                    )

    model = engagement.setdefault("model", {})
    model.setdefault("branch_dim", 48)
    model.setdefault("num_heads", 4)
    model.setdefault("dropout", 0.15)
    if (model["branch_dim"] * 2) % model["num_heads"] != 0:
        raise ValueError("Twice branch_dim must be divisible by num_heads")

    training = engagement.setdefault("training", {})
    defaults = {
        "seed": 42,
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 15,
    }
    for key, value in defaults.items():
        training.setdefault(key, value)
    if training["epochs"] <= 0 or training["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")

    evaluation = engagement.setdefault("evaluation", {})
    evaluation.setdefault("batch_size", training["batch_size"])
    evaluation.setdefault("warmup_batches", 1)
    return resolved

"""Layout-aware factory for the unchanged legacy engagement classifier."""

from __future__ import annotations

from typing import Any, Mapping

from experiments_v2.core.contracts import PairDefinition


LEGACY_ARCHITECTURE = "legacy_pure_behavioral_attention"
LEGACY_INPUT_ORDER = ("interaction", "affect")


def resolve_engagement_model_config(
    *, pair: PairDefinition, model_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve branch widths and the total input width from pair metadata."""

    architecture = str(model_config.get("architecture", LEGACY_ARCHITECTURE))
    if architecture != LEGACY_ARCHITECTURE:
        raise ValueError(f"Unsupported engagement architecture: {architecture!r}")
    if pair.matrix_order != LEGACY_INPUT_ORDER:
        raise ValueError(
            f"{LEGACY_ARCHITECTURE} requires layout {LEGACY_INPUT_ORDER}, "
            f"got {pair.matrix_order}"
        )

    dim_inter = pair.dimension("interaction")
    dim_affect = pair.dimension("affect")
    raw_input_dim = pair.matrix_dim
    configured_dimensions = {
        "dim_inter": dim_inter,
        "dim_affect": dim_affect,
        "raw_input_dim": raw_input_dim,
    }
    for key, expected in configured_dimensions.items():
        if key in model_config and int(model_config[key]) != expected:
            raise ValueError(
                f"Configured {key}={model_config[key]} conflicts with pair-derived "
                f"value {expected}"
            )

    branch_dim = int(model_config["branch_dim"])
    num_heads = int(model_config["num_heads"])
    if branch_dim <= 0 or num_heads <= 0:
        raise ValueError("branch_dim and num_heads must be positive")
    if (branch_dim * 2) % num_heads != 0:
        raise ValueError("Twice branch_dim must be divisible by num_heads")

    return {
        "architecture": architecture,
        "raw_input_dim": raw_input_dim,
        "temporal_frames": pair.temporal_frames,
        "feature_order": list(pair.matrix_order),
        "feature_layout": [entry.as_manifest() for entry in pair.feature_layout],
        "dim_inter": dim_inter,
        "dim_affect": dim_affect,
        "branch_dim": branch_dim,
        "num_heads": num_heads,
        "num_classes": int(model_config.get("num_classes", 3)),
        "dropout": float(model_config["dropout"]),
    }


def classifier_constructor_config(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only arguments accepted by the unchanged legacy classifier."""

    return {
        key: resolved_config[key]
        for key in (
            "dim_inter",
            "dim_affect",
            "branch_dim",
            "num_heads",
            "num_classes",
            "dropout",
        )
    }


def create_engagement_model(
    *,
    pair: PairDefinition,
    model_config: Mapping[str, Any],
    model_class: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate the legacy architecture with pair-derived feature widths."""

    resolved = resolve_engagement_model_config(pair=pair, model_config=model_config)
    if model_class is None:
        from src.models.model_behavioral import PureBehavioralAttentionClassifier

        model_class = PureBehavioralAttentionClassifier
    model = model_class(**classifier_constructor_config(resolved))
    return model, resolved


def validate_checkpoint_input_contract(
    *, pair: PairDefinition, checkpoint_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject a checkpoint whose recorded input contract no longer matches its pair."""

    return resolve_engagement_model_config(pair=pair, model_config=checkpoint_config)

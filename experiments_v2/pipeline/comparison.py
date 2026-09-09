"""Raw, unit-labelled metric deltas against an official V2 baseline."""

from __future__ import annotations

from typing import Any, Mapping


def comparison_values(metrics: Mapping[str, Any]) -> dict[str, Any]:
    performance = metrics.get("performance", {})
    model_cost = metrics.get("model_cost", {})
    extraction = metrics.get("extraction", {}).get("feature_generation", {})
    feature_method_size = model_cost.get("combined_feature_method_size_mb")
    engagement_size = model_cost.get("engagement_checkpoint_size_mb")
    model_size = None
    if feature_method_size is not None and engagement_size is not None:
        model_size = float(feature_method_size) + float(engagement_size)
    elif engagement_size is not None:
        model_size = engagement_size
    return {
        "accuracy": performance.get("accuracy"),
        "f1_macro": performance.get("f1_macro"),
        "checkpoint_size_mb": engagement_size,
        "model_size_mb": model_size,
        "parameter_count": model_cost.get("engagement_parameter_count"),
        "extraction_seconds": extraction.get("total_seconds"),
        "inference_seconds": performance.get("inference_seconds"),
        "inference_ms_per_video": performance.get("inference_ms_per_video"),
        "fps": performance.get("engagement_fps"),
    }


def baseline_deltas(
    candidate_metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = comparison_values(candidate_metrics)
    reference = baseline.get("comparison_values", {})

    def difference(candidate_key: str, *, multiplier: float = 1.0) -> float | int | None:
        current = candidate.get(candidate_key)
        previous = reference.get(candidate_key)
        if current is None or previous is None:
            return None
        value = (float(current) - float(previous)) * multiplier
        if candidate_key == "parameter_count":
            return int(value)
        return value

    return {
        "baseline_id": baseline.get("baseline_id"),
        "accuracy_delta": difference("accuracy", multiplier=100.0),
        "f1_delta": difference("f1_macro", multiplier=100.0),
        "size_delta_mb": difference("checkpoint_size_mb"),
        "accuracy_delta_pp": difference("accuracy", multiplier=100.0),
        "macro_f1_delta_pp": difference("f1_macro", multiplier=100.0),
        "model_size_delta_mb": difference("model_size_mb"),
        "parameter_delta": difference("parameter_count"),
        "extraction_time_delta": difference("extraction_seconds"),
        "inference_time_delta": difference("inference_seconds"),
        "inference_ms_per_video_delta": difference("inference_ms_per_video"),
        "fps_delta": difference("fps"),
        "units": {
            "accuracy_delta": "percentage_points",
            "f1_delta": "percentage_points",
            "size_delta_mb": "megabytes",
            "accuracy_delta_pp": "percentage_points",
            "macro_f1_delta_pp": "percentage_points",
            "model_size_delta_mb": "megabytes",
            "parameter_delta": "parameters",
            "extraction_time_delta": "seconds",
            "inference_time_delta": "seconds_for_test_split",
            "inference_ms_per_video_delta": "milliseconds_per_video",
            "fps_delta": "videos_per_second",
        },
    }

"""Read-only accounting of learned weight files used by legacy A1 and I1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments_v2.core.artifacts import sha256_file


def _weight(path: Path, *, role: str, reference: str) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else None
    return {
        "role": role,
        "reference": reference,
        "path": str(path.resolve()) if exists else str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024.0 * 1024.0) if size_bytes is not None else None,
        "sha256": sha256_file(path) if exists else None,
    }


def _huggingface_weight(model_id: str) -> Path:
    cache_name = "models--" + model_id.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    revision_path = root / "refs" / "main"
    revisions = []
    if revision_path.is_file():
        revisions.append(revision_path.read_text(encoding="utf-8").strip())
    revisions.extend(
        path.name for path in sorted((root / "snapshots").glob("*")) if path.is_dir()
    )
    for revision in dict.fromkeys(revisions):
        snapshot = root / "snapshots" / revision
        for filename in ("model.safetensors", "pytorch_model.bin"):
            candidate = snapshot / filename
            if candidate.is_file():
                return candidate
    return root / "snapshots" / "MISSING_MODEL_WEIGHTS"


def inspect_feature_method_footprint(
    *, project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Account only learned weights; trackers and descriptors have no learned file."""

    entries = {
        category: next(
            entry
            for entry in config["methods"][category]
            if entry.get("enabled", True)
        )
        for category in ("affect", "interaction")
    }
    affect_parameters = entries["affect"]["parameters"]
    insightface_root = Path(str(affect_parameters["insightface_root"])).expanduser()
    retinaface_model = str(affect_parameters["retinaface_model"])
    fer_model_id = str(affect_parameters["fer_model_id"])
    affect_components = [
        _weight(
            insightface_root / "models" / retinaface_model / "det_500m.onnx",
            role="face_detector",
            reference=retinaface_model,
        ),
        _weight(
            _huggingface_weight(fer_model_id),
            role="facial_emotion_recognizer",
            reference=fer_model_id,
        ),
    ]

    interaction_reference = str(entries["interaction"]["parameters"]["model"])
    interaction_path = Path(interaction_reference).expanduser()
    if not interaction_path.is_absolute():
        interaction_path = project_root / interaction_path
    interaction_components = [
        _weight(
            interaction_path,
            role="person_detector",
            reference=interaction_reference,
        )
    ]

    def summarize(components: list[dict[str, Any]]) -> dict[str, Any]:
        sizes = [item["size_bytes"] for item in components]
        complete = all(value is not None for value in sizes)
        total_bytes = sum(int(value) for value in sizes if value is not None)
        return {
            "components": components,
            "complete": complete,
            "learned_weight_size_bytes": total_bytes if complete else None,
            "learned_weight_size_mb": total_bytes / (1024.0 * 1024.0)
            if complete
            else None,
            "parameter_count": None,
            "parameter_count_note": "not exposed by the immutable feature manifests",
        }

    affect = summarize(affect_components)
    interaction = summarize(interaction_components)
    combined_complete = affect["complete"] and interaction["complete"]
    combined_bytes = (
        int(affect["learned_weight_size_bytes"])
        + int(interaction["learned_weight_size_bytes"])
        if combined_complete
        else None
    )
    return {
        "scope": "learned weight files used by A1 and I1",
        "affect": affect,
        "interaction": interaction,
        "combined": {
            "complete": combined_complete,
            "learned_weight_size_bytes": combined_bytes,
            "learned_weight_size_mb": combined_bytes / (1024.0 * 1024.0)
            if combined_bytes is not None
            else None,
            "parameter_count": None,
        },
    }

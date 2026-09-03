"""I1 adapter for the frozen YOLOv8 32-D interaction implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from experiments_v2.adapters.base import (
    load_legacy_manifest,
    run_logged,
    validate_numpy_tree,
)
from experiments_v2.core.contracts import MethodSpec


class LegacyInteractionAdapter:
    spec = MethodSpec(
        code="I1",
        method_id="METHOD_I1",
        name="legacy_interaction",
        category="interaction",
        version="1",
        feature_dim=32,
        feature_schema="yolov8_person_geometry_32_v1",
        input_kind="rgb_640x640_npz",
        trainable=False,
        components=("yolov8_person_detector", "geometry_descriptor"),
    )

    def model_identity(self, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = str(parameters.get("model", "yolov8n.pt"))
        return {
            "architecture": "yolov8_person_geometry_32",
            "pretrained": True,
            "components": [
                {
                    "role": "person_detector",
                    "library": "ultralytics",
                    "reference": reference,
                },
                {
                    "role": "geometry_descriptor",
                    "reference": "legacy_32d_descriptor_v1",
                },
            ],
        }

    def extract_features(
        self,
        *,
        project_root: Path,
        input_dir: Path,
        output_dir: Path,
        parameters: Mapping[str, Any],
        log_path: Path,
    ) -> Mapping[str, Any]:
        script = project_root / "src" / "data" / "extract_interaction_features.py"
        command = [
            sys.executable,
            str(script),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--model",
            str(parameters.get("model", "yolov8n.pt")),
            "--confidence",
            str(parameters.get("confidence", 0.25)),
            "--device",
            str(parameters.get("device", "auto")),
        ]
        execution = run_logged(command, cwd=project_root, log_path=log_path)
        validation = self.validate_features(output_dir)
        return {**execution, **validation}

    def validate_features(self, output_dir: Path) -> Mapping[str, Any]:
        legacy_manifest = load_legacy_manifest(output_dir / "extraction_manifest.json")
        if legacy_manifest.get("feature_schema") != self.spec.feature_schema:
            raise ValueError(
                "Unexpected legacy interaction schema: "
                f"{legacy_manifest.get('feature_schema')!r}"
            )
        if legacy_manifest.get("shape_per_video") != [8, self.spec.feature_dim]:
            raise ValueError("Legacy interaction manifest has an incompatible shape contract")
        file_count = validate_numpy_tree(output_dir, (8, self.spec.feature_dim))
        return {"legacy_manifest": legacy_manifest, "validated_files": file_count}

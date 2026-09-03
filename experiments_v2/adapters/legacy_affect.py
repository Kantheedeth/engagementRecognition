"""A1 adapter for the frozen RetinaFace + ByteTrack + FER implementation."""

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


class LegacyAffectAdapter:
    spec = MethodSpec(
        code="A1",
        method_id="METHOD_A1",
        name="legacy_affect",
        category="affect",
        version="1",
        feature_dim=8,
        feature_schema="retinaface_bytetrack_fer_v1",
        input_kind="rgb_640x640_npz",
        trainable=False,
        components=("retinaface", "bytetrack", "fer"),
    )

    def model_identity(self, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        fer_backend = parameters.get("fer_backend", "huggingface")
        fer_reference = (
            parameters.get("fer_model_id", "abhilash88/face-emotion-detection")
            if fer_backend == "huggingface"
            else parameters.get("fer_checkpoint")
        )
        return {
            "architecture": "retinaface_bytetrack_fer",
            "pretrained": True,
            "components": [
                {
                    "role": "face_detector",
                    "library": "insightface",
                    "reference": parameters.get("retinaface_model", "buffalo_s"),
                },
                {
                    "role": "tracker",
                    "library": "ultralytics-bytetrack",
                    "reference": "configuration-only",
                },
                {
                    "role": "fer",
                    "backend": fer_backend,
                    "reference": fer_reference,
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
        script = project_root / "src" / "data" / "extract_affect_features.py"
        command = [
            sys.executable,
            str(script),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--device",
            str(parameters.get("device", "auto")),
            "--retinaface_model",
            str(parameters.get("retinaface_model", "buffalo_s")),
            "--insightface_root",
            str(parameters.get("insightface_root", "~/.insightface")),
            "--det_size",
            str(parameters.get("det_size", 640)),
            "--det_threshold",
            str(parameters.get("det_threshold", 0.45)),
            "--max_faces",
            str(parameters.get("max_faces", 64)),
            "--fer_backend",
            str(parameters.get("fer_backend", "huggingface")),
            "--fer_model_id",
            str(parameters.get("fer_model_id", "abhilash88/face-emotion-detection")),
            "--fer_input_size",
            str(parameters.get("fer_input_size", 224)),
            "--track_high_threshold",
            str(parameters.get("track_high_threshold", 0.45)),
            "--track_low_threshold",
            str(parameters.get("track_low_threshold", 0.10)),
            "--new_track_threshold",
            str(parameters.get("new_track_threshold", 0.45)),
            "--track_match_threshold",
            str(parameters.get("track_match_threshold", 0.80)),
            "--track_buffer",
            str(parameters.get("track_buffer", 8)),
            "--expected_faces",
            str(parameters.get("expected_faces", 8)),
            "--emotion_momentum",
            str(parameters.get("emotion_momentum", 0.60)),
            "--missed_detection_decay",
            str(parameters.get("missed_detection_decay", 0.35)),
            "--max_feature_age",
            str(parameters.get("max_feature_age", 1)),
        ]
        command.append("--tracking" if parameters.get("tracking", True) else "--no-tracking")
        if parameters.get("local_files_only", False):
            command.append("--local_files_only")
        if parameters.get("fer_checkpoint"):
            command.extend(["--fer_checkpoint", str(parameters["fer_checkpoint"])])
        if parameters.get("fer_source_emotions"):
            command.append("--fer_source_emotions")
            command.extend(str(value) for value in parameters["fer_source_emotions"])

        execution = run_logged(command, cwd=project_root, log_path=log_path)
        validation = self.validate_features(output_dir)
        return {**execution, **validation}

    def validate_features(self, output_dir: Path) -> Mapping[str, Any]:
        legacy_manifest = load_legacy_manifest(output_dir / "extraction_manifest.json")
        if legacy_manifest.get("feature_schema") != self.spec.feature_schema:
            raise ValueError(
                f"Unexpected legacy affect schema: {legacy_manifest.get('feature_schema')!r}"
            )
        if legacy_manifest.get("shape_per_video") != [8, self.spec.feature_dim]:
            raise ValueError("Legacy affect manifest has an incompatible shape contract")
        file_count = validate_numpy_tree(output_dir, (8, self.spec.feature_dim))
        return {"legacy_manifest": legacy_manifest, "validated_files": file_count}

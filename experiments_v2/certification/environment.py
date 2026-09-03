"""Non-installing inspection of the documented Python 3.10 ML environment."""

from __future__ import annotations

import importlib.util
from importlib import metadata
import platform
import re
import sys
from typing import Any, Iterable


DEPENDENCIES = {
    "numpy": {"distribution": "numpy", "import": "numpy", "minimum": "1.26"},
    "torch": {"distribution": "torch", "import": "torch", "minimum": "2.2"},
    "torchvision": {
        "distribution": "torchvision",
        "import": "torchvision",
        "minimum": "0.17",
    },
    "opencv": {
        "distribution": "opencv-python",
        "import": "cv2",
        "minimum": "4.8",
    },
    "scikit_learn": {
        "distribution": "scikit-learn",
        "import": "sklearn",
        "minimum": None,
    },
    "matplotlib": {
        "distribution": "matplotlib",
        "import": "matplotlib",
        "minimum": None,
    },
    "insightface": {
        "distribution": "insightface",
        "import": "insightface",
        "minimum": "0.7.3",
    },
    "onnxruntime": {
        "distribution": "onnxruntime",
        "import": "onnxruntime",
        "minimum": "1.17",
    },
    "ultralytics": {
        "distribution": "ultralytics",
        "import": "ultralytics",
        "minimum": "8.3",
    },
    "lap": {"distribution": "lap", "import": "lap", "minimum": "0.5.12"},
    "transformers": {
        "distribution": "transformers",
        "import": "transformers",
        "minimum": "4.45",
    },
    "huggingface_hub": {
        "distribution": "huggingface-hub",
        "import": "huggingface_hub",
        "minimum": "0.25",
    },
    "tqdm": {"distribution": "tqdm", "import": "tqdm", "minimum": "4.66"},
}

CORE_TRAINING = ("numpy", "torch", "scikit_learn", "matplotlib")
FEATURE_EXTRACTION = tuple(DEPENDENCIES)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:4])


def _meets_minimum(version: str | None, minimum: str | None) -> bool:
    if version is None:
        return False
    if minimum is None:
        return True
    current = _version_tuple(version)
    required = _version_tuple(minimum)
    length = max(len(current), len(required))
    return current + (0,) * (length - len(current)) >= required + (0,) * (
        length - len(required)
    )


def _profile(
    names: Iterable[str], packages: dict[str, dict[str, Any]], python_match: bool
) -> dict[str, Any]:
    missing = [name for name in names if not packages[name]["available"]]
    incompatible = [
        name
        for name in names
        if packages[name]["available"] and not packages[name]["version_satisfied"]
    ]
    ready = python_match and not missing and not incompatible
    return {
        "status": "READY" if ready else "NOT_READY",
        "ready": ready,
        "required_dependencies": list(names),
        "missing_dependencies": missing,
        "incompatible_dependencies": incompatible,
    }


def inspect_environment(required_python: str = "3.10") -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = {}
    for name, specification in DEPENDENCIES.items():
        import_name = str(specification["import"])
        available = importlib.util.find_spec(import_name) is not None
        version = None
        if available:
            try:
                version = metadata.version(str(specification["distribution"]))
            except metadata.PackageNotFoundError:
                version = "unknown"
        packages[name] = {
            "distribution": specification["distribution"],
            "import_name": import_name,
            "minimum_version": specification["minimum"],
            "available": available,
            "version": version,
            "version_satisfied": _meets_minimum(
                version, specification["minimum"]
            ),
        }

    detected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_match = detected_python == required_python
    hardware = {"cuda_available": False, "mps_available": False, "probe_error": None}
    if packages["torch"]["available"]:
        try:
            import torch

            hardware["cuda_available"] = bool(torch.cuda.is_available())
            hardware["mps_available"] = bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception as exc:
            hardware["probe_error"] = f"{type(exc).__name__}: {exc}"

    profiles = {
        "reuse_and_training": _profile(CORE_TRAINING, packages, python_match),
        "feature_extraction_and_training": _profile(
            FEATURE_EXTRACTION, packages, python_match
        ),
    }
    any_profile_ready = any(profile["ready"] for profile in profiles.values())
    return {
        "status": "READY" if any_profile_ready else "NOT_READY",
        "ready": any_profile_ready,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "detected_major_minor": detected_python,
            "required_major_minor": required_python,
            "certification_match": python_match,
            "readme_supported": sys.version_info >= (3, 8),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "hardware": hardware,
        "packages": packages,
        "profiles": profiles,
    }

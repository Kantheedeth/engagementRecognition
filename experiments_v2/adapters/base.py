"""Subprocess and validation helpers shared by legacy adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> Mapping[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write("COMMAND\n")
        log.write(json.dumps(command))
        log.write("\n\nOUTPUT\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Legacy extractor failed with exit code {result.returncode}; see {log_path}"
        )
    return {"command": command, "returncode": result.returncode, "log": str(log_path)}


def validate_numpy_tree(output_dir: Path, expected_shape: tuple[int, int]) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to validate extracted legacy features") from exc

    paths = sorted(output_dir.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"No extracted .npy feature files found under {output_dir}")
    for path in paths:
        array = np.load(path, allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(f"{path} has shape {array.shape}; expected {expected_shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{path} has dtype {array.dtype}; expected float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{path} contains non-finite values")
    return len(paths)


def load_legacy_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy extraction manifest was not produced: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid legacy manifest: {path}")
    return manifest

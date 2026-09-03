"""Immutable IDs, fingerprints, manifests, and dataset identity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    normalized = prefix.strip().upper()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError(f"Invalid ID prefix: {prefix!r}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{normalized}_{timestamp}_{uuid.uuid4().hex[:8].upper()}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size_mb(path: Path) -> float | None:
    return path.stat().st_size / (1024.0 * 1024.0) if path.is_file() else None


def directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024.0 * 1024.0
    )


def create_exclusive_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json_exclusive(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    """Write a new JSON file without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def inventory_identity(root: Path, suffixes: Iterable[str] = (".npz", ".pt")) -> dict[str, Any]:
    """Create a cheap change-sensitive identity for preprocessed artifacts."""
    allowed = set(suffixes)
    entries = []
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if allowed and path.suffix not in allowed:
                continue
            stat = path.stat()
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "root": str(root.resolve()),
        "file_count": len(entries),
        "inventory_fingerprint": fingerprint(entries),
    }


def dataset_identity(
    split_files: Mapping[str, Path],
    input_dir: Path,
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    split_identity = {}
    for split, path in sorted(split_files.items()):
        split_identity[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    identity = {
        "splits": split_identity,
        "preprocessed_inputs": inventory_identity(input_dir),
        "preprocessing": dict(preprocessing),
    }
    # Cache identity intentionally depends on the declared preprocessing contract
    # and split contents, not on the continued presence of expensive inputs. This
    # allows a verified feature cache to remain reusable after inputs are archived.
    cache_identity = {
        "splits": split_identity,
        "preprocessing": dict(preprocessing),
    }
    return {**identity, "fingerprint": fingerprint(cache_identity)}


def find_manifest_by_fingerprint(root: Path, expected: str) -> tuple[Path, dict[str, Any]] | None:
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("manifest.json")):
        try:
            manifest = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("fingerprint") == expected and manifest.get("status") == "complete":
            return path, manifest
    return None

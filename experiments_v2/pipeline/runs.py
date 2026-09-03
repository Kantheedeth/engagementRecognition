"""Exclusive allocation of immutable V2 experiment run directories."""

from __future__ import annotations

from pathlib import Path

from experiments_v2.core.artifacts import create_exclusive_dir, new_id


class RunStore:
    def __init__(self, artifacts_root: Path) -> None:
        self.root = artifacts_root / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> tuple[str, Path]:
        run_id = new_id("RUN")
        return run_id, create_exclusive_dir(self.root / run_id)

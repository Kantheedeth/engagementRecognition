"""Shared contracts for methods and versioned experimental artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


VALID_CATEGORIES = {"affect", "interaction"}


@dataclass(frozen=True)
class MethodSpec:
    """Static description of a registered experimental method."""

    code: str
    method_id: str
    name: str
    category: str
    version: str
    feature_dim: int
    feature_schema: str
    input_kind: str
    trainable: bool = False
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Unsupported method category: {self.category!r}")
        if not self.code or not self.method_id.startswith("METHOD_"):
            raise ValueError("Methods require a code and a METHOD_* identifier")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    method_id: str
    category: str
    fingerprint: str
    directory: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class FeatureArtifact:
    feature_id: str
    method_id: str
    model_id: str
    category: str
    fingerprint: str
    directory: Path
    data_dir: Path
    feature_dim: int
    manifest: Mapping[str, Any]
    reused: bool = False


@dataclass(frozen=True)
class FeatureLayoutEntry:
    """One contiguous method-owned segment in a pair feature matrix."""

    category: str
    method_id: str
    model_id: str
    feature_id: str
    feature_dim: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Unsupported layout category: {self.category!r}")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.start < 0 or self.end != self.start + self.feature_dim:
            raise ValueError(
                f"Invalid {self.category} segment [{self.start}:{self.end}] "
                f"for feature_dim={self.feature_dim}"
            )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "feature_id": self.feature_id,
            "feature_dim": self.feature_dim,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class PairDefinition:
    pair_id: str
    feature_layout: tuple[FeatureLayoutEntry, ...]
    temporal_frames: int
    directory: Path
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pair_id.startswith("PAIR_"):
            raise ValueError("Pair IDs must start with PAIR_")
        if self.temporal_frames <= 0:
            raise ValueError("temporal_frames must be positive")
        if not self.feature_layout:
            raise ValueError("A pair requires at least one feature layout entry")
        expected_start = 0
        categories: set[str] = set()
        for entry in self.feature_layout:
            if entry.category in categories:
                raise ValueError(f"Duplicate pair category: {entry.category}")
            if entry.start != expected_start:
                raise ValueError(
                    f"Pair layout is not contiguous at {entry.category}: "
                    f"expected start {expected_start}, got {entry.start}"
                )
            categories.add(entry.category)
            expected_start = entry.end

    @property
    def matrix_order(self) -> tuple[str, ...]:
        return tuple(entry.category for entry in self.feature_layout)

    @property
    def matrix_dim(self) -> int:
        return sum(entry.feature_dim for entry in self.feature_layout)

    def entry(self, category: str) -> FeatureLayoutEntry:
        for entry in self.feature_layout:
            if entry.category == category:
                return entry
        raise KeyError(f"Pair {self.pair_id} has no {category!r} feature segment")

    def dimension(self, category: str) -> int:
        return self.entry(category).feature_dim

    @property
    def affect_method_id(self) -> str:
        return self.entry("affect").method_id

    @property
    def affect_model_id(self) -> str:
        return self.entry("affect").model_id

    @property
    def affect_feature_id(self) -> str:
        return self.entry("affect").feature_id

    @property
    def affect_dim(self) -> int:
        return self.dimension("affect")

    @property
    def interaction_method_id(self) -> str:
        return self.entry("interaction").method_id

    @property
    def interaction_model_id(self) -> str:
        return self.entry("interaction").model_id

    @property
    def interaction_feature_id(self) -> str:
        return self.entry("interaction").feature_id

    @property
    def interaction_dim(self) -> int:
        return self.dimension("interaction")


class MethodAdapter(Protocol):
    """Dataset-level adapter suitable for legacy and future composite methods."""

    spec: MethodSpec

    def model_identity(self, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the immutable identity of all models used by this method."""

    def extract_features(
        self,
        *,
        project_root: Path,
        input_dir: Path,
        output_dir: Path,
        parameters: Mapping[str, Any],
        log_path: Path,
    ) -> Mapping[str, Any]:
        """Extract and validate a complete dataset feature set."""

    def validate_features(self, output_dir: Path) -> Mapping[str, Any]:
        """Revalidate a published feature set before cache reuse."""

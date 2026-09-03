"""Lazy registry for affect and interaction method adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from experiments_v2.core.contracts import MethodAdapter, MethodSpec


class MethodRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, str] = {}
        self._specs: dict[str, MethodSpec] = {}

    def register(self, *, spec: MethodSpec, factory: str) -> None:
        if spec.code in self._factories:
            raise ValueError(f"Method code already registered: {spec.code}")
        if any(existing.method_id == spec.method_id for existing in self._specs.values()):
            raise ValueError(f"Method ID already registered: {spec.method_id}")
        if ":" not in factory:
            raise ValueError("Factory must use 'module:attribute' syntax")
        self._factories[spec.code] = factory
        self._specs[spec.code] = spec

    def spec(self, code: str) -> MethodSpec:
        try:
            return self._specs[code]
        except KeyError as exc:
            raise KeyError(f"Unknown method code: {code}") from exc

    def create(self, code: str) -> MethodAdapter:
        self.spec(code)
        module_name, attribute = self._factories[code].split(":", maxsplit=1)
        factory: Any = getattr(import_module(module_name), attribute)
        adapter = factory()
        if adapter.spec != self._specs[code]:
            raise ValueError(f"Factory for {code} returned a mismatched method specification")
        return adapter

    def list(self, category: str | None = None) -> list[MethodSpec]:
        specs = self._specs.values()
        return sorted(
            (spec for spec in specs if category is None or spec.category == category),
            key=lambda spec: spec.code,
        )

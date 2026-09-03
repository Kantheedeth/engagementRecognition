"""Built-in registrations for frozen legacy baselines."""

from experiments_v2.adapters.legacy_affect import LegacyAffectAdapter
from experiments_v2.adapters.legacy_interaction import LegacyInteractionAdapter
from experiments_v2.registry.method_registry import MethodRegistry


def create_builtin_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(
        spec=LegacyAffectAdapter.spec,
        factory="experiments_v2.adapters.legacy_affect:LegacyAffectAdapter",
    )
    registry.register(
        spec=LegacyInteractionAdapter.spec,
        factory="experiments_v2.adapters.legacy_interaction:LegacyInteractionAdapter",
    )
    return registry

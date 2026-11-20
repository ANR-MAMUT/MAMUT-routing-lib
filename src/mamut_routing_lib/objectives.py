"""Which objective an instance is solved and stored under, by default.

Two different defaults:

- :func:`base_objective` is the objective the family *publishes* its BKS under
  (``docs/benchmarks/formats/solutions.md``): Sintef2008 ranks fleet first
  (``HierarchicalVehicleCost``), Blauth2024 adds a fleet fixed cost
  (``FleetCostDuration``), other TD families use ``Duration`` and every other
  static family ``MonoCost`` (Ortec2022 included: the EURO Meets NeurIPS 2022
  challenge ranked total cost, and all its BKS are ``MonoCost``). ``bks submit`` and ``check`` default to it, so a
  submitted solution is compared with the published file.
- :func:`default_objective` is what a solve uses when the caller names none:
  ``MonoCost`` for static instances (warning when the family publishes another
  objective, as ``mamut-routing solve`` always has) and ``Duration`` for
  time-dependent ones.
"""

from __future__ import annotations

from typing import Any

from mamut_routing_lib.enums import BenchmarkName, ObjectiveFunction, ProblemType

__all__ = [
    "HIERARCHICAL_FAMILIES",
    "FLEET_COST_FAMILIES",
    "base_objective",
    "default_objective",
    "require_supported_objective",
    "supported_objectives",
]

#: Static families whose BKS rank solutions by fleet size first.
HIERARCHICAL_FAMILIES = frozenset({BenchmarkName.SINTEF_2008.value})
#: Time-dependent families whose objective adds a fixed cost per used vehicle.
FLEET_COST_FAMILIES = frozenset({BenchmarkName.BLAUTH_2024.value})


def _benchmark(instance: Any) -> str | None:
    value = getattr(instance, "benchmark_name", None)
    return getattr(value, "value", value)


def _problem_type(instance: Any) -> ProblemType:
    from mamut_routing_lib.artifacts import instance_problem_type

    return instance_problem_type(instance)


def base_objective(instance: Any) -> ObjectiveFunction:
    """The objective the instance's family publishes its BKS under."""
    problem = _problem_type(instance)
    benchmark = _benchmark(instance)
    if problem in (ProblemType.TDVRP, ProblemType.TDVRPTW):
        if benchmark in FLEET_COST_FAMILIES:
            return ObjectiveFunction.FLEET_COST_DURATION
        return ObjectiveFunction.DURATION
    if problem is ProblemType.VRPTW and benchmark in HIERARCHICAL_FAMILIES:
        return ObjectiveFunction.HIERARCHICAL_VEHICLE_COST
    return ObjectiveFunction.MONO_COST


def supported_objectives(instance: Any) -> tuple[ObjectiveFunction, ...]:
    """The objectives a solution of this instance can be checked and stored under.

    CVRP: ``MonoCost``. VRPTW: ``MonoCost`` and ``HierarchicalVehicleCost``. Time-dependent
    instances: ``Duration`` and ``FleetCostDuration`` (the TD checker's objectives).
    """
    problem = _problem_type(instance)
    if problem in (ProblemType.TDVRP, ProblemType.TDVRPTW):
        return (ObjectiveFunction.DURATION, ObjectiveFunction.FLEET_COST_DURATION)
    if problem is ProblemType.VRPTW:
        return (ObjectiveFunction.MONO_COST, ObjectiveFunction.HIERARCHICAL_VEHICLE_COST)
    return (ObjectiveFunction.MONO_COST,)


def require_supported_objective(instance: Any, objective: ObjectiveFunction) -> ObjectiveFunction:
    """``objective`` if it applies to the instance, else ``ValueError`` naming the ones that do."""
    allowed = supported_objectives(instance)
    if objective not in allowed:
        raise ValueError(
            f"{ObjectiveFunction(objective).value} does not apply to {_problem_type(instance).value} instances "
            f"(use {' or '.join(item.value for item in allowed)})"
        )
    return objective


def default_objective(instance: Any) -> ObjectiveFunction:
    """The objective a solve uses when none is given: MonoCost (static) or Duration (TD)."""
    if _problem_type(instance) in (ProblemType.TDVRP, ProblemType.TDVRPTW):
        return ObjectiveFunction.DURATION
    return ObjectiveFunction.MONO_COST

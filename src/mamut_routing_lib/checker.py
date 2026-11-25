"""Check and price static routes under the family's binary64-exact-v1 profile."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from mamut_routing_lib.enums import ObjectiveFunction, SolutionCheckStatus
from mamut_routing_lib.exact_types import ExactBackend
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkInstanceVRPTW, BenchmarkInstanceCVRP, BenchmarkSolution, LoadedStaticInstance


class SolutionCheckResult(BaseModel):
    """Checked route count, display cost and lossless exact evaluation for a valid solution.

    A ``valid`` result carries a nonnegative ``num_routes``. Its ``routing_cost``
    is present, except when ``exact_evaluation`` reports ``display_overflow``:
    the exact cost then has no finite binary64 display and ``routing_cost`` is
    None. A solution that names another instance or declares another objective
    is not checked at all. The checker raises ``ValueError`` for it instead of
    returning a status.
    """
    model_config = ConfigDict(extra="forbid")

    status: SolutionCheckStatus
    routing_cost: int | float | None
    num_routes: int | None
    error_message: str = ""
    exact_evaluation: dict | None = None

    @model_validator(mode="after")
    def _valid_result_carries_count_and_cost(self) -> "SolutionCheckResult":
        if self.status != SolutionCheckStatus.VALID:
            return self
        if self.num_routes is None or self.num_routes < 0:
            raise ValueError("a valid result requires a nonnegative num_routes")
        overflow = isinstance(self.exact_evaluation, dict) and self.exact_evaluation.get("display_overflow") is True
        if (self.routing_cost is None) != overflow:
            raise ValueError("a valid result carries a display cost exactly when its exact cost does not overflow binary64")
        return self

    def is_valid(self) -> bool:
        """True when the status is ``valid``."""
        return self.status == SolutionCheckStatus.VALID

    @classmethod
    def make_invalid(cls, status: SolutionCheckStatus, error_message: str) -> "SolutionCheckResult":
        """A rejection with ``status`` and a human-readable ``error_message``."""
        return cls(status=status, routing_cost=None, num_routes=None, error_message=error_message)

    @classmethod
    def make_valid(cls, routing_cost: int | float, num_routes: int) -> "SolutionCheckResult":
        """A ``valid`` result carrying the priced cost and route count."""
        return cls(status=SolutionCheckStatus.VALID, routing_cost=routing_cost, num_routes=num_routes)


def _iter_routes(solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]]) -> list[list[int]]:
    if isinstance(solution_or_routes, list):
        return solution_or_routes
    return solution_or_routes.routes


def _require_static_matrix(instance: LoadedStaticInstance) -> None:
    if not isinstance(instance, LoadedStaticInstance):
        raise TypeError("Expected a loaded static instance with an arc matrix. Hydrate slim collection "
                        "instances first. Time-dependent instances require the TD checker and ATFs.")


def compute_route_cost(instance: LoadedStaticInstance, route: list[int], *, backend: ExactBackend = "native") -> Fraction:
    """Return the exact cost of the listed depot-to-depot arcs, including a partial tour."""
    return compute_solution_cost(instance, [route], backend=backend)


def compute_solution_cost(
    instance: LoadedStaticInstance,
    solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]],
    *,
    backend: ExactBackend = "native",
) -> Fraction:
    """Sum the traversed arcs under binary64-exact-v1. Use check_solution to also check feasibility."""
    from mamut_routing_lib.exact_static import prepare_static

    _require_static_matrix(instance)
    return prepare_static(instance, backend=backend).routing_cost(_iter_routes(solution_or_routes))


def canonical_route_order(routes: list[list[int]]) -> list[list[int]]:
    """Routes sorted by their first customer: the canonical storage and summation order of every BKS."""
    return sorted(routes, key=lambda route: route[0])


def shortest_decimal_cost_value(value: int | float) -> Fraction:
    """Keep integers exact and interpret a float's shortest round-trip decimal spelling.

    For example, the float 0.1 maps to 1/10. Multiple source decimal tokens
    can round to that same float, so the spelling alone cannot recover them.
    """
    if isinstance(value, bool):
        raise TypeError("a cost cannot be a boolean")
    if isinstance(value, Integral):
        return Fraction(int(value))
    return Fraction(repr(float(value)))


def compute_shortest_decimal_solution_cost(
    instance: LoadedStaticInstance,
    solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]],
) -> Fraction:
    """Exact rational routing cost: the sum of ``shortest_decimal_cost_value`` over every arc, depot to depot.

    Independent of route order and, on a symmetric matrix, of route direction.
    No feasibility check.
    """
    _require_static_matrix(instance)
    total = Fraction(0)
    depot = instance.depot
    for route in _iter_routes(solution_or_routes):
        previous_node = depot
        for customer in route:
            total += shortest_decimal_cost_value(instance.arc_costs[previous_node][customer])
            previous_node = customer
        total += shortest_decimal_cost_value(instance.arc_costs[previous_node][depot])
    return total


def check_solution(
    instance: LoadedStaticInstance,
    solution: BenchmarkSolution | BenchmarkBKS,
    *,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | None = None,
    instance_path: str | Path | None = None,
) -> SolutionCheckResult:
    """Check an embedded static instance under binary64-exact-v1. Supply its path to verify a certified BKS.

    Raises ``ValueError`` when the solution names another instance or declares
    another objective, since no check status describes a mismatched submission.
    """
    from mamut_routing_lib.solutions import _check_solution_model

    _require_static_matrix(instance)
    return _check_solution_model(instance, instance_path, instance, solution, objective_function, backend)


def check_cvrp_solution(
    instance: BenchmarkInstanceCVRP,
    solution: BenchmarkSolution | BenchmarkBKS,
    *,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | None = None,
    instance_path: str | Path | None = None,
) -> SolutionCheckResult:
    """Check an embedded CVRP solution under the family profile."""
    if not isinstance(instance, BenchmarkInstanceCVRP):
        raise TypeError("check_cvrp_solution requires an embedded CVRP matrix.")
    return check_solution(instance, solution, backend=backend, objective_function=objective_function, instance_path=instance_path)


def check_vrptw_solution(
    instance: BenchmarkInstanceVRPTW,
    solution: BenchmarkSolution | BenchmarkBKS,
    *,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | None = None,
    instance_path: str | Path | None = None,
) -> SolutionCheckResult:
    """Check an embedded VRPTW solution under the family profile."""
    if not isinstance(instance, BenchmarkInstanceVRPTW):
        raise TypeError("check_vrptw_solution requires an embedded VRPTW matrix with time windows.")
    return check_solution(instance, solution, backend=backend, objective_function=objective_function, instance_path=instance_path)


def get_objective_tuple(
    routes: list[list[int]],
    cost: int | float | Fraction,
    objective_function: ObjectiveFunction,
) -> tuple[int | float | Fraction, ...]:
    """The comparison key of a solution under ``objective_function``: ``(cost,)`` or ``(num_routes, cost)`` for the hierarchical objective."""
    if objective_function in (
        ObjectiveFunction.MONO_COST,
        ObjectiveFunction.DURATION,
        ObjectiveFunction.FLEET_COST_DURATION,
    ):
        # Mono-objective: for FleetCostDuration the vehicle count is already
        # priced into the cost (duration + fleet_fixed_cost * num_routes), so
        # there is no lexicographic tier.
        return (cost,)
    return (len(routes), cost)


def is_better_solution(
    candidate_routes: list[list[int]],
    candidate_cost: int | float | Fraction,
    existing_routes: list[list[int]],
    existing_cost: int | float | Fraction,
    objective_function: ObjectiveFunction,
) -> bool:
    """True when the candidate strictly precedes the existing solution under ``objective_function`` (see ``get_objective_tuple``).

    Pass the authoritative rational costs to compare under an exact family profile.
    """
    return get_objective_tuple(candidate_routes, candidate_cost, objective_function) < get_objective_tuple(
        existing_routes,
        existing_cost,
        objective_function,
    )


def is_better_exact(
    candidate_routes: list[list[int]],
    candidate_exact_cost: Fraction,
    existing_routes: list[list[int]],
    existing_exact_cost: Fraction,
    objective_function: ObjectiveFunction,
) -> bool:
    """Compare the given rational costs under the objective, retaining the incumbent on a tie."""
    return get_objective_tuple(candidate_routes, candidate_exact_cost, objective_function) < get_objective_tuple(
        existing_routes,
        existing_exact_cost,
        objective_function,
    )

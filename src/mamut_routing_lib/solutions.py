"""Solutions coming from outside the library: checking (and, below, parsing and submitting).

``check_routes`` prices a route set on an instance with the reference checker,
static or time-dependent: static instances are hydrated from their sidecars
first, TD instances are loaded with their ATF sidecars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType

__all__ = ["check_routes", "is_time_dependent"]


def is_time_dependent(instance: Any) -> bool:
    from mamut_routing_lib.artifacts import instance_problem_type

    return instance_problem_type(instance) in (ProblemType.TDVRP, ProblemType.TDVRPTW)


def check_routes(
    instance: Any,
    instance_path: str | Path,
    routes: list[list[int]],
    objective_function: ObjectiveFunction | None = None,
    *,
    declared_cost: int | float | None = None,
) -> Any:
    """Check ``routes`` (library ids, depot implicit) on the instance at ``instance_path``.

    Returns a ``SolutionCheckResult`` (static) or ``TDSolutionCheckResult`` (TD); both expose
    ``status``, ``routing_cost``, ``num_routes``, ``error_message`` and ``is_valid()``. A
    ``declared_cost`` makes the checker refuse any deviation (``objective_value_mismatch``).
    """
    from mamut_routing_lib.artifacts import get_instance_identifier
    from mamut_routing_lib.models import BenchmarkSolution

    path = Path(instance_path)
    try:
        candidate = BenchmarkSolution(
            instance_name=get_instance_identifier(instance),
            routes=[[int(stop) for stop in route] for route in routes],
            cost=declared_cost,
            metadata={},
        )
    except ValueError as error:  # pydantic: a route visits a customer twice, or a bad id
        return _invalid(instance, error)
    if is_time_dependent(instance):
        from mamut_routing_lib.td import check_td_solution, load_td_instance

        objective = objective_function or ObjectiveFunction.DURATION
        loaded = load_td_instance(path)
        return check_td_solution(loaded, candidate, objective_function=objective)

    from mamut_routing_lib.artifacts import hydrate_collection_instance
    from mamut_routing_lib.checker import check_solution

    checkable = hydrate_collection_instance(instance, path)
    return check_solution(checkable, candidate)


def _invalid(instance: Any, error: Exception) -> Any:
    """The checker's rejection for a route set the solution model refuses to build."""
    from mamut_routing_lib.checker import SolutionCheckResult, SolutionCheckStatus

    message = str(error)
    if "elementary" in message:
        status = SolutionCheckStatus.CUSTOMER_SERVED_MULTIPLE_TIMES
        message = "a route visits a customer more than once"
    else:
        status = SolutionCheckStatus.INVALID_CUSTOMER_INDEX
    if is_time_dependent(instance):
        from mamut_routing_lib.td.checker import TDSolutionCheckResult

        return TDSolutionCheckResult.make_invalid(status, message)
    return SolutionCheckResult.make_invalid(status, message)

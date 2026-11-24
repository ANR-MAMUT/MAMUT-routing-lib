"""Time-dependent checking under the td-binary64-exact-v1 family profile."""

from __future__ import annotations

from fractions import Fraction

from pydantic import BaseModel, ConfigDict, field_serializer

from mamut_routing_lib.checker import SolutionCheckResult, canonical_route_order
from mamut_routing_lib.enums import ObjectiveFunction, SolutionCheckStatus
from mamut_routing_lib.exact_types import TD_PROFILE, ExactBackend, ExactRouteError
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution
from mamut_routing_lib.td.artifacts import InstanceATFs, LoadedTDInstance
from mamut_routing_lib.td.models import AnyTDBenchmarkInstance


TD_CHECKER_CONTRACT = TD_PROFILE
TD_OBJECTIVES = (ObjectiveFunction.DURATION, ObjectiveFunction.FLEET_COST_DURATION)


class TDRouteEvaluation(BaseModel):
    """Exact duration and earliest minimizing departure of one feasible route."""

    model_config = ConfigDict(extra="forbid")
    route: list[int]
    feasible: bool
    duration: Fraction | None = None
    departure_time: Fraction | None = None

    @field_serializer("duration", "departure_time", when_used="json")
    def _display_time(self, time: Fraction | None) -> float | None:
        # Both times are bounded by finite materialized time operands. Full results
        # carry the reduced integer pairs alongside these compatibility displays.
        return float(time) if time is not None else None


class TDSolutionCheckResult(SolutionCheckResult):
    """The checked solution and its route witnesses in canonical route order."""

    route_evaluations: list[TDRouteEvaluation] = []


def compute_route_duration(
    instance: AnyTDBenchmarkInstance,
    atfs: InstanceATFs,
    route: list[int],
    *,
    backend: ExactBackend = "native",
) -> TDRouteEvaluation:
    """Compute the minimum duration and earliest attaining departure of a single route."""
    from mamut_routing_lib.exact_td import prepare_td

    checker = prepare_td(instance, atfs, backend=backend, objective_function=ObjectiveFunction.DURATION)
    try:
        witness = checker.route_timing(route)
    except ExactRouteError as error:
        if error.status not in (SolutionCheckStatus.TIME_WINDOW_VIOLATED, SolutionCheckStatus.ROUTE_TIMING_INFEASIBLE):
            raise
        return TDRouteEvaluation(route=route, feasible=False)
    return TDRouteEvaluation(route=route, feasible=True, duration=witness.duration, departure_time=witness.departure_time)


def compute_solution_cost(
    instance: AnyTDBenchmarkInstance,
    atfs: InstanceATFs,
    routes: list[list[int]],
    objective_function: ObjectiveFunction | None = None,
    *,
    backend: ExactBackend = "native",
) -> Fraction:
    """Price a complete TD solution under its family's default objective or the specified objective."""
    from mamut_routing_lib.exact_td import prepare_td

    return prepare_td(instance, atfs, backend=backend, objective_function=objective_function).evaluate(routes).cost


def check_td_solution(
    loaded: LoadedTDInstance,
    solution: BenchmarkSolution | BenchmarkBKS,
    objective_function: ObjectiveFunction | None = None,
    *,
    backend: ExactBackend = "native",
) -> TDSolutionCheckResult:
    """Check the loaded TD solution with its family objective and the selected exact backend."""
    from mamut_routing_lib.solutions import _check_solution_model

    # A certified result asserts the identity of the on-disk edition. An ordinary
    # solution is evaluated directly on the supplied materialized operands.
    path = loaded.instance_path if isinstance(solution, BenchmarkBKS) and "standard" in solution.metadata else None
    return _check_solution_model(loaded.instance, path, loaded, solution, objective_function, backend)

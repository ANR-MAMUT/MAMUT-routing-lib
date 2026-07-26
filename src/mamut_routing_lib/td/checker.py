"""Canonical checker for TDVRPTW / TDVRP solutions (Duration objectives).

The checker is the authoritative definition of the TD objectives: BKS costs
are whatever this pure-Python, fully deterministic implementation computes on
the canonical instance artifacts. All arithmetic is plain IEEE-754 double
precision with exact comparisons — no epsilon thresholds anywhere.

Two objectives are supported. ``Duration`` is the historical one: the sum of
per-route optimal durations. ``FleetCostDuration`` (Plan 11, Blauth2024) adds
a per-used-vehicle fixed cost: the same canonical-order duration fold, then a
single ``+ fleet_fixed_cost * num_routes`` (one IEEE-754 multiply and one
add), where ``fleet_fixed_cost`` is the instance's normative field in the
instance's time unit. Scoring ``FleetCostDuration`` on an instance without
that field raises; scoring ``Duration`` on an instance that carries it is
legal (the field is simply ignored).

Route evaluation composes, sequentially and left-to-right, the arc
arrival-time functions ``α`` and vertex ready-time functions ``θ`` into the
route ready-time function ``δ_r`` (Lera-Romero 2020, Visser & Spliet 2020).
Time-window feasibility falls out of domain restriction during composition:
an empty domain means the route is infeasible. The optimal route duration is
``min_t (δ_r(t) - t)``, attained at a breakpoint.

The total solution cost sums per-route durations in canonical route order
(routes sorted by their first customer), because floating-point addition is
order-sensitive and the canonical order makes the total reproducible.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mamut_routing_lib.checker import SolutionCheckStatus
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution
from mamut_routing_lib.td.artifacts import InstanceATFs, LoadedTDInstance
from mamut_routing_lib.td.models import AnyTDBenchmarkInstance, BenchmarkInstanceTDVRPTW
from mamut_routing_lib.td.pwlf import NDCPWLF, make_theta

#: Objectives the TD checker can score. Everything else is a static-checker
#: concern and is refused loudly.
TD_OBJECTIVES = (ObjectiveFunction.DURATION, ObjectiveFunction.FLEET_COST_DURATION)


def _fleet_fixed_cost_for(
    instance: AnyTDBenchmarkInstance,
    objective_function: ObjectiveFunction,
) -> float | None:
    """Resolve the fixed-cost term of ``objective_function`` on ``instance``.

    Returns ``None`` for ``Duration`` (no term), the instance's
    ``fleet_fixed_cost`` for ``FleetCostDuration``, and raises on any other
    objective or when the required field is missing.
    """
    if objective_function == ObjectiveFunction.DURATION:
        return None
    if objective_function == ObjectiveFunction.FLEET_COST_DURATION:
        if instance.fleet_fixed_cost is None:
            raise ValueError(
                f"Instance {instance.instance_name} has no fleet_fixed_cost; "
                "the FleetCostDuration objective requires it"
            )
        return float(instance.fleet_fixed_cost)
    raise ValueError(f"The TD checker only scores {[o.value for o in TD_OBJECTIVES]}, got {objective_function!r}")


class TDRouteEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: list[int]
    feasible: bool
    duration: float | None = None
    departure_time: float | None = None


class TDSolutionCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SolutionCheckStatus
    routing_cost: float | None
    num_routes: int | None
    error_message: str = ""
    route_evaluations: list[TDRouteEvaluation] = []

    def is_valid(self) -> bool:
        return self.status == SolutionCheckStatus.VALID

    @classmethod
    def make_invalid(cls, status: SolutionCheckStatus, error_message: str) -> "TDSolutionCheckResult":
        return cls(status=status, routing_cost=None, num_routes=None, error_message=error_message)


def _vertex_time_window(
    instance: AnyTDBenchmarkInstance,
    vertex: int,
) -> tuple[float, float] | None:
    if isinstance(instance, BenchmarkInstanceTDVRPTW):
        earliest, latest = instance.time_windows[vertex]
        return float(earliest), float(latest)
    return None


def compute_route_ready_time_function(
    instance: AnyTDBenchmarkInstance,
    atfs: InstanceATFs,
    route: list[int],
) -> NDCPWLF:
    """Route ready-time function ``δ_r`` over feasible depot departure times.

    Returns an empty function when the route is time-infeasible.
    """
    horizon_start, horizon_end = atfs.horizon
    depot = instance.depot

    depot_tw = _vertex_time_window(instance, depot)
    if depot_tw is not None:
        departure_low = max(horizon_start, depot_tw[0])
        departure_high = min(horizon_end, depot_tw[1])
    else:
        departure_low, departure_high = horizon_start, horizon_end
    if departure_low > departure_high:
        return NDCPWLF.empty()

    acc = NDCPWLF.identity(departure_low, departure_high)
    previous = depot
    for vertex in route:
        acc = atfs.arcs[(previous, vertex)].compose(acc)
        if acc.is_empty():
            return acc
        service_time = float(instance.service_times[vertex])
        time_window = _vertex_time_window(instance, vertex)
        if time_window is not None:
            theta = make_theta(time_window[0], time_window[1], service_time)
        else:
            upper = acc.max_image
            theta = NDCPWLF([0.0, upper], [service_time, upper + service_time])
        acc = theta.compose(acc)
        if acc.is_empty():
            return acc
        previous = vertex

    acc = atfs.arcs[(previous, depot)].compose(acc)
    if acc.is_empty():
        return acc
    if depot_tw is not None:
        # Restrict the return arrival to the depot due date, without any
        # waiting clamp: the route ends upon arrival.
        acc = NDCPWLF.identity(0.0, depot_tw[1]).compose(acc)
    return acc


def compute_route_duration(
    instance: AnyTDBenchmarkInstance,
    atfs: InstanceATFs,
    route: list[int],
) -> TDRouteEvaluation:
    """Optimal Duration ``Δ*_r`` and earliest optimal depot departure ``t*_r``."""
    delta = compute_route_ready_time_function(instance, atfs, route)
    if delta.is_empty():
        return TDRouteEvaluation(route=route, feasible=False)
    duration, departure = delta.min_shifted_image()
    return TDRouteEvaluation(route=route, feasible=True, duration=duration, departure_time=departure)


def canonical_route_order(routes: list[list[int]]) -> list[list[int]]:
    return sorted(routes, key=lambda route: route[0])


def compute_solution_cost(
    instance: AnyTDBenchmarkInstance,
    atfs: InstanceATFs,
    routes: list[list[int]],
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
) -> float:
    """Total cost under ``objective_function``, canonical route order.

    Duration: sum of per-route durations. FleetCostDuration: the same fold,
    then a single ``+ fleet_fixed_cost * len(routes)``. Raises on infeasible
    routes and on a missing ``fleet_fixed_cost``.
    """
    fleet_fixed_cost = _fleet_fixed_cost_for(instance, objective_function)
    total = 0.0
    for route in canonical_route_order(routes):
        evaluation = compute_route_duration(instance, atfs, route)
        if not evaluation.feasible:
            raise ValueError(f"route {route} is time-infeasible")
        assert evaluation.duration is not None
        total += evaluation.duration
    if fleet_fixed_cost is not None:
        total += fleet_fixed_cost * len(routes)
    return total


def check_td_solution(
    loaded: LoadedTDInstance,
    solution: BenchmarkSolution | BenchmarkBKS,
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
) -> TDSolutionCheckResult:
    instance = loaded.instance
    atfs = loaded.atfs
    routes = solution.routes
    is_tdvrptw = isinstance(instance, BenchmarkInstanceTDVRPTW)

    # Contract misuse guards, not solution defects: raise instead of returning
    # an invalid result.
    fleet_fixed_cost = _fleet_fixed_cost_for(instance, objective_function)
    declared_objective = getattr(solution, "objective_function", None)
    if declared_objective is not None and declared_objective != objective_function:
        raise ValueError(
            f"Solution declares objective {declared_objective!r} but the check "
            f"was requested for {objective_function!r}"
        )

    served_customers: set[int] = set()
    for route in routes:
        current_load = 0
        for customer in route:
            if customer < 1 or customer > instance.num_customers:
                return TDSolutionCheckResult.make_invalid(
                    SolutionCheckStatus.INVALID_CUSTOMER_INDEX,
                    f"Invalid customer index: {customer}",
                )
            if customer in served_customers:
                return TDSolutionCheckResult.make_invalid(
                    SolutionCheckStatus.CUSTOMER_SERVED_MULTIPLE_TIMES,
                    f"Customer {customer} served more than once.",
                )
            served_customers.add(customer)
            current_load += instance.demands[customer]
            if current_load > instance.vehicle_capacity:
                return TDSolutionCheckResult.make_invalid(
                    SolutionCheckStatus.VEHICLE_CAPACITY_EXCEEDED,
                    f"Vehicle capacity exceeded on route: {route}",
                )

    missing_customers = set(range(1, instance.num_customers + 1)) - served_customers
    if missing_customers:
        return TDSolutionCheckResult.make_invalid(
            SolutionCheckStatus.NOT_ALL_CUSTOMERS_SERVED,
            f"Not all customers served. Missing: {sorted(missing_customers)}",
        )

    if instance.num_vehicles is not None and len(routes) > instance.num_vehicles:
        return TDSolutionCheckResult.make_invalid(
            SolutionCheckStatus.TOO_MANY_VEHICLES_USED,
            "Number of routes exceeds the declared number of vehicles.",
        )

    evaluations: list[TDRouteEvaluation] = []
    for route in routes:
        evaluation = compute_route_duration(instance, atfs, route)
        if not evaluation.feasible:
            status = (
                SolutionCheckStatus.TIME_WINDOW_VIOLATED
                if is_tdvrptw
                else SolutionCheckStatus.ROUTE_TIMING_INFEASIBLE
            )
            return TDSolutionCheckResult.make_invalid(
                status,
                f"Route has no feasible depot departure time: {route}",
            )
        evaluations.append(evaluation)

    total = 0.0
    for evaluation in sorted(evaluations, key=lambda item: item.route[0]):
        assert evaluation.duration is not None
        total += evaluation.duration
    if fleet_fixed_cost is not None:
        total += fleet_fixed_cost * len(routes)

    if solution.cost is not None and solution.cost != total:
        return TDSolutionCheckResult.make_invalid(
            SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
            f"Provided cost {solution.cost!r} does not match computed "
            f"{objective_function.value} {total!r}.",
        )

    return TDSolutionCheckResult(
        status=SolutionCheckStatus.VALID,
        routing_cost=total,
        num_routes=len(routes),
        route_evaluations=evaluations,
    )

"""Evaluation of materialized TD routes under ``td-binary64-exact-v1``."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from collections.abc import Callable, Mapping, Sequence

from mamut_routing_lib.checker import SolutionCheckStatus
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_types import (
    TD_PROFILE, ExactBackend, ExactEvaluation, ExactRouteError, ExactRouteEvaluation, binary64_operand,
    canonical_checked_routes, exact_resource, nonnegative_integer,
)
from mamut_routing_lib.objectives import base_objective, require_supported_objective
from mamut_routing_lib.td.artifacts import InstanceATFs
from mamut_routing_lib.td.models import AnyTDBenchmarkInstance, BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW


Arc = tuple[tuple[float, ...], tuple[float, ...]]
Window = tuple[float, float]


def _window(bounds) -> Window:
    low, high = map(binary64_operand, bounds)
    if high < low:
        raise ValueError("inverted time interval")
    return low, high


def _snapshot_arcs(atfs: InstanceATFs, count: int, horizon: Window) -> Mapping[tuple[int, int], Arc]:
    arcs = {}
    for (tail, head), arc in atfs.arcs.items():
        if type(tail) is not int or type(head) is not int or not (0 <= tail <= count and 0 <= head <= count) or tail == head:
            raise ValueError(f"invalid directed TD arc ({tail}, {head})")
        xs = tuple(map(binary64_operand, arc.xs))
        ys = tuple(map(binary64_operand, arc.ys))
        if len(xs) < 2 or len(xs) != len(ys) or (xs[0], xs[-1]) != horizon:
            raise ValueError(f"invalid knot arrays or horizon on TD arc ({tail}, {head})")
        previous_x, previous_y = xs[0], ys[0]
        for x, y in zip(xs, ys):
            if x < previous_x or y < previous_y or y < x:
                raise ValueError(f"non-FIFO or negative travel on TD arc ({tail}, {head})")
            previous_x, previous_y = x, y
        arcs[tail, head] = xs, ys
    return MappingProxyType(arcs)


def _solver(backend: str) -> tuple[str, Callable]:
    if backend == "python":
        from mamut_routing_lib.td._fraction_partition import solve
        return "python-fraction-partition", solve
    if backend != "native":
        raise ValueError(f"unknown TD exact backend {backend!r}")
    try:
        from mamut_routing_lib._td_exact import PreparedRoute
    except ImportError as exc:
        raise ImportError("The native TD backend needs the compiled _td_exact extension and GMP. "
                          "Install the native wheel or select backend='python'.") from exc

    def solve(*args):
        result = PreparedRoute(*args).solve()
        if result is None:
            return None
        duration, departure = result
        return Fraction(*duration), Fraction(*departure)

    return "cpp-gmp-compose", solve


@dataclass(frozen=True, slots=True)
class PreparedTD:
    """Immutable snapshot of the materialized inputs, reusable across submissions."""

    num_customers: int
    demands: tuple[int, ...]
    capacity: int
    fleet_limit: int | None
    horizon: Window
    windows: tuple[Window, ...] | None
    services: tuple[float, ...]
    arcs: Mapping[tuple[int, int], Arc]
    objective: ObjectiveFunction
    backend: str
    fixed_charge: Fraction
    _solve: Callable = field(repr=False, compare=False)

    def _route(self, route: tuple[int, ...]) -> ExactRouteEvaluation:
        vertices = (0, *route, 0)
        try:
            arcs = [self.arcs[tail, head] for tail, head in zip(vertices, vertices[1:])]
        except KeyError as exc:
            raise ValueError(f"TD arc {exc.args[0]} was not materialized for this route") from exc
        low, high = self.horizon
        close = None
        windows = [None] * len(route)
        if self.windows is not None:
            opens, close = self.windows[0]
            low, high = max(low, opens), min(high, close)
            windows = [self.windows[v] for v in route]
        pair = self._solve(arcs, windows, [self.services[v] for v in route], low, high, close)
        if pair is None:
            status = (SolutionCheckStatus.ROUTE_TIMING_INFEASIBLE if self.windows is None
                      else SolutionCheckStatus.TIME_WINDOW_VIOLATED)
            raise ExactRouteError(status, f"route {list(route)} has no feasible depot departure")
        return ExactRouteEvaluation(duration=pair[0], departure_time=pair[1])

    def evaluate(self, routes: Sequence[Sequence[int]]) -> ExactEvaluation:
        ordered = canonical_checked_routes(routes, num_customers=self.num_customers, demands=self.demands,
                                          capacity=self.capacity, fleet_limit=self.fleet_limit)
        evaluations = tuple(self._route(route) for route in ordered)
        cost = sum((witness.duration for witness in evaluations), start=self.fixed_charge * len(ordered))
        return ExactEvaluation(profile=TD_PROFILE, backend=self.backend, objective=self.objective,
                               routes=ordered, cost=cost, route_evaluations=evaluations)


def prepare_td(instance: AnyTDBenchmarkInstance, atfs: InstanceATFs, *, backend: ExactBackend = "native",
               objective_function: ObjectiveFunction | str | None = None) -> PreparedTD:
    """Prepare an instance with full ATFs or with the arc subset materialized for specific routes.

    ``native`` uses GMP prefix composition. ``python`` partitions departing times with Fraction.
    Evaluating a new route requires every directed arc in that route to be present in the snapshot.
    """
    if not isinstance(instance, (BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW)):
        raise ValueError("prepare_td requires a TDVRP or TDVRPTW instance")
    count = nonnegative_integer(instance.num_customers, "num_customers")
    if type(instance.depot) is not int or instance.depot != 0:
        raise ValueError("expected depot 0")
    if instance.num_vehicles is not None:
        nonnegative_integer(instance.num_vehicles, "num_vehicles")
    capacity = exact_resource(instance.vehicle_capacity, "vehicle_capacity")
    demands = tuple(exact_resource(value, "demand") for value in instance.demands)
    services = tuple(map(binary64_operand, instance.service_times))
    if len(demands) != count + 1 or len(services) != count + 1 or demands[0] != 0 or services[0] != 0:
        raise ValueError("demand and service vectors must include all vertices and be zero at the depot")
    windows = None
    if isinstance(instance, BenchmarkInstanceTDVRPTW):
        windows = tuple(map(_window, instance.time_windows))
        if len(windows) != count + 1:
            raise ValueError("time windows must include all vertices")
    horizon = _window(instance.horizon)
    if horizon[0] == horizon[1]:
        raise ValueError("TD horizon must have positive length")
    if (atfs.instance_name != instance.instance_name or atfs.benchmark_name != instance.benchmark_name.value
            or atfs.num_customers != count or _window(atfs.horizon) != horizon):
        raise ValueError("ATF metadata disagrees with the instance")
    objective = (base_objective(instance) if objective_function is None else ObjectiveFunction(objective_function))
    require_supported_objective(instance, objective)
    fixed = None if instance.fleet_fixed_cost is None else binary64_operand(instance.fleet_fixed_cost)
    if fixed is None and objective == ObjectiveFunction.FLEET_COST_DURATION:
        raise ValueError("FleetCostDuration requires fleet_fixed_cost")
    charge = Fraction(fixed) if objective == ObjectiveFunction.FLEET_COST_DURATION else Fraction(0)
    name, solve = _solver(backend)
    return PreparedTD(count, demands, capacity, instance.num_vehicles, horizon, windows, services,
                      _snapshot_arcs(atfs, count, horizon), objective, name, charge, solve)

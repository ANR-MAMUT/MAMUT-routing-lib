"""Prepared checkers for the static binary64-exact-v1 profile."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain

from mamut_routing_lib.checker import SolutionCheckStatus
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_types import (
    STATIC_PROFILE, ExactBackend, ExactEvaluation, ExactRouteError, binary64_operand,
    canonical_checked_routes, exact_resource, nonnegative_integer,
)
from mamut_routing_lib.models import BenchmarkInstance, BenchmarkInstanceCVRP
from mamut_routing_lib.objectives import base_objective, require_supported_objective

Routes = tuple[tuple[int, ...], ...]


def _fraction_checker(rows, opens, closes, services, vertices):
    matrix = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != vertices:
            raise ValueError("invalid arc matrix dimensions")
        converted = tuple(Fraction.from_float(binary64_operand(x)) for x in row)
        if converted[i]:
            raise ValueError("nonzero arc matrix diagonal")
        matrix.append(converted)
    opens, closes, services = (tuple(map(Fraction.from_float, v)) for v in (opens, closes, services))

    def evaluate(routes: Routes) -> Fraction | None:
        cost = Fraction(0)
        for route in routes:
            previous = 0
            departure = opens[0] if opens else 0
            for customer in chain(route, (0,)):
                arc = matrix[previous][customer]
                cost += arc
                if opens:
                    start = max(departure + arc, opens[customer])
                    if start > closes[customer]:
                        return None
                    if customer:
                        departure = start + services[customer]
                previous = customer
        return cost

    return evaluate


@dataclass(frozen=True, slots=True)
class PreparedStatic:
    """An owned numerical snapshot, reusable for submissions to the same instance."""

    num_customers: int
    demands: tuple[int, ...]
    capacity: int
    fleet_limit: int | None
    objective: ObjectiveFunction
    backend: str
    _price: Callable[[Routes], Fraction | None] = field(repr=False)

    def evaluate(self, routes: Sequence[Sequence[int]]) -> ExactEvaluation:
        """Verify the submitted routes and price the exact sum of the traversed arcs."""
        canonical = canonical_checked_routes(
            routes, num_customers=self.num_customers, demands=self.demands,
            capacity=self.capacity, fleet_limit=self.fleet_limit,
        )
        cost = self._price(canonical)
        if cost is None:
            raise ExactRouteError(SolutionCheckStatus.TIME_WINDOW_VIOLATED,
                                  "A customer or depot return time window is violated.")
        return ExactEvaluation(
            profile=STATIC_PROFILE, backend=self.backend, objective=self.objective,
            routes=canonical, cost=cost,
        )


def prepare_static(
    instance: BenchmarkInstance | BenchmarkInstanceCVRP,
    *,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | str | None = None,
) -> PreparedStatic:
    """Prepare a hydrated CVRP or VRPTW instance under the exact binary64 profile.

    The native backend stores exact cost and temporal operands on a common
    integer scale. The Python reference stores their exact binary64 fractions.
    """
    if not isinstance(instance, (BenchmarkInstance, BenchmarkInstanceCVRP)):
        raise ValueError("prepare_static requires a materialized static instance with an arc matrix")
    objective = base_objective(instance) if objective_function is None else ObjectiveFunction(objective_function)
    require_supported_objective(instance, objective)
    n = nonnegative_integer(instance.num_customers, "num_customers")
    if type(instance.depot) is not int or instance.depot != 0:
        raise ValueError("the depot must be vertex 0")
    fleet = instance.num_vehicles
    if fleet is not None:
        nonnegative_integer(fleet, "num_vehicles")
    capacity = exact_resource(instance.vehicle_capacity, "vehicle_capacity")
    vertices = n + 1
    if not isinstance(instance.demands, (list, tuple)) or len(instance.demands) != vertices:
        raise ValueError("invalid demands dimensions")
    demands = tuple(exact_resource(x, "demand") for x in instance.demands)
    if demands[0] != 0:
        raise ValueError("depot demand must be zero")
    rows = instance.arc_costs
    if not isinstance(rows, (list, tuple)) or len(rows) != vertices:
        raise ValueError("invalid arc matrix dimensions")

    opens, closes, services = [], [], []
    if isinstance(instance, BenchmarkInstance):
        windows = instance.time_windows
        if not isinstance(windows, (list, tuple)) or len(windows) != vertices:
            raise ValueError("invalid time window dimensions")
        for window in windows:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError("time windows must have an opening and closing time")
            early, late = map(binary64_operand, window)
            if early > late:
                raise ValueError("inverted time window")
            opens.append(early)
            closes.append(late)
        if not isinstance(instance.service_times, (list, tuple)) or len(instance.service_times) != vertices:
            raise ValueError("invalid service time dimensions")
        services = list(map(binary64_operand, instance.service_times))
        if services[0]:
            raise ValueError("depot service time must be zero")

    if backend == "python":
        price = _fraction_checker(rows, opens, closes, services, vertices)
        actual_backend = "python-fraction"
    elif backend == "native":
        try:
            from mamut_routing_lib._static_exact import StaticModel
        except ImportError as exc:
            raise ImportError("Install mamut-routing-lib with its native static extension, "
                              "or select backend='python' for the Fraction reference.") from exc
        model = StaticModel(vertices, rows, opens, closes, services)

        def price(routes: Routes) -> Fraction | None:
            pair = model.evaluate(routes)
            if pair is None:
                return None
            units, scale = pair
            return Fraction(units, 1 << scale)

        actual_backend = "cpp-dyadic-prepared"
    else:
        raise ValueError("static backend must be 'native' or 'python'")
    return PreparedStatic(n, demands, capacity, fleet, objective, actual_backend, price)

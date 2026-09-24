"""The static solution checker: the authority on what a CVRP/VRPTW solution costs.

``check_solution`` validates a solution against an embedded-matrix instance
(customer indices, single service, capacity, time windows with waiting at
``ready`` and hard ``due`` dates, fleet size) and prices it by summing the
instance's ``arc_costs`` along each route, depot to depot, in plain Python
arithmetic (ints stay ints, floats accumulate left to right). Every stored BKS
cost is this value. Slim collection instances must be hydrated first
(``artifacts.hydrate_collection_instance``); time-dependent instances use
``mamut_routing_lib.td.checker``.

``is_better_solution`` defines the order of each objective: ``MonoCost``,
``Duration`` and ``FleetCostDuration`` compare the cost alone,
``HierarchicalVehicleCost`` compares the route count first.

The float total depends on the order in which arcs are added, so the same
route set listed in another order, reversed on a symmetric matrix, or a
different route set with the same exact cost can price a few ulps apart. The
BKS store therefore never ranks solutions on float totals: it compares
``compute_exact_solution_cost`` (the exact rational sum of the decimal arc
costs written in the instance, see ``exact_cost_value``) with
``is_better_exact``, so a tie is a tie. Stored BKS keep their routes in
``canonical_route_order`` and their ``cost`` is the float total in that order.
"""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from numbers import Integral

from pydantic import BaseModel, ConfigDict

from mamut_routing_lib.artifacts import AnyBenchmarkInstance
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkInstance, BenchmarkInstanceCVRP, BenchmarkSolution


class SolutionCheckStatus(str, Enum):
    """Outcome of a check: ``valid`` or the first violation met, in route order.

    The checker stops at the first violation, so a solution with several defects
    reports only one. ``route_timing_infeasible`` is reserved for the
    time-dependent checker.
    """
    VALID = "valid"
    INVALID_CUSTOMER_INDEX = "invalid_customer_index"
    CUSTOMER_SERVED_MULTIPLE_TIMES = "customer_served_multiple_times"
    VEHICLE_CAPACITY_EXCEEDED = "vehicle_capacity_exceeded"
    TIME_WINDOW_VIOLATED = "time_window_violated"
    NOT_ALL_CUSTOMERS_SERVED = "not_all_customers_served"
    TOO_MANY_VEHICLES_USED = "too_many_vehicles_used"
    OBJECTIVE_VALUE_MISMATCH = "objective_value_mismatch"
    ROUTE_TIMING_INFEASIBLE = "route_timing_infeasible"


class SolutionCheckResult(BaseModel):
    """Status plus, when valid, the priced ``routing_cost`` and ``num_routes``; ``error_message`` explains a rejection."""
    model_config = ConfigDict(extra="forbid")

    status: SolutionCheckStatus
    routing_cost: int | float | None
    num_routes: int | None
    error_message: str = ""

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


def compute_route_cost(instance: AnyBenchmarkInstance, route: list[int]) -> int | float:
    """Cost of one route: ``depot -> route[0] -> ... -> route[-1] -> depot`` over ``instance.arc_costs`` (no feasibility check)."""
    total_cost: int | float = 0
    previous_node = instance.depot
    for customer in route:
        total_cost += instance.arc_costs[previous_node][customer]
        previous_node = customer
    total_cost += instance.arc_costs[previous_node][instance.depot]
    return total_cost


def compute_solution_cost(
    instance: AnyBenchmarkInstance,
    solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]],
) -> int | float:
    """Sum of ``compute_route_cost`` over the routes of a solution, BKS or plain route list."""
    return sum(compute_route_cost(instance, route) for route in _iter_routes(solution_or_routes))


def canonical_route_order(routes: list[list[int]]) -> list[list[int]]:
    """Routes sorted by their first customer: the canonical storage and summation order of every BKS."""
    return sorted(routes, key=lambda route: route[0])


def exact_cost_value(value: int | float) -> Fraction:
    """The exact decimal value a stored cost denotes.

    Integers map to themselves; a float maps to the decimal number its
    shortest ``repr`` spells (``0.1`` is one tenth, not the nearest binary
    double). That is the value written in the instance or BKS JSON, and the map
    is injective and strictly increasing on finite doubles, so it never merges
    two distinct costs or swaps their order.
    """
    if isinstance(value, bool):
        raise TypeError("a cost cannot be a boolean")
    if isinstance(value, Integral):
        return Fraction(int(value))
    return Fraction(repr(float(value)))


def compute_exact_solution_cost(
    instance: AnyBenchmarkInstance,
    solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]],
) -> Fraction:
    """Exact rational routing cost: the sum of ``exact_cost_value`` over every arc, depot to depot.

    Independent of route order and, on a symmetric matrix, of route direction.
    No feasibility check.
    """
    total = Fraction(0)
    depot = instance.depot
    for route in _iter_routes(solution_or_routes):
        previous_node = depot
        for customer in route:
            total += exact_cost_value(instance.arc_costs[previous_node][customer])
            previous_node = customer
        total += exact_cost_value(instance.arc_costs[previous_node][depot])
    return total


def _check_common_route_constraints(
    instance: AnyBenchmarkInstance,
    routes: list[list[int]],
    *,
    validate_time_windows: bool,
) -> SolutionCheckResult:
    served_customers: set[int] = set()
    total_cost: int | float = 0

    for route in routes:
        current_load = 0
        current_time = 0
        previous_node = instance.depot

        for customer in route:
            if customer < 1 or customer > instance.num_customers:
                return SolutionCheckResult.make_invalid(
                    SolutionCheckStatus.INVALID_CUSTOMER_INDEX,
                    f"Invalid customer index: {customer}",
                )

            if customer in served_customers:
                return SolutionCheckResult.make_invalid(
                    SolutionCheckStatus.CUSTOMER_SERVED_MULTIPLE_TIMES,
                    f"Customer {customer} served more than once.",
                )
            served_customers.add(customer)

            current_load += instance.demands[customer]
            if current_load > instance.vehicle_capacity:
                return SolutionCheckResult.make_invalid(
                    SolutionCheckStatus.VEHICLE_CAPACITY_EXCEEDED,
                    f"Vehicle capacity exceeded on route: {route}",
                )

            travel_cost = instance.arc_costs[previous_node][customer]
            total_cost += travel_cost

            if validate_time_windows:
                assert isinstance(instance, BenchmarkInstance)
                arrival_time = current_time + travel_cost
                ready_time, due_date = instance.time_windows[customer]
                if arrival_time < ready_time:
                    arrival_time = ready_time
                if arrival_time > due_date:
                    return SolutionCheckResult.make_invalid(
                        SolutionCheckStatus.TIME_WINDOW_VIOLATED,
                        f"Time window violated for customer {customer} on route: {route}",
                    )
                current_time = arrival_time + instance.service_times[customer]

            previous_node = customer

        total_cost += instance.arc_costs[previous_node][instance.depot]

        if validate_time_windows:
            assert isinstance(instance, BenchmarkInstance)
            arrival_time = current_time + instance.arc_costs[previous_node][instance.depot]
            depot_ready_time, depot_due_date = instance.time_windows[instance.depot]
            if arrival_time < depot_ready_time:
                arrival_time = depot_ready_time
            if arrival_time > depot_due_date:
                return SolutionCheckResult.make_invalid(
                    SolutionCheckStatus.TIME_WINDOW_VIOLATED,
                    f"Time window violated when returning to depot on route: {route}",
                )

    missing_customers = set(range(1, instance.num_customers + 1)) - served_customers
    if missing_customers:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.NOT_ALL_CUSTOMERS_SERVED,
            f"Not all customers served. Missing: {sorted(missing_customers)}",
        )

    if instance.num_vehicles is not None and len(routes) > instance.num_vehicles:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.TOO_MANY_VEHICLES_USED,
            "Number of routes exceeds the declared number of vehicles.",
        )

    return SolutionCheckResult.make_valid(routing_cost=total_cost, num_routes=len(routes))


def check_cvrp_solution(
    instance: BenchmarkInstanceCVRP,
    solution: BenchmarkSolution | BenchmarkBKS,
) -> SolutionCheckResult:
    """Validate and price a solution on a CVRP instance (capacity, coverage, fleet; no timing)."""
    result = _check_common_route_constraints(instance, solution.routes, validate_time_windows=False)
    if result.is_valid() and solution.cost is not None and solution.cost != result.routing_cost:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
            "Provided cost does not match computed routing cost.",
        )
    return result


def check_vrptw_solution(
    instance: BenchmarkInstance,
    solution: BenchmarkSolution | BenchmarkBKS,
) -> SolutionCheckResult:
    """Validate and price a solution on a VRPTW instance (capacity, coverage, fleet, time windows with waiting)."""
    result = _check_common_route_constraints(instance, solution.routes, validate_time_windows=True)
    if result.is_valid() and solution.cost is not None and solution.cost != result.routing_cost:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
            "Provided cost does not match computed routing cost.",
        )
    return result


def check_solution(
    instance: AnyBenchmarkInstance,
    solution: BenchmarkSolution | BenchmarkBKS,
) -> SolutionCheckResult:
    """Validate and price ``solution`` on a static instance; the single entry point.

    Dispatches on the instance model. Raises ``TypeError`` for time-dependent
    instances (use ``mamut_routing_lib.td.check_td_solution``) and for slim
    collection instances (hydrate them first).
    """
    if not isinstance(instance, (BenchmarkInstance, BenchmarkInstanceCVRP)):
        raise TypeError(
            "Time-dependent instances require the ATF sidecar; "
            "use mamut_routing_lib.td.check_td_solution instead."
        )
    if isinstance(instance, BenchmarkInstanceCVRP):
        return check_cvrp_solution(instance, solution)
    return check_vrptw_solution(instance, solution)


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
    candidate_cost: int | float,
    existing_routes: list[list[int]],
    existing_cost: int | float,
    objective_function: ObjectiveFunction,
) -> bool:
    """True when the candidate strictly precedes the existing solution under ``objective_function`` (see ``get_objective_tuple``).

    Compares the float costs as given, so it is order-sensitive at the ulp
    level; the BKS store uses :func:`is_better_exact` instead.
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
    """``is_better_solution`` on exact costs (``compute_exact_solution_cost``): ties are never improvements."""
    return get_objective_tuple(candidate_routes, candidate_exact_cost, objective_function) < get_objective_tuple(
        existing_routes,
        existing_exact_cost,
        objective_function,
    )

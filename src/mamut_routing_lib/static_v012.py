"""Replay the static arithmetic of mamut-routing-lib v0.12 on an embedded matrix."""

from __future__ import annotations

from mamut_routing_lib.checker import SolutionCheckResult, SolutionCheckStatus, _iter_routes, _require_static_matrix
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkInstanceVRPTW, BenchmarkInstanceCVRP, BenchmarkSolution, LoadedStaticInstance


def compute_route_cost(instance: LoadedStaticInstance, route: list[int]) -> int | float:
    """Cost of one route: ``depot -> route[0] -> ... -> route[-1] -> depot`` over ``instance.arc_costs`` (no feasibility check)."""
    _require_static_matrix(instance)
    total_cost: int | float = 0
    previous_node = instance.depot
    for customer in route:
        total_cost += instance.arc_costs[previous_node][customer]
        previous_node = customer
    total_cost += instance.arc_costs[previous_node][instance.depot]
    return total_cost


def compute_solution_cost(
    instance: LoadedStaticInstance,
    solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]],
) -> int | float:
    """Sum of ``compute_route_cost`` over the routes of a solution, BKS or plain route list."""
    _require_static_matrix(instance)
    return sum(compute_route_cost(instance, route) for route in _iter_routes(solution_or_routes))


def _evaluate_static_routes(
    instance: LoadedStaticInstance,
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
                assert isinstance(instance, BenchmarkInstanceVRPTW)
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
            assert isinstance(instance, BenchmarkInstanceVRPTW)
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
    if not isinstance(instance, BenchmarkInstanceCVRP):
        raise TypeError("check_cvrp_solution requires an embedded CVRP matrix.")
    result = _evaluate_static_routes(instance, solution.routes, validate_time_windows=False)
    if result.is_valid() and solution.cost is not None and solution.cost != result.routing_cost:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
            "Provided cost does not match computed routing cost.",
        )
    return result


def check_vrptw_solution(
    instance: BenchmarkInstanceVRPTW,
    solution: BenchmarkSolution | BenchmarkBKS,
) -> SolutionCheckResult:
    """Validate and price a solution on a VRPTW instance (capacity, coverage, fleet, time windows with waiting)."""
    if not isinstance(instance, BenchmarkInstanceVRPTW):
        raise TypeError("check_vrptw_solution requires an embedded VRPTW matrix with time windows.")
    result = _evaluate_static_routes(instance, solution.routes, validate_time_windows=True)
    if result.is_valid() and solution.cost is not None and solution.cost != result.routing_cost:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
            "Provided cost does not match computed routing cost.",
        )
    return result


def check_solution(
    instance: LoadedStaticInstance,
    solution: BenchmarkSolution | BenchmarkBKS,
) -> SolutionCheckResult:
    """Validate and price ``solution`` on a static instance; the single entry point.

    Dispatches on the instance model. Raises ``TypeError`` for time-dependent
    instances (use ``mamut_routing_lib.td.check_td_solution``) and for slim
    collection instances (hydrate them first).
    """
    _require_static_matrix(instance)
    if isinstance(instance, BenchmarkInstanceCVRP):
        return check_cvrp_solution(instance, solution)
    return check_vrptw_solution(instance, solution)

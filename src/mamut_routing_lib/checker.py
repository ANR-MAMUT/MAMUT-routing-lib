from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from mamut_routing_lib.artifacts import AnyBenchmarkInstance
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkInstance, BenchmarkInstanceCVRP, BenchmarkInstanceVRPTW, BenchmarkSolution


class SolutionCheckStatus(str, Enum):
    VALID = "valid"
    INVALID_CUSTOMER_INDEX = "invalid_customer_index"
    CUSTOMER_SERVED_MULTIPLE_TIMES = "customer_served_multiple_times"
    VEHICLE_CAPACITY_EXCEEDED = "vehicle_capacity_exceeded"
    TIME_WINDOW_VIOLATED = "time_window_violated"
    NOT_ALL_CUSTOMERS_SERVED = "not_all_customers_served"
    TOO_MANY_VEHICLES_USED = "too_many_vehicles_used"
    OBJECTIVE_VALUE_MISMATCH = "objective_value_mismatch"


class SolutionCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SolutionCheckStatus
    routing_cost: int | float | None
    num_routes: int | None
    error_message: str = ""

    def is_valid(self) -> bool:
        return self.status == SolutionCheckStatus.VALID

    @classmethod
    def make_invalid(cls, status: SolutionCheckStatus, error_message: str) -> "SolutionCheckResult":
        return cls(status=status, routing_cost=None, num_routes=None, error_message=error_message)

    @classmethod
    def make_valid(cls, routing_cost: int | float, num_routes: int) -> "SolutionCheckResult":
        return cls(status=SolutionCheckStatus.VALID, routing_cost=routing_cost, num_routes=num_routes)


def _iter_routes(solution_or_routes: BenchmarkSolution | BenchmarkBKS | list[list[int]]) -> list[list[int]]:
    if isinstance(solution_or_routes, list):
        return solution_or_routes
    return solution_or_routes.routes


def compute_route_cost(instance: AnyBenchmarkInstance, route: list[int]) -> int | float:
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
    return sum(compute_route_cost(instance, route) for route in _iter_routes(solution_or_routes))


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
                assert isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTW))
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
            assert isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTW))
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
    result = _check_common_route_constraints(instance, solution.routes, validate_time_windows=False)
    if result.is_valid() and solution.cost is not None and solution.cost != result.routing_cost:
        return SolutionCheckResult.make_invalid(
            SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
            "Provided cost does not match computed routing cost.",
        )
    return result


def check_vrptw_solution(
    instance: BenchmarkInstance | BenchmarkInstanceVRPTW,
    solution: BenchmarkSolution | BenchmarkBKS,
) -> SolutionCheckResult:
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
    if isinstance(instance, BenchmarkInstanceCVRP):
        return check_cvrp_solution(instance, solution)
    return check_vrptw_solution(instance, solution)


def get_objective_tuple(
    routes: list[list[int]],
    cost: int | float,
    objective_function: ObjectiveFunction,
) -> tuple[int | float, ...]:
    if objective_function == ObjectiveFunction.MONO_COST:
        return (cost,)
    return (len(routes), cost)


def is_better_solution(
    candidate_routes: list[list[int]],
    candidate_cost: int | float,
    existing_routes: list[list[int]],
    existing_cost: int | float,
    objective_function: ObjectiveFunction,
) -> bool:
    return get_objective_tuple(candidate_routes, candidate_cost, objective_function) < get_objective_tuple(
        existing_routes,
        existing_cost,
        objective_function,
    )

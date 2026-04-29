from __future__ import annotations

import pytest

pytest.importorskip("pyvrp")

from mamut_routing_lib.checker import check_solution
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.solvers.pyvrp import solve_cvrp, solve_instance, solve_vrptw


def test_solve_cvrp_returns_feasible_routes(toy_cvrp_instance) -> None:
    result = solve_cvrp(toy_cvrp_instance, time_limit_s=2, seed=1)
    assert result.solver_is_feasible
    assert result.route_count >= 1
    assert result.method == "hgs-v1"
    assert result.objective_function == ObjectiveFunction.MONO_COST.value
    candidate = BenchmarkSolution(instance_name=toy_cvrp_instance.instance_name, routes=result.routes)
    assert check_solution(toy_cvrp_instance, candidate).is_valid()


def test_solve_vrptw_mono_cost_returns_feasible_routes(toy_vrptw_instance) -> None:
    result = solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.MONO_COST,
        time_limit_s=2,
        seed=1,
    )
    assert result.solver_is_feasible
    assert result.route_count >= 1
    assert result.vehicle_penalty == 0
    candidate = BenchmarkSolution(instance_name=toy_vrptw_instance.instance_name, routes=result.routes)
    assert check_solution(toy_vrptw_instance, candidate).is_valid()


def test_solve_vrptw_hierarchical_uses_vehicle_penalty(toy_vrptw_instance) -> None:
    mono = solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.MONO_COST,
        time_limit_s=2,
        seed=1,
    )
    hierarchical = solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.HIERARCHICAL_VEHICLE_COST,
        time_limit_s=2,
        seed=1,
    )
    assert hierarchical.solver_is_feasible
    assert hierarchical.vehicle_penalty > 0
    # Hierarchical should never use more vehicles than mono on the same instance.
    assert hierarchical.route_count <= mono.route_count


def test_solve_instance_dispatches_on_type(toy_cvrp_instance, toy_vrptw_instance) -> None:
    cvrp_result = solve_instance(toy_cvrp_instance, time_limit_s=2, seed=1)
    assert cvrp_result.problem_type == "CVRP"

    vrptw_result = solve_instance(
        toy_vrptw_instance,
        time_limit_s=2,
        seed=1,
        objective_function=ObjectiveFunction.MONO_COST,
    )
    assert vrptw_result.problem_type == "VRPTW"

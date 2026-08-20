from __future__ import annotations

import time

import pytest

pytest.importorskip("pyvrp")

import numpy as np

from mamut_routing_lib.checker import check_solution
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.solvers.pyvrp import (
    SolveMonitor,
    SolveTick,
    solve_cvrp,
    solve_instance,
    solve_vrptw,
)


def test_solve_cvrp_returns_feasible_routes(toy_cvrp_instance) -> None:
    result = solve_cvrp(toy_cvrp_instance, time_limit_s=2, seed=1)
    assert result.solver_is_feasible
    assert result.route_count >= 1
    assert result.method == "pyvrp-ils-v1"
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


def _toy_collection_kwargs() -> dict:
    return {
        "instance_name": "poryos-n3-collection",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Poryos2026",
        "num_customers": 3,
        "vehicle_capacity": 10,
        "coordinates": [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)],
        "demands": [0, 3, 4, 2],
        "depot": 0,
        "metric_variant": "euclidean",
        "arc_costs_source": {"model": "euclidean", "decimals": 3},
        "metadata": {},
    }


def test_solve_cvrp_collection_resolves_euclidean_arcs_and_scales() -> None:
    from mamut_routing_lib.models import BenchmarkInstanceCVRPCollection
    from mamut_routing_lib.solvers.pyvrp import hydrate_collection_instance

    instance = BenchmarkInstanceCVRPCollection(**_toy_collection_kwargs())
    result = solve_instance(instance, time_limit_s=1, seed=1, instance_path="unused-for-euclidean.vrp.json")
    assert result.method == "pyvrp-ils-v1"
    assert result.solver_is_feasible
    assert sorted(stop for route in result.routes for stop in route) == [1, 2, 3]
    assert result.metadata["arc_cost_scale"] == 1000

    hydrated = hydrate_collection_instance(instance, "unused-for-euclidean.vrp.json")
    assert hydrated.arc_costs[0][1] == 100.0


def test_solve_vrptw_collection_dispatches_with_time_windows() -> None:
    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.models import BenchmarkInstanceVRPTWCollection

    kwargs = _toy_collection_kwargs()
    kwargs["instance_name"] = "poryos-n3-collection-vrptw"
    kwargs["service_times"] = [0, 10, 10, 10]
    kwargs["time_windows"] = [(0, 100000), (0, 100000), (0, 100000), (0, 100000)]
    instance = BenchmarkInstanceVRPTWCollection(**kwargs)
    result = solve_instance(
        instance,
        time_limit_s=2,
        seed=1,
        objective_function=ObjectiveFunction.MONO_COST,
        instance_path="unused.vrp.json",
    )
    assert result.method == "pyvrp-ils-v3"
    assert result.solver_is_feasible
    assert sorted(stop for route in result.routes for stop in route) == [1, 2, 3]


def test_collection_with_sidecar_source_requires_instance_path() -> None:
    from mamut_routing_lib.models import BenchmarkInstanceCVRPCollection

    instance = BenchmarkInstanceCVRPCollection(**_toy_collection_kwargs())
    with pytest.raises(ValueError, match="instance_path"):
        solve_instance(instance, time_limit_s=1, seed=1)


def test_solve_monitor_reports_ticks_and_stays_out_of_the_result(toy_vrptw_instance) -> None:
    """The monitor is a reporting hook, not a search parameter: same solve, plus a trace."""
    ticks: list[SolveTick] = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    watched = solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.MONO_COST,
        time_limit_s=2,
        seed=1,
        monitor=monitor,
    )
    plain = solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.MONO_COST,
        time_limit_s=2,
        seed=1,
    )
    assert ticks, "PyVRP calls the stopping criterion every iteration; none were relayed"
    assert watched.routes == plain.routes
    assert not monitor.stopped_early
    assert [tick.iterations for tick in ticks] == sorted(tick.iterations for tick in ticks)
    assert all(tick.elapsed_s >= 0 for tick in ticks)


def test_solve_monitor_stops_the_search_when_the_callback_asks(toy_vrptw_instance) -> None:
    """The cancellation path: returning True ends the solve well inside its time budget."""
    monitor = SolveMonitor(lambda tick: True, interval_s=0.0)
    started = time.perf_counter()
    result = solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.MONO_COST,
        time_limit_s=30,
        seed=1,
        monitor=monitor,
    )
    assert monitor.stopped_early
    assert time.perf_counter() - started < 15, "the monitor did not cut the 30 s budget short"
    assert result is not None


def test_solve_monitor_names_each_hierarchical_phase(toy_vrptw_instance) -> None:
    ticks: list[SolveTick] = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    solve_vrptw(
        toy_vrptw_instance,
        objective_function=ObjectiveFunction.HIERARCHICAL_VEHICLE_COST,
        time_limit_s=3,
        seed=1,
        monitor=monitor,
    )
    assert {"minimising fleet", "optimising cost"} <= {tick.phase for tick in ticks}


def test_solve_monitor_throttles_to_its_interval() -> None:
    """A tick per iteration would rewrite the job record thousands of times a second."""
    ticks: list[SolveTick] = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=3600.0)
    for _ in range(1000):
        assert monitor(10.0) is False
    assert len(ticks) == 1, "only the first call should have got through the interval"
    assert ticks[0].iterations == 1


def test_solve_monitor_reports_no_incumbent_while_infeasible() -> None:
    """PyVRP evaluates an infeasible best at INT_MAX; that must not surface as a cost."""
    ticks: list[SolveTick] = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    monitor(float(np.iinfo(np.int64).max))
    monitor(4200.0)
    assert [tick.best_cost for tick in ticks] == [None, 4200.0]


def test_solve_monitor_divides_by_the_cost_scale() -> None:
    """Collection instances solve in milli-units; the reported incumbent is in real units."""
    ticks: list[SolveTick] = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    monitor.rescale(1000)
    monitor(4200.0)
    assert ticks[0].best_cost == 4.2

"""Solver-agnostic base types: the monitor contract every driver relies on, capability checks."""

from __future__ import annotations

import threading

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers.base import (
    INFEASIBLE_COST,
    InstanceProfile,
    SolveMonitor,
    SolverCapabilities,
    SolveTick,
    capability_support,
    profile_instance,
)


def _ticks() -> tuple[list[SolveTick], SolveMonitor]:
    ticks: list[SolveTick] = []
    return ticks, SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)


def test_observe_reports_elapsed_without_iterations_for_external_drivers() -> None:
    ticks, monitor = _ticks()
    assert monitor.observe() is False
    assert monitor.observe(1234.0) is False
    assert [tick.iterations for tick in ticks] == [None, None]
    assert [tick.best_cost for tick in ticks] == [None, 1234.0]


def test_observe_keeps_the_best_incumbent_between_progress_lines() -> None:
    ticks, monitor = _ticks()
    monitor.observe(50.0)
    monitor.observe()  # a poll with no new progress line still reports the best known cost
    assert ticks[-1].best_cost == 50.0
    assert monitor.best_cost == 50.0


def test_call_counts_iterations_like_the_pyvrp_criterion() -> None:
    ticks, monitor = _ticks()
    for cost in (30.0, 20.0, 25.0):
        monitor(cost)
    assert [tick.iterations for tick in ticks] == [1, 2, 3]
    assert monitor.best_cost == 20.0


def test_request_stop_is_seen_on_the_next_observation_even_when_throttled() -> None:
    monitor = SolveMonitor(lambda tick: False, interval_s=3600.0)
    assert monitor.observe(1.0) is False
    stopper = threading.Thread(target=monitor.request_stop)
    stopper.start()
    stopper.join()
    assert monitor.stop_requested
    assert monitor.observe(1.0) is True
    assert monitor.stopped_early


def test_callback_returning_true_latches_the_stop() -> None:
    monitor = SolveMonitor(lambda tick: True, interval_s=0.0)
    assert monitor.observe() is True
    assert monitor.stop_requested and monitor.stopped_early


def test_trajectory_records_improvements_and_merges_close_points() -> None:
    monitor = SolveMonitor(None, record_trajectory=True, trajectory_min_gap_s=3600.0)
    monitor.observe(100.0)
    monitor.observe(90.0)
    monitor.observe(95.0)  # not an improvement
    assert len(monitor.trajectory) == 1
    assert monitor.trajectory[0][1] == 90.0


def test_infeasible_sentinel_and_scale() -> None:
    ticks, monitor = _ticks()
    monitor.rescale(1000)
    monitor(INFEASIBLE_COST)
    monitor(4200.0)
    assert [tick.best_cost for tick in ticks] == [None, 4.2]


def test_phase_change_restarts_the_running_best() -> None:
    _, monitor = _ticks()
    monitor.observe(10.0)
    monitor.phase("optimising cost")
    monitor.observe(500.0)
    assert monitor.best_cost == 500.0


def _profile(**overrides) -> InstanceProfile:
    values = dict(
        problem_type=ProblemType.CVRP,
        benchmark_name="Mamut2026",
        metric_variant="fastest",
        num_customers=10,
        depot=0,
        fixed_fleet=None,
        has_coordinates=True,
        coordinates_define_costs=False,
        symmetric=False,
    )
    values.update(overrides)
    return InstanceProfile(**values)


_COORDS = SolverCapabilities(
    objectives={ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST})},
    input="coordinates",
    symmetric_only=True,
    fleet_limit=False,
)


def test_capability_support_refuses_problem_type_and_objective() -> None:
    td = capability_support(_COORDS, _profile(problem_type=ProblemType.TDVRP), ObjectiveFunction.DURATION, solver_label="X")
    assert not td.supported and "TDVRP" in td.reasons[0]
    hier = capability_support(_COORDS, _profile(), ObjectiveFunction.HIERARCHICAL_VEHICLE_COST, solver_label="X")
    assert not hier.supported and "HierarchicalVehicleCost" in hier.reasons[0]


def test_capability_support_needs_symmetry_and_coordinate_costs_for_coordinate_solvers() -> None:
    road = capability_support(_COORDS, _profile(), ObjectiveFunction.MONO_COST, solver_label="X")
    assert not road.supported
    assert any("symmetric" in reason for reason in road.reasons)
    assert any("coordinates" in reason for reason in road.reasons)
    euclid = capability_support(
        _COORDS,
        _profile(metric_variant="euclidean", symmetric=True, coordinates_define_costs=True, fixed_fleet=12),
        ObjectiveFunction.MONO_COST,
        solver_label="X",
    )
    assert euclid.supported
    assert euclid.warnings and "fixed fleet" in euclid.warnings[0]


def test_profile_instance_on_embedded_matrices(toy_cvrp_instance, toy_vrptw_instance) -> None:
    profile = profile_instance(toy_cvrp_instance)
    assert profile.problem_type is ProblemType.CVRP
    assert profile.symmetric is True  # the toy matrix is symmetric
    assert profile.coordinates_define_costs is False  # fastest metric
    assert profile_instance(toy_vrptw_instance).problem_type is ProblemType.VRPTW


def test_a_cancel_poll_stops_the_solve_without_a_tick() -> None:
    from mamut_routing_lib.solvers.base import SolveMonitor

    flag = {"cancelled": False}
    ticks: list = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=3600.0, cancelled=lambda: flag["cancelled"])
    assert not monitor.observe(10.0) and not monitor.stop_requested
    flag["cancelled"] = True
    assert monitor.stop_requested
    assert monitor.observe(9.0) and monitor.stopped_early
    flag["cancelled"] = False
    assert monitor.stop_requested  # a stop latches

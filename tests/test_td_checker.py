from __future__ import annotations

import pytest

from mamut_routing_lib.checker import SolutionCheckStatus
from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.td import (
    check_td_solution,
    compute_route_duration,
    load_td_instance,
    td_instance_from_payload,
)
from td_utils import make_toy_atfs, toy_instance_payload, write_toy_instance_files


@pytest.fixture()
def toy_loaded(tmp_path):
    instance_path = write_toy_instance_files(tmp_path)
    return load_td_instance(instance_path)


@pytest.fixture()
def toy_loaded_tdvrp(tmp_path):
    instance_path = write_toy_instance_files(tmp_path, with_time_windows=False)
    return load_td_instance(instance_path)


class TestRouteDuration:
    def test_constant_arcs_route(self, toy_loaded):
        # 0 -> 2 -> 0 with constant arcs 15 and 10: duration 25 from any departure.
        evaluation = compute_route_duration(toy_loaded.instance, toy_loaded.atfs, [2])
        assert evaluation.feasible
        assert evaluation.duration == 25.0
        assert evaluation.departure_time == 0.0

    def test_td_arc_optimal_departure_is_delayed(self, toy_loaded):
        # 0 -> 1 -> 2 -> 0. Departing at t: arrive 1 at t+10; arc (1, 2) departing
        # u <= 50 arrives at 30 + 0.6 u, so waiting pays: from t = 40 (u = 50)
        # onward the travel is fluid and the total duration is 10 + 10 + 10 = 30.
        evaluation = compute_route_duration(toy_loaded.instance, toy_loaded.atfs, [1, 2])
        assert evaluation.feasible
        assert evaluation.duration == 30.0
        assert evaluation.departure_time == 40.0

    def test_time_window_forces_waiting(self, toy_loaded):
        payload = toy_instance_payload()
        payload["time_windows"] = [[0, 100], [0, 100], [50, 100]]
        payload["service_times"] = [0, 0, 5]
        instance = td_instance_from_payload(payload)
        # 0 -> 2 -> 0: arrive at 2 at t+15, wait until 50, serve 5, return 10.
        # Duration = 65 - t, decreasing until t = 35, then constant 30.
        evaluation = compute_route_duration(instance, toy_loaded.atfs, [2])
        assert evaluation.feasible
        assert evaluation.duration == 30.0
        assert evaluation.departure_time == 35.0

    def test_unreachable_time_window_is_infeasible(self, toy_loaded):
        payload = toy_instance_payload()
        payload["time_windows"] = [[0, 100], [0, 100], [0, 10]]
        instance = td_instance_from_payload(payload)
        evaluation = compute_route_duration(instance, toy_loaded.atfs, [2])
        assert not evaluation.feasible

    def test_tdvrp_has_no_time_windows(self, toy_loaded_tdvrp):
        evaluation = compute_route_duration(
            toy_loaded_tdvrp.instance, toy_loaded_tdvrp.atfs, [1, 2]
        )
        assert evaluation.feasible
        assert evaluation.duration == 30.0


class TestCheckTDSolution:
    def test_valid_solution(self, toy_loaded):
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1, 2]], cost=30.0)
        result = check_td_solution(toy_loaded, solution)
        assert result.is_valid()
        assert result.routing_cost == 30.0
        assert result.num_routes == 1

    def test_valid_two_routes_costs_sum_canonically(self, toy_loaded):
        # [2] alone: 15 + 10 = 25; [1] alone: 10 + 10 = 20. Total 45.
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[2], [1]], cost=45.0)
        result = check_td_solution(toy_loaded, solution)
        assert result.is_valid()
        assert result.routing_cost == 45.0

    def test_cost_mismatch_detected(self, toy_loaded):
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1, 2]], cost=29.999999)
        result = check_td_solution(toy_loaded, solution)
        assert result.status == SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH

    def test_missing_customer_detected(self, toy_loaded):
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1]])
        result = check_td_solution(toy_loaded, solution)
        assert result.status == SolutionCheckStatus.NOT_ALL_CUSTOMERS_SERVED

    def test_duplicate_customer_detected(self, toy_loaded):
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1], [1, 2]])
        result = check_td_solution(toy_loaded, solution)
        assert result.status == SolutionCheckStatus.CUSTOMER_SERVED_MULTIPLE_TIMES

    def test_invalid_index_detected(self, toy_loaded):
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1, 2, 3]])
        result = check_td_solution(toy_loaded, solution)
        assert result.status == SolutionCheckStatus.INVALID_CUSTOMER_INDEX

    def test_capacity_violation_detected(self, toy_loaded):
        # demands are 4 + 4 = 8 <= 10 on one route, but capacity 10 with both plus
        # nothing else; tighten by putting both on one route with capacity 7.
        payload = toy_instance_payload()
        payload["vehicle_capacity"] = 7
        instance = td_instance_from_payload(payload)
        loaded = toy_loaded
        loaded.instance = instance
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1, 2]])
        result = check_td_solution(loaded, solution)
        assert result.status == SolutionCheckStatus.VEHICLE_CAPACITY_EXCEEDED

    def test_too_many_vehicles_detected(self, toy_loaded):
        payload = toy_instance_payload()
        payload["num_vehicles"] = 1
        toy_loaded.instance = td_instance_from_payload(payload)
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1], [2]])
        result = check_td_solution(toy_loaded, solution)
        assert result.status == SolutionCheckStatus.TOO_MANY_VEHICLES_USED

    def test_time_window_violation_detected(self, toy_loaded):
        payload = toy_instance_payload()
        payload["time_windows"] = [[0, 100], [0, 100], [0, 10]]
        toy_loaded.instance = td_instance_from_payload(payload)
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1], [2]])
        result = check_td_solution(toy_loaded, solution)
        assert result.status == SolutionCheckStatus.TIME_WINDOW_VIOLATED

    def test_tdvrp_timing_infeasibility_status(self, toy_loaded_tdvrp):
        # Shrink the horizon so no departure completes the route in time:
        # rebuild ATFs over [0, 20] where 0 -> 1 -> 2 -> 0 needs at least 50.
        atfs = make_toy_atfs()
        for key, atf in list(atfs.arcs.items()):
            atfs.arcs[key] = type(atf)([0.0, 20.0], [atf.ys[0], atf.evaluate(20.0)])
        atfs.horizon = (0.0, 20.0)
        payload = toy_instance_payload(with_time_windows=False)
        payload["horizon"] = [0, 20]
        toy_loaded_tdvrp.instance = td_instance_from_payload(payload)
        toy_loaded_tdvrp.atfs = atfs
        solution = BenchmarkSolution(instance_name="TOY1", routes=[[1, 2]])
        result = check_td_solution(toy_loaded_tdvrp, solution)
        assert result.status == SolutionCheckStatus.ROUTE_TIMING_INFEASIBLE

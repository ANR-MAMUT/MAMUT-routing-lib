from __future__ import annotations

import pytest

pytest.importorskip("pyvrp")

from mamut_routing_lib.solvers.pyvrp import to_pyvrp_problem, to_vrplib_dict


def test_to_vrplib_dict_for_cvrp_has_no_time_windows(toy_cvrp_instance) -> None:
    payload = to_vrplib_dict(toy_cvrp_instance)
    assert payload["type"] == "CVRP"
    assert payload["dimension"] == toy_cvrp_instance.num_customers + 1
    assert payload["capacity"] == toy_cvrp_instance.vehicle_capacity
    assert "time_window" not in payload
    assert "service_time" not in payload


def test_to_vrplib_dict_for_vrptw_includes_time_windows(toy_vrptw_instance) -> None:
    payload = to_vrplib_dict(toy_vrptw_instance)
    assert payload["type"] == "CVRPTW"
    assert payload["time_window"].shape == (toy_vrptw_instance.num_customers + 1, 2)
    assert payload["service_time"].shape == (toy_vrptw_instance.num_customers + 1,)


def test_to_pyvrp_problem_falls_back_to_num_customers_when_num_vehicles_absent(toy_cvrp_instance) -> None:
    # The toy fixture has num_vehicles unset; expect fallback to num_customers.
    problem = to_pyvrp_problem(toy_cvrp_instance)
    assert problem.num_vehicles == toy_cvrp_instance.num_customers
    assert problem.num_clients == toy_cvrp_instance.num_customers


def test_to_pyvrp_problem_rejects_unknown_round_func(toy_cvrp_instance) -> None:
    with pytest.raises(TypeError):
        to_pyvrp_problem(toy_cvrp_instance, round_func=12345)  # type: ignore[arg-type]

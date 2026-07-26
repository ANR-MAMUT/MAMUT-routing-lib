"""Tests for the FleetCostDuration objective (Plan 11, Blauth2024).

Toy ground truth (see td_utils): route durations [1] = 20, [2] = 25,
[1, 2] = 30; Duration([[1], [2]]) = 45.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
from mamut_routing_lib.checker import get_objective_tuple
from mamut_routing_lib.enums import BenchmarkName, InstanceOrigin, ObjectiveFunction
from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.td import (
    TD_OBJECTIVES,
    check_td_solution,
    compute_solution_cost,
    create_td_bks_from_solution,
    load_td_instance,
    save_td_solution_as_bks_if_improved,
)
from mamut_routing_lib.td.models import BenchmarkInstanceTDVRPTW

from td_utils import toy_instance_payload, write_toy_instance_files

AUTHORS = "Test Author"
FLEET = ObjectiveFunction.FLEET_COST_DURATION


@pytest.fixture(params=[True, False], ids=["tdvrptw", "tdvrp"])
def toy_fleet_loaded(request, tmp_path):
    instance_path = write_toy_instance_files(
        tmp_path,
        with_time_windows=request.param,
        fleet_fixed_cost=1000.0,
        num_vehicles=None,
    )
    return load_td_instance(instance_path)


@pytest.fixture
def toy_plain_loaded(tmp_path):
    instance_path = write_toy_instance_files(tmp_path)
    return load_td_instance(instance_path)


def _solution(loaded, routes, cost=None):
    return BenchmarkSolution(instance_name=loaded.instance.instance_name, routes=routes, cost=cost)


def test_enums_exist():
    assert ObjectiveFunction("FleetCostDuration") is FLEET
    assert BenchmarkName("Blauth2024") is BenchmarkName.BLAUTH_2024
    assert InstanceOrigin("Blauth2024") is InstanceOrigin.BLAUTH_2024
    assert TD_OBJECTIVES == (ObjectiveFunction.DURATION, FLEET)


def test_fleet_fixed_cost_field_round_trips(toy_fleet_loaded):
    assert toy_fleet_loaded.instance.fleet_fixed_cost == 1000.0


def test_fleet_fixed_cost_rejects_negative_and_nan():
    payload = toy_instance_payload(fleet_fixed_cost=-1.0)
    with pytest.raises(ValidationError, match="fleet_fixed_cost"):
        BenchmarkInstanceTDVRPTW.model_validate(payload)
    payload = toy_instance_payload(fleet_fixed_cost=float("nan"))
    with pytest.raises(ValidationError, match="fleet_fixed_cost"):
        BenchmarkInstanceTDVRPTW.model_validate(payload)


def test_fleet_fixed_cost_defaults_to_none(toy_plain_loaded):
    assert toy_plain_loaded.instance.fleet_fixed_cost is None


def test_fleet_cost_duration_scoring_is_exact(toy_fleet_loaded):
    # Duration fold first, then one multiply and one add: 45 + 1000 * 2.
    result = check_td_solution(toy_fleet_loaded, _solution(toy_fleet_loaded, [[1], [2]]), FLEET)
    assert result.is_valid()
    assert result.routing_cost == 2045.0
    assert result.num_routes == 2

    single = check_td_solution(toy_fleet_loaded, _solution(toy_fleet_loaded, [[1, 2]]), FLEET)
    assert single.routing_cost == 1030.0
    assert single.num_routes == 1


def test_duration_scoring_ignores_fleet_fixed_cost(toy_fleet_loaded):
    result = check_td_solution(toy_fleet_loaded, _solution(toy_fleet_loaded, [[1], [2]]))
    assert result.is_valid()
    assert result.routing_cost == 45.0


def test_compute_solution_cost_matches_checker(toy_fleet_loaded):
    instance, atfs = toy_fleet_loaded.instance, toy_fleet_loaded.atfs
    assert compute_solution_cost(instance, atfs, [[1], [2]], FLEET) == 2045.0
    assert compute_solution_cost(instance, atfs, [[1], [2]]) == 45.0


def test_fleet_cost_duration_requires_field(toy_plain_loaded):
    with pytest.raises(ValueError, match="fleet_fixed_cost"):
        check_td_solution(toy_plain_loaded, _solution(toy_plain_loaded, [[1], [2]]), FLEET)


def test_declared_cost_checked_under_requested_objective(toy_fleet_loaded):
    exact = check_td_solution(toy_fleet_loaded, _solution(toy_fleet_loaded, [[1], [2]], cost=2045.0), FLEET)
    assert exact.is_valid()
    mismatch = check_td_solution(toy_fleet_loaded, _solution(toy_fleet_loaded, [[1], [2]], cost=45.0), FLEET)
    assert not mismatch.is_valid()
    assert "FleetCostDuration" in mismatch.error_message


def test_bks_objective_mismatch_raises(toy_fleet_loaded):
    duration_bks = create_td_bks_from_solution(
        toy_fleet_loaded, _solution(toy_fleet_loaded, [[1, 2]]), authors=AUTHORS
    )
    with pytest.raises(ValueError, match="declares objective"):
        check_td_solution(toy_fleet_loaded, duration_bks, FLEET)

    fleet_bks = create_td_bks_from_solution(
        toy_fleet_loaded, _solution(toy_fleet_loaded, [[1, 2]]), authors=AUTHORS, objective_function=FLEET
    )
    with pytest.raises(ValueError, match="declares objective"):
        check_td_solution(toy_fleet_loaded, fleet_bks)


def test_objective_tuple_is_mono():
    assert get_objective_tuple([[1], [2]], 2045.0, FLEET) == (2045.0,)


def test_exactness_with_blauth_scale_fixed_cost(tmp_path):
    # Integer-ms-scale F stays exact in doubles: 45 + 2 * 36000000 exactly.
    path = write_toy_instance_files(tmp_path, fleet_fixed_cost=36000000.0, num_vehicles=None)
    loaded = load_td_instance(path)
    result = check_td_solution(loaded, _solution(loaded, [[1], [2]]), FLEET)
    assert result.routing_cost == 72000045.0


def test_unlimited_fleet_skips_vehicle_bound(toy_fleet_loaded):
    # num_vehicles is null on this fixture: two routes must pass.
    assert toy_fleet_loaded.instance.num_vehicles is None
    result = check_td_solution(toy_fleet_loaded, _solution(toy_fleet_loaded, [[1], [2]]), FLEET)
    assert result.is_valid()


def test_bks_lifecycle_per_objective_store(toy_fleet_loaded):
    loaded = toy_fleet_loaded
    fleet_path = get_bks_path_for_instance(loaded.instance_path, FLEET)
    duration_path = get_bks_path_for_instance(loaded.instance_path, ObjectiveFunction.DURATION)
    assert fleet_path.name == "TOY1.bks.FleetCostDuration.json"

    created = save_td_solution_as_bks_if_improved(
        loaded, _solution(loaded, [[1], [2]]), authors=AUTHORS, objective_function=FLEET
    )
    assert created.action == "created"
    assert created.path == fleet_path
    assert load_bks(fleet_path).cost == 2045.0
    assert load_bks(fleet_path).objective_function == FLEET

    kept = save_td_solution_as_bks_if_improved(
        loaded, _solution(loaded, [[1], [2]]), authors=AUTHORS, objective_function=FLEET
    )
    assert kept.action == "kept_existing"

    replaced = save_td_solution_as_bks_if_improved(
        loaded, _solution(loaded, [[1, 2]]), authors=AUTHORS, objective_function=FLEET
    )
    assert replaced.action == "replaced"
    assert load_bks(fleet_path).cost == 1030.0

    # The Duration store is a separate file, untouched by the fleet store.
    duration_created = save_td_solution_as_bks_if_improved(
        loaded, _solution(loaded, [[1], [2]]), authors=AUTHORS
    )
    assert duration_created.action == "created"
    assert duration_created.path == duration_path
    assert load_bks(duration_path).cost == 45.0
    assert load_bks(fleet_path).cost == 1030.0


def test_bks_refuses_non_td_objective(toy_fleet_loaded):
    with pytest.raises(ValueError, match="only supports"):
        save_td_solution_as_bks_if_improved(
            toy_fleet_loaded,
            _solution(toy_fleet_loaded, [[1, 2]]),
            authors=AUTHORS,
            objective_function=ObjectiveFunction.MONO_COST,
        )

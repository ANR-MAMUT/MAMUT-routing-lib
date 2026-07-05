"""Tests for the TD-aware BKS helpers (Duration objective, TD checker refereed)."""

from __future__ import annotations

import pytest

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.td import (
    check_td_solution,
    create_td_bks_from_solution,
    load_td_instance,
    save_td_solution_as_bks_if_improved,
)

from td_utils import write_toy_instance_files

AUTHORS = "Test Author"


@pytest.fixture(params=[True, False], ids=["tdvrptw", "tdvrp"])
def toy_loaded(request, tmp_path):
    instance_path = write_toy_instance_files(tmp_path, with_time_windows=request.param)
    return load_td_instance(instance_path)


def _solution(loaded, routes):
    return BenchmarkSolution(instance_name=loaded.instance.instance_name, routes=routes)


def test_create_td_bks_prices_with_td_checker(toy_loaded):
    solution = _solution(toy_loaded, [[1], [2]])
    bks = create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)
    reference = check_td_solution(toy_loaded, solution)
    assert bks.objective_function == ObjectiveFunction.DURATION
    assert bks.cost == reference.routing_cost
    assert bks.metadata["authors"] == AUTHORS
    assert bks.metadata["validated_cost"] == reference.routing_cost
    assert bks.metadata["validated_num_routes"] == 2


def test_create_td_bks_accepts_instance_path(toy_loaded):
    solution = _solution(toy_loaded, [[1], [2]])
    bks = create_td_bks_from_solution(toy_loaded.instance_path, solution, authors=AUTHORS)
    assert bks.cost == check_td_solution(toy_loaded, solution).routing_cost


def test_create_td_bks_rejects_invalid_solution(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2], [2]])  # customer 2 served twice
    with pytest.raises(ValueError, match="invalid TD solution"):
        create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)


def test_create_td_bks_rejects_cost_mismatch(toy_loaded):
    reference = check_td_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]))
    solution = BenchmarkSolution(
        instance_name=toy_loaded.instance.instance_name,
        routes=[[1], [2]],
        cost=reference.routing_cost + 1.0,
    )
    # check_td_solution itself enforces the exact cost match.
    with pytest.raises(ValueError, match="does not match computed Duration"):
        create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)


def test_create_td_bks_accepts_exact_cost(toy_loaded):
    reference = check_td_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]))
    solution = BenchmarkSolution(
        instance_name=toy_loaded.instance.instance_name,
        routes=[[1], [2]],
        cost=reference.routing_cost,
    )
    bks = create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)
    assert bks.cost == reference.routing_cost


def test_create_td_bks_requires_authors(toy_loaded):
    with pytest.raises(ValueError, match="authors"):
        create_td_bks_from_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]), authors="  ")


def test_save_creates_then_keeps_then_replaces(toy_loaded):
    bks_path = get_bks_path_for_instance(toy_loaded.instance_path, ObjectiveFunction.DURATION)
    assert not bks_path.exists()

    # The single-route solution [1, 2] is strictly cheaper than {[1], [2]} on
    # the toy instance (one depot round trip saved).
    worse = _solution(toy_loaded, [[1], [2]])
    better = _solution(toy_loaded, [[1, 2]])
    worse_cost = check_td_solution(toy_loaded, worse).routing_cost
    better_cost = check_td_solution(toy_loaded, better).routing_cost
    assert better_cost < worse_cost

    created = save_td_solution_as_bks_if_improved(toy_loaded, worse, authors=AUTHORS)
    assert created.action == "created"
    assert created.path == bks_path
    assert load_bks(bks_path).cost == worse_cost

    kept = save_td_solution_as_bks_if_improved(toy_loaded, worse, authors=AUTHORS)
    assert kept.action == "kept_existing"
    assert load_bks(bks_path).cost == worse_cost

    replaced = save_td_solution_as_bks_if_improved(toy_loaded, better, authors=AUTHORS)
    assert replaced.action == "replaced"
    assert replaced.previous_path == bks_path
    stored = load_bks(bks_path)
    assert stored.cost == better_cost
    assert stored.routes == [[1, 2]]
    assert stored.objective_function == ObjectiveFunction.DURATION


def test_save_carries_extra_metadata(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    result = save_td_solution_as_bks_if_improved(
        toy_loaded,
        solution,
        authors=AUTHORS,
        metadata={"solver": "kayros-0.0.1.dev0", "seed": 42},
    )
    assert result.action == "created"
    stored = load_bks(result.path)
    assert stored.metadata["solver"] == "kayros-0.0.1.dev0"
    assert stored.metadata["seed"] == 42
    assert stored.metadata["authors"] == AUTHORS

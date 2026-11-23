"""Static BKS ranking under exact arithmetic on the published binary64 inputs."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import pytest

from mamut_routing_lib import (
    BenchmarkBKS,
    BenchmarkInstanceCVRP,
    BenchmarkSolution,
    ObjectiveFunction,
    create_bks_from_solution,
    load_bks,
    save_bks_if_improved,
    save_json_to_file,
)

AUTHORS = "Test Author"

MATRIX_ORDER = [
    [0.0, 6544.272, 3101.899, 13053.596, 1541.482],
    [6544.272, 0.0, 10764.052, 7377.209, 1254.179],
    [3101.899, 10764.052, 0.0, 10197.971, 846.164],
    [13053.596, 7377.209, 10197.971, 0.0, 8729.549],
    [1541.482, 1254.179, 846.164, 8729.549, 0.0],
]
MATRIX_DECIMAL = [
    [0.0, 11.5, 3.0, 18.2, 18.1],
    [11.5, 0.0, 12.7, 15.8, 2.3],
    [3.0, 12.7, 0.0, 24.7, 4.2],
    [18.2, 15.8, 24.7, 0.0, 7.3],
    [18.1, 2.3, 4.2, 7.3, 0.0],
]


def _instance(matrix: list[list[float]], capacity: int = 100) -> BenchmarkInstanceCVRP:
    return BenchmarkInstanceCVRP(
        instance_name="mamut-ties-n4",
        instance_origin="OsmCvrpGen",
        benchmark_name="Mamut2026",
        num_customers=4,
        vehicle_capacity=capacity,
        coordinates=[(0.0, 0.0)] * 5,
        demands=[0, 1, 1, 1, 1],
        depot=0,
        arc_costs=matrix,
        metadata={},
    )


def _priced(instance: BenchmarkInstanceCVRP, routes: list[list[int]]) -> float:
    return float(sum(Fraction(float(instance.arc_costs[i][j]))
                     for route in routes for i, j in zip([0, *route], [*route, 0])))


def _store(tmp_path: Path, instance: BenchmarkInstanceCVRP, routes: list[list[int]], objective=ObjectiveFunction.MONO_COST):
    instance_path = tmp_path / f"{instance.instance_name}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), instance_path)
    bks = create_bks_from_solution(
        instance,
        BenchmarkSolution(instance_name=instance.instance_name, routes=routes),
        objective,
        authors="Incumbent",
    )
    assert save_bks_if_improved(instance, bks, instance_path).action == "created"
    return instance_path


def _candidate(instance, routes, objective=ObjectiveFunction.MONO_COST) -> BenchmarkBKS:
    # A newly submitted cost asserts the once-rounded exact route sum.
    return BenchmarkBKS(
        instance_name=instance.instance_name,
        objective_function=objective,
        routes=routes,
        cost=_priced(instance, routes),
        metadata={"authors": AUTHORS},
    )


@pytest.mark.parametrize(
    ("matrix", "incumbent", "challenger"),
    [
        pytest.param(MATRIX_ORDER, [[1, 2], [3, 4]], [[3, 4], [1, 2]], id="reordered"),
        pytest.param(MATRIX_ORDER, [[1, 2], [3, 4]], [[2, 1], [4, 3]], id="reversed"),
    ],
)
def test_float_noise_is_a_tie(tmp_path: Path, matrix, incumbent, challenger) -> None:
    instance = _instance(matrix)
    instance_path = _store(tmp_path, instance, incumbent)
    stored = load_bks(instance_path.with_name(f"{instance.instance_name}.bks.MonoCost.json"))
    candidate = _candidate(instance, challenger)
    assert candidate.cost == stored.cost

    result = save_bks_if_improved(instance, candidate, instance_path)

    assert result.action == "kept_existing"
    assert result.tie
    assert load_bks(result.path).metadata["authors"] == "Incumbent"


def test_a_real_improvement_replaces(tmp_path: Path) -> None:
    instance = _instance(MATRIX_DECIMAL)
    instance_path = _store(tmp_path, instance, [[1], [2], [3], [4]])
    result = save_bks_if_improved(instance, _candidate(instance, [[1, 2], [3, 4]]), instance_path)
    assert result.action == "replaced"
    assert not result.tie
    assert load_bks(result.path).metadata["authors"] == AUTHORS


def test_a_worse_solution_is_not_a_tie(tmp_path: Path) -> None:
    instance = _instance(MATRIX_DECIMAL)
    instance_path = _store(tmp_path, instance, [[1, 2], [3, 4]])
    result = save_bks_if_improved(instance, _candidate(instance, [[1], [2], [3], [4]]), instance_path)
    assert result.action == "kept_existing"
    assert not result.tie


def test_hierarchical_objective_ranks_route_count_first(tmp_path: Path) -> None:
    instance = _instance(MATRIX_DECIMAL)
    instance_path = _store(tmp_path, instance, [[1, 3], [2, 4]], ObjectiveFunction.HIERARCHICAL_VEHICLE_COST)
    one_route = [[1, 2, 3, 4]]
    challenger = _candidate(instance, one_route, ObjectiveFunction.HIERARCHICAL_VEHICLE_COST)
    assert challenger.cost > _priced(instance, [[1, 3], [2, 4]])
    result = save_bks_if_improved(instance, challenger, instance_path)
    assert result.action == "replaced"


def test_created_bks_stores_canonical_routes_and_their_price() -> None:
    instance = _instance(MATRIX_ORDER)
    solver_routes = [[3, 4], [1, 2]]
    bks = create_bks_from_solution(
        instance,
        BenchmarkSolution(
            instance_name=instance.instance_name,
            routes=solver_routes,
            cost=_priced(instance, solver_routes),
        ),
        ObjectiveFunction.MONO_COST,
        authors=AUTHORS,
        metadata={"validated_cost": 1.0, "validated_num_routes": 99},
    )
    assert bks.routes == [[1, 2], [3, 4]]
    assert bks.cost == _priced(instance, [[1, 2], [3, 4]])
    assert bks.metadata["validated_cost"] == bks.cost
    assert bks.metadata["validated_num_routes"] == 2


def test_invalid_candidates_are_refused(tmp_path: Path) -> None:
    instance = _instance(MATRIX_DECIMAL)
    instance_path = _store(tmp_path, instance, [[1, 2], [3, 4]])
    incomplete = BenchmarkBKS(
        instance_name=instance.instance_name,
        objective_function=ObjectiveFunction.MONO_COST,
        routes=[[1, 2]],
        cost=0.5,
        metadata={"authors": AUTHORS},
    )
    with pytest.raises(ValueError, match="Candidate BKS is invalid"):
        save_bks_if_improved(instance, incomplete, instance_path)
    misnamed = _candidate(instance, [[1, 2], [3, 4]]).model_copy(update={"instance_name": "another"})
    with pytest.raises(ValueError, match="does not match the instance"):
        save_bks_if_improved(instance, misnamed, instance_path)

def test_decimal_tie_is_a_strict_binary64_improvement(tmp_path):
    instance = _instance(MATRIX_DECIMAL)
    incumbent, challenger = [[1, 3], [2, 4]], [[1, 2], [3, 4]]
    path = _store(tmp_path, instance, incumbent)
    assert _priced(instance, incumbent) == _priced(instance, challenger)
    result = save_bks_if_improved(instance, _candidate(instance, challenger), path, backend="python")
    assert result.action == "replaced" and not result.tie
    bks = load_bks(result.path)
    assert bks.routes == challenger
    assert bks.metadata["exact_evaluation"]["evaluation"]["backend"] == "python-fraction"
    assert "standard" not in bks.metadata


def test_historical_display_is_not_an_exact_claim(tmp_path):
    instance = _instance(MATRIX_ORDER)
    path = tmp_path / "toy.vrp.json"
    incumbent = _candidate(instance, [[1, 2], [3, 4]])
    incumbent.cost = 123.0
    store = path.with_name("toy.bks.MonoCost.json")
    save_json_to_file(incumbent.model_dump(mode="json"), store)
    original = store.read_bytes()
    result = save_bks_if_improved(instance, _candidate(instance, [[3, 4], [1, 2]]), path, backend="python")
    assert result.action == "kept_existing" and result.tie
    assert store.read_bytes() == original


def test_corrupted_exact_cost_is_rejected(tmp_path):
    instance = _instance(MATRIX_ORDER)
    routes = [[1, 2], [3, 4]]
    path = _store(tmp_path, instance, routes)
    store = path.with_name(f"{instance.instance_name}.bks.MonoCost.json")
    record = load_bks(store)
    record.metadata["exact_evaluation"]["authoritative_cost"]["numerator"] = "0"
    save_json_to_file(record.model_dump(mode="json"), store)
    before = store.read_bytes()
    with pytest.raises(ValueError, match="authoritative_cost"):
        save_bks_if_improved(instance, _candidate(instance, routes), path)
    assert store.read_bytes() == before


def test_manifest_bound_incumbent_requires_manifest_verification(tmp_path):
    instance = _instance(MATRIX_DECIMAL)
    routes = [[1], [2], [3], [4]]
    path = _store(tmp_path, instance, routes)
    store = path.with_name(f"{instance.instance_name}.bks.MonoCost.json")
    record = load_bks(store)
    record.metadata["standard"] = {**record.metadata.pop("exact_evaluation"), "manifest_sha256": "f"*64}
    save_json_to_file(record.model_dump(mode="json"), store)
    before = store.read_bytes()
    assert save_bks_if_improved(instance, _candidate(instance, routes), path).tie
    with pytest.raises(ValueError, match="manifest"):
        save_bks_if_improved(instance, _candidate(instance, [[1, 2], [3, 4]]), path)
    assert store.read_bytes() == before


def test_bks_exact_cost_can_overflow_its_binary64_display(tmp_path):
    instance = _instance(MATRIX_ORDER)
    instance.arc_costs[0][1] = sys.float_info.max
    instance.arc_costs[4][0] = sys.float_info.max
    solution = BenchmarkSolution(instance_name=instance.instance_name, routes=[[1, 2, 3, 4]])
    bks = create_bks_from_solution(instance, solution, ObjectiveFunction.MONO_COST,
                                   authors=AUTHORS, backend="python")
    assert bks.cost is None
    assert bks.metadata["exact_evaluation"]["display_overflow"] is True
    update = save_bks_if_improved(instance, bks, tmp_path / "toy.vrp.json", backend="python")
    assert update.action == "created" and update.candidate_cost is None
    assert load_bks(update.path).metadata["exact_evaluation"]["authoritative_cost"] == (
        bks.metadata["exact_evaluation"]["authoritative_cost"]
    )


def test_solver_claim_requires_an_explicit_matching_target():
    instance = _instance(MATRIX_ORDER)
    solution = BenchmarkSolution(instance_name=instance.instance_name, routes=[[1, 2, 3, 4]])
    claim = dict(status="solver-reported-optimal", target_profile="td-fold/1", solver="solver X")
    with pytest.raises(ValueError, match="target_profile"):
        create_bks_from_solution(instance, solution, ObjectiveFunction.MONO_COST,
                                 authors=AUTHORS, metadata={"optimality": claim}, backend="python")
    claim["target_profile"] = "binary64-exact-v1"
    result = create_bks_from_solution(instance, solution, ObjectiveFunction.MONO_COST,
                                      authors=AUTHORS, metadata={"optimality": claim}, backend="python")
    assert result.metadata["optimality"] == claim

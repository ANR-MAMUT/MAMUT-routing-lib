"""The BKS store on slim collection instances (hydrated from their arc-cost source)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mamut_routing_lib import BenchmarkSolution, load_bks, save_solution_as_bks_if_improved
from mamut_routing_lib.artifacts import get_bks_path_for_instance, hydrate_collection_instance, load_benchmark_instance
from mamut_routing_lib.checker import check_solution
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.models import BenchmarkInstanceCVRP, BenchmarkInstanceCVRPCollection
from mamut_routing_lib.sidecars import CollectionMarker, save_collection_marker


@pytest.fixture()
def slim_instance_path(tmp_path: Path) -> Path:
    root = tmp_path / "Mamut2026"
    save_collection_marker(CollectionMarker(family="Mamut2026"), root)
    base = "mamut-testville-n2-k1-poi"
    payload = {
        "instance_name": base,
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Mamut2026",
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 10,
        "coordinates": [[0, 0], [3, 4], [6, 8]],
        "demands": [0, 3, 4],
        "depot": 0,
        "metric_variant": "euclidean",
        "arc_costs_source": {"model": "euclidean", "decimals": 3},
        "metadata": {"problem_type": "CVRP"},
    }
    instance_path = root / "CVRP" / "euclidean" / "testville" / "n=2" / base / f"{base}.vrp.json"
    save_json_to_file(payload, instance_path)
    return instance_path


def test_hydrate_collection_instance_gives_checkable_embedded_model(slim_instance_path: Path) -> None:
    slim = load_benchmark_instance(slim_instance_path)
    assert isinstance(slim, BenchmarkInstanceCVRPCollection)
    with pytest.raises(TypeError):
        check_solution(slim, BenchmarkSolution(instance_name=slim.instance_name, routes=[[1, 2]]))
    hydrated = hydrate_collection_instance(slim, slim_instance_path)
    assert isinstance(hydrated, BenchmarkInstanceCVRP)
    assert hydrated.arc_costs[0][1] == 5.0 and hydrated.arc_costs[1][2] == 5.0
    result = check_solution(hydrated, BenchmarkSolution(instance_name=slim.instance_name, routes=[[1, 2]]))
    assert result.is_valid() and result.routing_cost == 20.0
    assert hydrate_collection_instance(hydrated) is hydrated


def test_save_solution_as_bks_if_improved_hydrates_collection_instances(slim_instance_path: Path) -> None:
    name = "mamut-testville-n2-k1-poi"
    first = save_solution_as_bks_if_improved(
        slim_instance_path,
        ObjectiveFunction.MONO_COST,
        BenchmarkSolution(instance_name=name, routes=[[1], [2]]),
        authors="tests",
        metadata={"method": "hand"},
    )
    assert first.action == "created" and first.candidate_cost == 30.0
    bks_path = get_bks_path_for_instance(slim_instance_path, ObjectiveFunction.MONO_COST)
    assert load_bks(bks_path).cost == 30.0
    better = save_solution_as_bks_if_improved(
        slim_instance_path, ObjectiveFunction.MONO_COST,
        BenchmarkSolution(instance_name=name, routes=[[1, 2]]), authors="tests",
    )
    assert better.action == "replaced" and load_bks(bks_path).cost == 20.0
    worse = save_solution_as_bks_if_improved(
        slim_instance_path, ObjectiveFunction.MONO_COST,
        BenchmarkSolution(instance_name=name, routes=[[2], [1]]), authors="tests",
    )
    assert worse.action == "kept_existing" and load_bks(bks_path).cost == 20.0

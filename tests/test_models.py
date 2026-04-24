from __future__ import annotations

import pytest
from pydantic import ValidationError

from mamut_routing_lib.enums import MetricVariant, ObjectiveFunction, ProblemType
from mamut_routing_lib.models import (
    ArtifactPaths,
    BenchmarkBKS,
    BenchmarkInstance,
    BenchmarkInstanceCVRP,
    BenchmarkInstanceVRPTW,
    InstanceMetadata,
)


def make_metadata_kwargs(problem_type: str = "CVRP", metric_variant: str = "fastest") -> dict:
    return {
        "authors": "Florian Rascoussier (0nyr) and Adrien Pichon (Anzury)",
        "generated_at": "2026-04-14T10:47:08.075",
        "problem_type": problem_type,
        "metric_variant": metric_variant,
        "place_slug": "brest",
        "source_base_name": "brest_poi-n101-k14",
        "source_city": "Brest",
        "source_seed": 1565438598640290434,
        "source_folder": "instances_v2/osm/brest/n101",
        "num_vehicles_lb": 14,
        "generator_version": "mamut-routing-lib.seed-v1",
        "artifact_paths": {
            "vrp_json": "benchmarks/CVRP/Mamut2026/fastest/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.vrp.json",
            "vrp": "benchmarks/CVRP/Mamut2026/fastest/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.vrp",
            "meta": "benchmarks/CVRP/Mamut2026/sidecars/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.meta.json",
            "manifest": "benchmarks/CVRP/Mamut2026/sidecars/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.manifest.json",
        },
        "sibling_variant_paths": {
            "euclidean": "benchmarks/CVRP/Mamut2026/euclidean/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.vrp.json"
        },
        "derived_problem_paths": {
            "fastest": "benchmarks/VRPTW/Mamut2026/fastest/brest/n=2/mamut-n2-beefdad/mamut-n2-beefdad.vrp.json"
        },
    }


def make_valid_historical_instance_kwargs() -> dict:
    return {
        "instance_name": "C101",
        "instance_origin": "Solomon1987",
        "benchmark_name": "Sintef2008",
        "num_customers": 2,
        "num_vehicles": 2,
        "vehicle_capacity": 10,
        "coordinates": [(0, 0), (1, 1), (2, 2)],
        "demands": [0, 1, 2],
        "service_times": [0, 10, 10],
        "time_windows": [(0, 100), (0, 100), (0, 100)],
        "depot": 0,
        "arc_costs": [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ],
    }


def make_valid_cvrp_instance_kwargs() -> dict:
    return {
        "instance_id": "mamut-n2-deadbee",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Mamut2026",
        "num_customers": 2,
        "vehicle_capacity": 10,
        "coordinates": [(0.0, 0.0), (1.5, 1.5), (2.25, 2.25)],
        "demands": [0, 1, 2],
        "depot": 0,
        "arc_costs": [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ],
        "metadata": make_metadata_kwargs(),
    }


def make_valid_vrptw_instance_kwargs() -> dict:
    payload = make_valid_cvrp_instance_kwargs()
    payload["instance_id"] = "mamut-n2-beefdad"
    payload["service_times"] = [0, 10, 20]
    payload["time_windows"] = [(0, 86400), (100, 5000), (200, 6000)]
    payload["metadata"] = make_metadata_kwargs(problem_type="VRPTW", metric_variant="euclidean")
    payload["metadata"]["artifact_paths"] = {
        "vrp_json": "benchmarks/VRPTW/Mamut2026/euclidean/brest/n=2/mamut-n2-beefdad/mamut-n2-beefdad.vrp.json",
        "vrp": "benchmarks/VRPTW/Mamut2026/euclidean/brest/n=2/mamut-n2-beefdad/mamut-n2-beefdad.vrp",
        "meta": "benchmarks/VRPTW/Mamut2026/sidecars/brest/n=2/mamut-n2-beefdad/mamut-n2-beefdad.meta.json",
        "manifest": "benchmarks/VRPTW/Mamut2026/sidecars/brest/n=2/mamut-n2-beefdad/mamut-n2-beefdad.manifest.json",
    }
    payload["metadata"]["source_problem_paths"] = {
        "cvrp_vrp_json": "benchmarks/CVRP/Mamut2026/euclidean/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.vrp.json",
        "cvrp_vrp": "benchmarks/CVRP/Mamut2026/euclidean/brest/n=2/mamut-n2-deadbee/mamut-n2-deadbee.vrp",
    }
    return payload


def test_historical_benchmark_instance_accepts_arc_costs() -> None:
    instance = BenchmarkInstance(**make_valid_historical_instance_kwargs())
    assert instance.arc_costs[0][2] == 2


def test_historical_benchmark_instance_rejects_malformed_arc_costs_matrix() -> None:
    payload = make_valid_historical_instance_kwargs()
    payload["arc_costs"] = [
        [0, 1],
        [1, 0],
    ]

    with pytest.raises(ValidationError):
        BenchmarkInstance(**payload)


def test_cvrp_instance_accepts_structured_metadata_and_float_coordinates() -> None:
    instance = BenchmarkInstanceCVRP(**make_valid_cvrp_instance_kwargs())

    assert instance.metadata.problem_type == ProblemType.CVRP
    assert instance.metadata.metric_variant == MetricVariant.FASTEST
    assert instance.num_vehicles is None
    assert instance.metadata.num_vehicles_lb == 14
    assert instance.coordinates[1] == (1.5, 1.5)


def test_vrptw_instance_accepts_time_fields_and_relative_paths() -> None:
    instance = BenchmarkInstanceVRPTW(**make_valid_vrptw_instance_kwargs())

    assert instance.metadata.problem_type == ProblemType.VRPTW
    assert instance.time_windows[0] == (0, 86400)


def test_metadata_rejects_absolute_artifact_paths() -> None:
    payload = make_metadata_kwargs()
    payload["artifact_paths"]["vrp_json"] = "/absolute/path/not/allowed.vrp.json"

    with pytest.raises(ValidationError):
        InstanceMetadata(**payload)


def test_artifact_paths_model_accepts_relative_paths() -> None:
    artifact_paths = ArtifactPaths(**make_metadata_kwargs()["artifact_paths"])

    assert artifact_paths.vrp.endswith(".vrp")


def test_benchmark_bks_accepts_supported_objectives() -> None:
    mono_cost_bks = BenchmarkBKS(
        instance_name="C101",
        objective_function="MonoCost",
        routes=[[1, 2]],
        cost=12,
        metadata={},
    )
    hierarchical_bks = BenchmarkBKS(
        instance_name="C101",
        objective_function="HierarchicalVehicleCost",
        routes=[[1, 2]],
        cost=12,
        metadata={},
    )

    assert mono_cost_bks.objective_function == ObjectiveFunction.MONO_COST
    assert hierarchical_bks.objective_function == ObjectiveFunction.HIERARCHICAL_VEHICLE_COST


def test_benchmark_bks_rejects_unknown_objective() -> None:
    with pytest.raises(ValidationError):
        BenchmarkBKS(
            instance_name="C101",
            objective_function="UnknownObjective",
            routes=[[1, 2]],
            cost=12,
            metadata={},
        )

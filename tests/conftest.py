from __future__ import annotations

import pytest

from mamut_routing_lib.models import BenchmarkInstanceCVRP, BenchmarkInstanceVRPTW


_TOY_METADATA_CVRP = {
    "authors": "Florian Rascoussier (0nyr) and Adrien Pichon (Anzury)",
    "generated_at": "2026-04-23T00:00:00",
    "problem_type": "CVRP",
    "metric_variant": "fastest",
    "place_slug": "testville",
    "source_base_name": "test",
    "source_city": "Testville",
    "source_seed": 1,
    "source_folder": "test",
    "num_vehicles_lb": 1,
    "artifact_paths": {
        "vrp_json": "benchmarks/CVRP/Mamut2026/fastest/testville/n=2/mamut-n2-testcvrp/mamut-n2-testcvrp.vrp.json",
        "vrp": "benchmarks/CVRP/Mamut2026/fastest/testville/n=2/mamut-n2-testcvrp/mamut-n2-testcvrp.vrp",
        "meta": "benchmarks/CVRP/Mamut2026/sidecars/testville/n=2/mamut-n2-testcvrp/mamut-n2-testcvrp.meta.json",
        "manifest": "benchmarks/CVRP/Mamut2026/sidecars/testville/n=2/mamut-n2-testcvrp/mamut-n2-testcvrp.manifest.json",
    },
}

_TOY_METADATA_VRPTW = {
    **_TOY_METADATA_CVRP,
    "problem_type": "VRPTW",
    "artifact_paths": {
        "vrp_json": "benchmarks/VRPTW/Mamut2026/fastest/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.vrp.json",
        "vrp": "benchmarks/VRPTW/Mamut2026/fastest/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.vrp",
        "meta": "benchmarks/VRPTW/Mamut2026/sidecars/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.meta.json",
        "manifest": "benchmarks/VRPTW/Mamut2026/sidecars/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.manifest.json",
    },
}


@pytest.fixture
def toy_cvrp_instance() -> BenchmarkInstanceCVRP:
    return BenchmarkInstanceCVRP(
        instance_id="mamut-n2-testcvrp",
        instance_origin="OsmCvrpGen",
        benchmark_name="Mamut2026",
        num_customers=2,
        vehicle_capacity=10,
        coordinates=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        demands=[0, 3, 4],
        depot=0,
        arc_costs=[
            [0, 5, 6],
            [5, 0, 3],
            [6, 3, 0],
        ],
        metadata=_TOY_METADATA_CVRP,
    )


@pytest.fixture
def toy_vrptw_instance() -> BenchmarkInstanceVRPTW:
    return BenchmarkInstanceVRPTW(
        instance_id="mamut-n2-testvrptw",
        instance_origin="OsmCvrpGen",
        benchmark_name="Mamut2026",
        num_customers=2,
        vehicle_capacity=10,
        coordinates=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        demands=[0, 3, 4],
        depot=0,
        arc_costs=[
            [0, 5, 6],
            [5, 0, 3],
            [6, 3, 0],
        ],
        service_times=[0, 2, 2],
        time_windows=[(0, 100), (0, 50), (0, 50)],
        metadata=_TOY_METADATA_VRPTW,
    )

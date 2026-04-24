from __future__ import annotations

from pathlib import Path

from mamut_routing_lib import (
    BenchmarkInstanceCVRP,
    BenchmarkInstanceVRPTW,
    BenchmarkSolution,
    DEFAULT_BKS_AUTHORS,
    ObjectiveFunction,
    check_cvrp_solution,
    check_vrptw_solution,
    create_bks_from_solution,
    discover_benchmark_instances,
    save_bks_if_improved,
    save_json_to_file,
)


def make_cvrp_instance() -> BenchmarkInstanceCVRP:
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
        metadata={
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
        },
    )


def make_vrptw_instance() -> BenchmarkInstanceVRPTW:
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
        metadata={
            "authors": "Florian Rascoussier (0nyr) and Adrien Pichon (Anzury)",
            "generated_at": "2026-04-23T00:00:00",
            "problem_type": "VRPTW",
            "metric_variant": "fastest",
            "place_slug": "testville",
            "source_base_name": "test",
            "source_city": "Testville",
            "source_seed": 1,
            "source_folder": "test",
            "num_vehicles_lb": 1,
            "artifact_paths": {
                "vrp_json": "benchmarks/VRPTW/Mamut2026/fastest/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.vrp.json",
                "vrp": "benchmarks/VRPTW/Mamut2026/fastest/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.vrp",
                "meta": "benchmarks/VRPTW/Mamut2026/sidecars/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.meta.json",
                "manifest": "benchmarks/VRPTW/Mamut2026/sidecars/testville/n=2/mamut-n2-testvrptw/mamut-n2-testvrptw.manifest.json",
            },
        },
    )


def test_discover_benchmark_instances_and_problem_specific_layout(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    cvrp_path = benchmark_root / "CVRP" / "Mamut2026" / "fastest" / "testville" / "n=2" / "mamut-n2-testcvrp" / "mamut-n2-testcvrp.vrp.json"
    vrptw_path = benchmark_root / "VRPTW" / "Mamut2026" / "euclidean" / "testville" / "n=2" / "mamut-n2-testvrptw" / "mamut-n2-testvrptw.vrp.json"
    save_json_to_file(make_cvrp_instance().model_dump(mode="json"), cvrp_path)
    save_json_to_file(make_vrptw_instance().model_dump(mode="json"), vrptw_path)

    discovered = discover_benchmark_instances(benchmark_root, benchmark_names=["Mamut2026"])

    assert len(discovered) == 2
    assert {item.problem_type.value for item in discovered} == {"CVRP", "VRPTW"}
    assert {item.instance_id for item in discovered} == {"mamut-n2-testcvrp", "mamut-n2-testvrptw"}


def test_cvrp_and_vrptw_checkers_validate_feasible_solutions() -> None:
    cvrp_solution = BenchmarkSolution(instance_name="mamut-n2-testcvrp", routes=[[1, 2]], cost=14)
    vrptw_solution = BenchmarkSolution(instance_name="mamut-n2-testvrptw", routes=[[1, 2]], cost=14)

    cvrp_check = check_cvrp_solution(make_cvrp_instance(), cvrp_solution)
    vrptw_check = check_vrptw_solution(make_vrptw_instance(), vrptw_solution)

    assert cvrp_check.is_valid()
    assert vrptw_check.is_valid()
    assert cvrp_check.routing_cost == 14
    assert vrptw_check.routing_cost == 14


def test_bks_replacement_only_happens_on_strict_improvement(tmp_path: Path) -> None:
    instance = make_cvrp_instance()
    instance_path = tmp_path / "mamut-n2-testcvrp.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    first_bks = create_bks_from_solution(
        instance,
        BenchmarkSolution(instance_name=instance.instance_id, routes=[[1, 2]], cost=14),
        ObjectiveFunction.MONO_COST,
        authors=DEFAULT_BKS_AUTHORS,
    )
    created = save_bks_if_improved(instance, first_bks, instance_path)
    assert created.action == "created"
    assert created.path.name == "mamut-n2-testcvrp.bks.MonoCost.json"
    assert first_bks.metadata["authors"] == DEFAULT_BKS_AUTHORS

    worse_bks = create_bks_from_solution(
        instance,
        BenchmarkSolution(instance_name=instance.instance_id, routes=[[1], [2]], cost=22),
        ObjectiveFunction.MONO_COST,
        authors=DEFAULT_BKS_AUTHORS,
    )
    kept = save_bks_if_improved(instance, worse_bks, instance_path)
    assert kept.action == "kept_existing"


def test_create_bks_from_solution_requires_authors_metadata() -> None:
    instance = make_cvrp_instance()

    try:
        create_bks_from_solution(
            instance,
            BenchmarkSolution(instance_name=instance.instance_id, routes=[[1, 2]], cost=14),
            ObjectiveFunction.MONO_COST,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("create_bks_from_solution should require authors")

"""Tests for the Ortec2022 5-part path layout and license metadata fields.

The Ortec2022 family introduces a new on-disk layout
``<problem>/<benchmark>/<subset>/n=<N>/<file>.vrp.json`` (the ``subset`` segment
partitions the dataset into ``final``/``public``). It also requires
``license``/``license_url`` to be carried on the typed ``InstanceMetadata`` (so
that Mamut2026-style instances can advertise a license too), while preserving
the historical free-form ``metadata`` dict path for historical instances.

We avoid checking in the upstream 1k+-line TXT fixtures here. Instead, we build
a small synthetic VRPTW instance + BKS that mimics the Ortec2022 dict-metadata
shape and validates strict-``==`` against the checker (mirroring
``project_bks_cost_exactness``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mamut_routing_lib import (
    BenchmarkBKS,
    BenchmarkInstance,
    BenchmarkSolution,
    BenchmarkName,
    ObjectiveFunction,
    check_vrptw_solution,
    discover_benchmark_instances,
    load_benchmark_instance,
    load_bks,
    save_bks,
    save_json_to_file,
)
from mamut_routing_lib.artifacts import build_instance_id, parse_layout
from mamut_routing_lib.cli import _local_instance_record
from mamut_routing_lib.enums import InstanceOrigin
from mamut_routing_lib.models import InstanceMetadata


# --- enum round-trips -------------------------------------------------------


def test_benchmark_name_ortec_2022_enum_value() -> None:
    assert BenchmarkName.ORTEC_2022.value == "Ortec2022"
    assert BenchmarkName("Ortec2022") is BenchmarkName.ORTEC_2022


def test_instance_origin_ortec_2022_enum_value() -> None:
    assert InstanceOrigin.ORTEC_2022.value == "Ortec2022"
    assert InstanceOrigin("Ortec2022") is InstanceOrigin.ORTEC_2022


# --- parse_layout & build_instance_id --------------------------------------


def test_parse_layout_accepts_5_part_subset_layout() -> None:
    relative = Path("VRPTW/Ortec2022/final/n=200/ORTEC-VRPTW-ASYM-x-d1-n212-k20.vrp.json")
    layout = parse_layout(relative, Path("/tmp/benchmarks") / relative)

    assert layout.problem_type.value == "VRPTW"
    assert layout.benchmark_name == "Ortec2022"
    assert layout.subset == "final"
    assert layout.num_customers == 200
    assert layout.instance_name == "ORTEC-VRPTW-ASYM-x-d1-n212-k20"
    assert layout.metric_variant is None
    assert layout.place_slug is None


def test_parse_layout_5_part_public_subset() -> None:
    relative = Path("VRPTW/Ortec2022/public/n=300/ORTEC-VRPTW-ASYM-y-d1-n324-k22.vrp.json")
    layout = parse_layout(relative, Path("/tmp") / relative)

    assert layout.subset == "public"
    assert layout.num_customers == 300


def test_build_instance_id_includes_subset_segment() -> None:
    instance_id = build_instance_id(
        problem_type="VRPTW",
        benchmark_name="Ortec2022",
        subset="final",
        num_customers=200,
        instance_name="ORTEC-VRPTW-ASYM-x-d1-n212-k20",
    )
    assert instance_id == "vrptw-ortec2022-final-n200-ORTEC-VRPTW-ASYM-x-d1-n212-k20"


def test_build_instance_id_subset_distinct_from_historical_form() -> None:
    historical = build_instance_id(
        problem_type="VRPTW",
        benchmark_name="Sintef2008",
        num_customers=100,
        instance_name="C101",
    )
    ortec_final = build_instance_id(
        problem_type="VRPTW",
        benchmark_name="Ortec2022",
        subset="final",
        num_customers=200,
        instance_name="X",
    )
    ortec_public = build_instance_id(
        problem_type="VRPTW",
        benchmark_name="Ortec2022",
        subset="public",
        num_customers=200,
        instance_name="X",
    )
    assert historical == "vrptw-sintef2008-n100-C101"
    assert ortec_final != ortec_public  # subset prevents collisions


# --- InstanceMetadata license fields ---------------------------------------


def _minimal_typed_metadata(**overrides) -> dict:
    base = {
        "authors": "Test",
        "generated_at": "2026-04-23T00:00:00",
        "problem_type": "VRPTW",
        "metric_variant": "fastest",
        "place_slug": "testville",
        "source_base_name": "test",
        "source_city": "Testville",
        "source_seed": 1,
        "source_folder": "test",
        "artifact_paths": {
            "vrp_json": "benchmarks/x/x.vrp.json",
            "vrp": "benchmarks/x/x.vrp",
            "meta": "benchmarks/x/x.meta.json",
            "manifest": "benchmarks/x/x.manifest.json",
        },
    }
    base.update(overrides)
    return base


def test_instance_metadata_accepts_license_fields() -> None:
    meta = InstanceMetadata(
        **_minimal_typed_metadata(
            license="CC BY-NC 4.0",
            license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        )
    )
    assert meta.license == "CC BY-NC 4.0"
    assert meta.license_url == "https://creativecommons.org/licenses/by-nc/4.0/"


def test_instance_metadata_license_defaults_to_none() -> None:
    meta = InstanceMetadata(**_minimal_typed_metadata())
    assert meta.license is None
    assert meta.license_url is None


# --- end-to-end with synthetic Ortec-shaped instance + BKS ----------------


def _ortec_shaped_instance() -> BenchmarkInstance:
    """Tiny VRPTW instance using the Ortec2022 dict-metadata shape.

    Asymmetric arc_costs, num_vehicles=None (unlimited), free-form metadata
    carrying authors/license/subset/provider — same field structure the
    migration script will produce.
    """
    return BenchmarkInstance(
        instance_name="ORTEC-VRPTW-ASYM-test-d1-n2-k2",
        instance_origin="Ortec2022",
        benchmark_name="Ortec2022",
        num_customers=2,
        num_vehicles=None,
        vehicle_capacity=100,
        coordinates=[(0, 0), (10, 0), (0, 10)],
        demands=[0, 5, 5],
        service_times=[0, 60, 60],
        time_windows=[(0, 1000), (0, 500), (0, 500)],
        depot=0,
        arc_costs=[
            [0, 10, 12],   # depot -> 1 = 10, depot -> 2 = 12 (asymmetric)
            [11, 0, 8],
            [13, 9, 0],
        ],
        metadata={
            "authors": "Wouter Kool, Danilo Numeroso, Abdo Abouelrous, Robbert Reijnen",
            "instance_provider": "ORTEC",
            "subset": "final",
            "license": "CC BY-NC 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "source_filename": "ORTEC-VRPTW-ASYM-test-d1-n2-k2.txt",
        },
    )


def test_ortec_shaped_instance_passes_checker_with_exact_cost() -> None:
    instance = _ortec_shaped_instance()
    # depot(0) -> 1 (10) -> 2 (8) -> depot(0) (13) = 31
    solution = BenchmarkSolution(instance_name=instance.instance_name, routes=[[1, 2]], cost=31)
    result = check_vrptw_solution(instance, solution)

    assert result.is_valid(), result.error_message
    assert result.routing_cost == 31  # strict ==
    assert result.num_routes == 1


def test_ortec_shaped_bks_validates_and_round_trips_via_load_bks(tmp_path: Path) -> None:
    instance = _ortec_shaped_instance()
    # 5-part layout target
    instance_path = (
        tmp_path
        / "benchmarks"
        / "VRPTW"
        / "Ortec2022"
        / "final"
        / "n=200"
        / f"{instance.instance_name}.vrp.json"
    )
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    bks = BenchmarkBKS(
        instance_name=instance.instance_name,
        routes=[[1, 2]],
        cost=31,
        objective_function=ObjectiveFunction.MONO_COST,
        metadata={
            "source": "EURO Meets NeurIPS 2022 VRP Competition",
            "status": "competition_bks",
            "license": "CC BY-NC 4.0",
            "validated_cost": 31,
        },
    )
    bks_path = instance_path.parent / f"{instance.instance_name}.bks.MonoCost.json"
    save_bks(bks, bks_path)
    assert bks_path.is_file()

    reloaded = load_bks(bks_path)
    assert reloaded.cost == 31
    assert reloaded.objective_function == ObjectiveFunction.MONO_COST


def test_discover_benchmark_instances_finds_5_part_layout(tmp_path: Path) -> None:
    instance = _ortec_shaped_instance()
    benchmarks_root = tmp_path / "benchmarks"
    instance_path = (
        benchmarks_root
        / "VRPTW"
        / "Ortec2022"
        / "final"
        / "n=200"
        / f"{instance.instance_name}.vrp.json"
    )
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    discovered = discover_benchmark_instances(benchmarks_root)
    assert len(discovered) == 1
    item = discovered[0]
    assert item.benchmark_name == "Ortec2022"
    assert item.subset == "final"
    assert item.num_customers == 200
    assert item.instance_id.startswith("vrptw-ortec2022-final-n200-")
    assert item.instance_name == instance.instance_name


def test_discover_benchmark_instances_distinguishes_final_and_public_subsets(tmp_path: Path) -> None:
    instance = _ortec_shaped_instance()
    benchmarks_root = tmp_path / "benchmarks"

    final_path = (
        benchmarks_root / "VRPTW" / "Ortec2022" / "final" / "n=200" / f"{instance.instance_name}.vrp.json"
    )
    public_path = (
        benchmarks_root / "VRPTW" / "Ortec2022" / "public" / "n=200" / f"{instance.instance_name}.vrp.json"
    )
    save_json_to_file(instance.model_dump(mode="json"), final_path)
    save_json_to_file(instance.model_dump(mode="json"), public_path)

    discovered = discover_benchmark_instances(benchmarks_root)
    subsets = sorted(item.subset for item in discovered if item.subset is not None)
    assert subsets == ["final", "public"]
    ids = {item.instance_id for item in discovered}
    # Subset segment keeps the two IDs distinct despite identical instance_name.
    assert len(ids) == 2


def test_local_instance_record_picks_up_subset_from_5_part_path(tmp_path: Path) -> None:
    instance = _ortec_shaped_instance()
    benchmarks_root = tmp_path / "benchmarks"
    instance_path = (
        benchmarks_root
        / "VRPTW"
        / "Ortec2022"
        / "public"
        / "n=200"
        / f"{instance.instance_name}.vrp.json"
    )
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    record = _local_instance_record(instance_path, instance, benchmarks_root)
    assert record.subset == "public"
    assert record.instance_id == f"vrptw-ortec2022-public-n200-{instance.instance_name}"


def test_load_benchmark_instance_round_trips_ortec_dict_metadata(tmp_path: Path) -> None:
    instance = _ortec_shaped_instance()
    instance_path = (
        tmp_path / "benchmarks" / "VRPTW" / "Ortec2022" / "final" / "n=200" / f"{instance.instance_name}.vrp.json"
    )
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    reloaded = load_benchmark_instance(instance_path)
    # Ortec instances use the dict-metadata branch of the union.
    assert isinstance(reloaded, BenchmarkInstance)
    assert reloaded.metadata["license"] == "CC BY-NC 4.0"
    assert reloaded.metadata["subset"] == "final"
    assert reloaded.num_vehicles is None


def test_unsupported_layout_still_raises() -> None:
    # 3-part path (problem/benchmark/file) was never supported and remains rejected.
    with pytest.raises(ValueError, match="Unsupported benchmark instance layout"):
        parse_layout(Path("VRPTW/Ortec2022/file.vrp.json"), Path("/tmp/VRPTW/Ortec2022/file.vrp.json"))

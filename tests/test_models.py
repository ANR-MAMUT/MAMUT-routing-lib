from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from mamut_routing_lib.enums import MetricVariant, ObjectiveFunction, ProblemType
from mamut_routing_lib.models import (
    ArtifactPaths,
    BenchmarkBKS,
    BenchmarkInstanceVRPTW,
    BenchmarkInstanceCVRP,
    InstanceMetadata,
    OptimalityMetadata,
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
            "vrp_json": "benchmarks/CVRP/Poryos2026/fastest/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.vrp.json",
            "vrp": "benchmarks/CVRP/Poryos2026/fastest/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.vrp",
            "meta": "benchmarks/CVRP/Poryos2026/sidecars/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.meta.json",
            "manifest": "benchmarks/CVRP/Poryos2026/sidecars/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.manifest.json",
        },
        "sibling_variant_paths": {
            "euclidean": "benchmarks/CVRP/Poryos2026/euclidean/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.vrp.json"
        },
        "derived_problem_paths": {
            "fastest": "benchmarks/VRPTW/Poryos2026/fastest/brest/n=2/poryos-n2-beefdad/poryos-n2-beefdad.vrp.json"
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
        "instance_name": "poryos-n2-deadbee",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Poryos2026",
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
    payload["instance_name"] = "poryos-n2-beefdad"
    payload["service_times"] = [0, 10, 20]
    payload["time_windows"] = [(0, 86400), (100, 5000), (200, 6000)]
    payload["metadata"] = make_metadata_kwargs(problem_type="VRPTW", metric_variant="euclidean")
    payload["metadata"]["artifact_paths"] = {
        "vrp_json": "benchmarks/VRPTW/Poryos2026/euclidean/brest/n=2/poryos-n2-beefdad/poryos-n2-beefdad.vrp.json",
        "vrp": "benchmarks/VRPTW/Poryos2026/euclidean/brest/n=2/poryos-n2-beefdad/poryos-n2-beefdad.vrp",
        "meta": "benchmarks/VRPTW/Poryos2026/sidecars/brest/n=2/poryos-n2-beefdad/poryos-n2-beefdad.meta.json",
        "manifest": "benchmarks/VRPTW/Poryos2026/sidecars/brest/n=2/poryos-n2-beefdad/poryos-n2-beefdad.manifest.json",
    }
    payload["metadata"]["source_problem_paths"] = {
        "cvrp_vrp_json": "benchmarks/CVRP/Poryos2026/euclidean/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.vrp.json",
        "cvrp_vrp": "benchmarks/CVRP/Poryos2026/euclidean/brest/n=2/poryos-n2-deadbee/poryos-n2-deadbee.vrp",
    }
    return payload


def test_historical_benchmark_instance_accepts_arc_costs() -> None:
    instance = BenchmarkInstanceVRPTW(**make_valid_historical_instance_kwargs())
    assert instance.arc_costs[0][2] == 2


def test_historical_benchmark_instance_rejects_malformed_arc_costs_matrix() -> None:
    payload = make_valid_historical_instance_kwargs()
    payload["arc_costs"] = [
        [0, 1],
        [1, 0],
    ]

    with pytest.raises(ValidationError):
        BenchmarkInstanceVRPTW(**payload)


def test_cvrp_instance_accepts_structured_metadata_and_float_coordinates() -> None:
    instance = BenchmarkInstanceCVRP(**make_valid_cvrp_instance_kwargs())

    assert instance.metadata.problem_type == ProblemType.CVRP
    assert instance.metadata.metric_variant == MetricVariant.FASTEST
    assert instance.num_vehicles is None
    assert instance.metadata.num_vehicles_lb == 14
    assert instance.coordinates[1] == (1.5, 1.5)


def test_vrptw_instance_accepts_time_fields_and_relative_paths() -> None:
    instance = BenchmarkInstanceVRPTW(**make_valid_vrptw_instance_kwargs())

    assert isinstance(instance.metadata, InstanceMetadata)
    assert instance.metadata.problem_type == ProblemType.VRPTW
    assert instance.time_windows[0] == (0, 86400)


def test_benchmark_instance_metadata_union_resolves_to_dict_for_historical() -> None:
    payload = make_valid_historical_instance_kwargs()
    payload["metadata"] = {"foo": "bar"}

    instance = BenchmarkInstanceVRPTW(**payload)

    assert isinstance(instance.metadata, dict)
    assert instance.metadata == {"foo": "bar"}


def test_benchmark_instance_metadata_union_resolves_to_structured_for_poryos2026() -> None:
    instance = BenchmarkInstanceVRPTW(**make_valid_vrptw_instance_kwargs())

    assert isinstance(instance.metadata, InstanceMetadata)
    assert instance.metadata.place_slug == "brest"


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


def _optimality_payload(**overrides) -> dict:
    payload = {
        "proven": True,
        "prover": "exact solver 1.0",
        "certificate": "optimal under test arithmetic",
        "date": "2026-07-07",
    }
    payload.update(overrides)
    return payload


def _bks_with_metadata(metadata: dict) -> BenchmarkBKS:
    return BenchmarkBKS(
        instance_name="C101",
        objective_function="Duration",
        routes=[[1, 2]],
        cost=12,
        metadata=metadata,
    )


def test_optimality_metadata_requires_core_fields() -> None:
    OptimalityMetadata(**_optimality_payload())
    for missing in ("proven", "prover", "certificate", "date"):
        payload = _optimality_payload()
        del payload[missing]
        with pytest.raises(ValidationError):
            OptimalityMetadata(**payload)


def test_optimality_metadata_proven_must_be_true() -> None:
    with pytest.raises(ValidationError):
        OptimalityMetadata(**_optimality_payload(proven=False))


def test_optimality_metadata_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        OptimalityMetadata(**_optimality_payload(surprise="key"))


def test_solution_metadata_validates_optimality_key() -> None:
    bks = _bks_with_metadata({"optimality": _optimality_payload(proven_optimum=12.0)})
    assert bks.metadata["optimality"]["proven"] is True
    assert bks.metadata["optimality"]["proven_optimum"] == 12.0
    # None-valued optionals are normalized away.
    assert "note" not in bks.metadata["optimality"]

    with pytest.raises(ValidationError):
        _bks_with_metadata({"optimality": {"proven": True}})


def test_solution_metadata_without_optimality_is_untouched() -> None:
    metadata = {"authors": "someone", "source": "somewhere", "seed": 3}
    bks = _bks_with_metadata(metadata)
    assert bks.metadata == metadata


def test_vrptw_public_class_and_deprecated_imports(tmp_path):
    import mamut_routing_lib as library
    from mamut_routing_lib import models

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from mamut_routing_lib import BenchmarkInstanceVRPTW as public_class
        from mamut_routing_lib.models import BenchmarkInstanceVRPTW as model_class

    assert public_class is model_class
    assert public_class.model_json_schema()["title"] == "BenchmarkInstanceVRPTW"
    assert public_class.__name__ == "BenchmarkInstanceVRPTW"
    instance = public_class.model_validate(make_valid_historical_instance_kwargs())
    path = tmp_path / "C101.vrp.json"
    path.write_text(instance.model_dump_json())
    assert type(library.load_benchmark_instance(path)) is public_class
    for module in (library, models):
        with pytest.warns(DeprecationWarning, match="Use BenchmarkInstanceVRPTW") as notice:
            old_class = module.BenchmarkInstance
        assert old_class is public_class
        assert Path(notice[0].filename).resolve() == Path(__file__).resolve()

    assert isinstance(instance, library.LoadedStaticInstance)
    assert isinstance(instance, library.AnyStaticBenchmarkInstance)
    assert isinstance(instance, library.AnyBenchmarkInstance)


def test_general_instance_type_includes_time_dependent_instances():
    from mamut_routing_lib import AnyBenchmarkInstance, AnyStaticBenchmarkInstance
    from mamut_routing_lib.td.models import BenchmarkInstanceTDVRPTW
    from td_utils import toy_instance_payload

    td = BenchmarkInstanceTDVRPTW.model_validate(toy_instance_payload())
    assert isinstance(td, AnyBenchmarkInstance)
    assert not isinstance(td, AnyStaticBenchmarkInstance)


def _collection_vrptw_kwargs() -> dict:
    kwargs = make_valid_historical_instance_kwargs()
    kwargs.pop("arc_costs")
    kwargs.update(benchmark_name="Poryos2026", instance_origin="OsmCvrpGen", metric_variant="euclidean",
                  arc_costs_source={"model": "euclidean"})
    return kwargs


@pytest.mark.parametrize("model,kwargs", [
    (BenchmarkInstanceVRPTW, make_valid_historical_instance_kwargs),
    (BenchmarkInstanceCVRP, make_valid_cvrp_instance_kwargs),
    ("BenchmarkInstanceVRPTWCollection", _collection_vrptw_kwargs),
    ("BenchmarkInstanceTDVRPTW", None),
])
def test_validation_rejects_a_depot_other_than_vertex_zero(model, kwargs):
    if isinstance(model, str):
        import mamut_routing_lib.models as static_models
        import mamut_routing_lib.td.models as td_models

        model = getattr(static_models, model, None) or getattr(td_models, model)
    if kwargs is None:
        from td_utils import toy_instance_payload

        payload = toy_instance_payload()
    else:
        payload = kwargs()
    model(**payload)
    with pytest.raises(ValidationError, match="depot must be vertex 0"):
        model(**{**payload, "depot": 1})


@pytest.mark.parametrize("model,kwargs", [
    (BenchmarkInstanceVRPTW, make_valid_historical_instance_kwargs),
    ("BenchmarkInstanceVRPTWCollection", _collection_vrptw_kwargs),
    ("BenchmarkInstanceTDVRPTW", None),
    ("BenchmarkInstanceTDVRP", None),
])
def test_validation_rejects_a_nonzero_depot_service_time(model, kwargs):
    import mamut_routing_lib.models as static_models
    import mamut_routing_lib.td.models as td_models

    if isinstance(model, str):
        model = getattr(static_models, model, None) or getattr(td_models, model)
    if kwargs is None:
        from td_utils import toy_instance_payload

        payload = toy_instance_payload(with_time_windows=model is td_models.BenchmarkInstanceTDVRPTW)
    else:
        payload = kwargs()
    model(**payload)
    services = list(payload["service_times"])
    services[0] = 0.5
    with pytest.raises(ValidationError, match="depot service time must be zero"):
        model(**{**payload, "service_times": services})


def test_load_rejects_a_descriptor_with_a_moved_depot(tmp_path):
    import json

    from mamut_routing_lib.artifacts import load_benchmark_instance

    path = tmp_path / "C101.vrp.json"
    path.write_text(json.dumps({**make_valid_historical_instance_kwargs(), "depot": 2}))
    with pytest.raises(ValidationError, match="depot must be vertex 0"):
        load_benchmark_instance(path)

"""Tests for family-first collection discovery and slim static instances."""

from __future__ import annotations

import math

import pytest

from mamut_routing_lib import (
    BenchmarkInstance,
    BenchmarkInstanceCVRPCollection,
    BenchmarkInstanceVRPTWCollection,
    MetricVariant,
    ProblemType,
    discover_benchmark_instances,
    load_benchmark_instance,
    resolve_arc_costs,
)
from mamut_routing_lib.artifacts import parse_collection_layout
from mamut_routing_lib.distances import (
    InstanceDistances,
    compute_distances_sha256,
    save_instance_distances,
)
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.sidecars import CollectionMarker, save_collection_marker
from pathlib import Path

BASE = "mamut-toyville-n2-hyb"
CITY = "toyville"


def static_payload(*, metric: str, arc_costs_source: dict, with_tw: bool = False) -> dict:
    payload = {
        "instance_name": BASE,
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Mamut2026",
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 10,
        "coordinates": [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
        "demands": [0, 4, 4],
        "depot": 0,
        "metric_variant": metric,
        "arc_costs_source": arc_costs_source,
        "metadata": {"base_instance_name": BASE},
    }
    if with_tw:
        payload["service_times"] = [0, 10, 10]
        payload["time_windows"] = [[0, 86400], [100, 5000], [200, 6000]]
    return payload


def td_payload(subinstance: str, *, with_tw: bool) -> dict:
    payload = {
        "instance_name": f"{BASE}-{subinstance}",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Mamut2026",
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 10,
        "coordinates": [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
        "demands": [0, 4, 4],
        "service_times": [0, 10, 10],
        "depot": 0,
        "horizon": [0.0, 86400.0],
        "td": {
            "model": "road-graph",
            "graph": {"path": f"sidecars/{CITY}/n=2/{BASE}/{BASE}.road.json.gz"},
            "traffic": {"path": f"sidecars/{CITY}/n=2/{BASE}/{BASE}.traffic-bpr-heavy.json.gz"},
            "sample_step": 60.0,
            "simplify_tolerance": 1.0,
        },
        "metadata": {"base_instance_name": BASE, "subinstance": subinstance},
    }
    if with_tw:
        payload["time_windows"] = [[0, 86400], [100, 50000], [200, 60000]]
    return payload


@pytest.fixture()
def benchmarks_root(tmp_path) -> Path:
    root = tmp_path / "benchmarks"

    # Family-first collection.
    collection = root / "Mamut2026"
    save_collection_marker(CollectionMarker(family="Mamut2026"), collection)
    distances = InstanceDistances(
        base_name=BASE,
        benchmark_name="Mamut2026",
        metric="fastest",
        num_customers=2,
        values=[[0.0, 120.5, 240.25], [120.5, 0.0, 130.75], [240.25, 130.75, 0.0]],
    )
    distances_rel = f"sidecars/{CITY}/n=2/{BASE}/{BASE}.distances-fastest.json.gz"
    save_instance_distances(distances, collection / distances_rel)
    distances_sha = compute_distances_sha256(distances)

    save_json_to_file(
        static_payload(metric="euclidean", arc_costs_source={"model": "euclidean", "decimals": 3}),
        collection / "CVRP" / "euclidean" / CITY / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    save_json_to_file(
        static_payload(
            metric="fastest",
            arc_costs_source={
                "model": "distances-sidecar",
                "distances": {"path": distances_rel, "sha256": distances_sha},
            },
        ),
        collection / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    save_json_to_file(
        static_payload(
            metric="fastest",
            arc_costs_source={
                "model": "distances-sidecar",
                "distances": {"path": distances_rel, "sha256": distances_sha},
            },
            with_tw=True,
        ),
        collection / "VRPTW" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    save_json_to_file(
        td_payload("bpr-heavy", with_tw=False),
        collection / "TDVRP" / CITY / "n=2" / BASE / "bpr-heavy" / f"{BASE}-bpr-heavy.vrp.json",
    )
    save_json_to_file(
        td_payload("bpr-heavy", with_tw=True),
        collection / "TDVRPTW" / CITY / "n=2" / BASE / "bpr-heavy" / f"{BASE}-bpr-heavy.vrp.json",
    )

    # Historic 4-part layout family next to the collection (regression).
    historic = {
        "instance_name": "toy-h1",
        "instance_origin": "Solomon1987",
        "benchmark_name": "Sintef2008",
        "num_customers": 2,
        "num_vehicles": 2,
        "vehicle_capacity": 10,
        "coordinates": [[0, 0], [1, 1], [2, 2]],
        "demands": [0, 4, 4],
        "depot": 0,
        "arc_costs": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "service_times": [0, 10, 10],
        "time_windows": [[0, 100], [0, 100], [0, 100]],
        "metadata": {},
    }
    save_json_to_file(historic, root / "VRPTW" / "Sintef2008" / "n=2" / "toy-h1.vrp.json")
    return root


class TestCollectionDiscovery:
    def test_discovers_collection_and_historic_together(self, benchmarks_root):
        items = discover_benchmark_instances(benchmarks_root)
        assert len(items) == 6
        by_pt = {}
        for item in items:
            by_pt.setdefault(item.problem_type, []).append(item)
        assert len(by_pt[ProblemType.CVRP]) == 2
        assert len(by_pt[ProblemType.VRPTW]) == 2  # collection + historic
        assert len(by_pt[ProblemType.TDVRP]) == 1
        assert len(by_pt[ProblemType.TDVRPTW]) == 1

    def test_collection_items_expose_base_and_subinstance(self, benchmarks_root):
        items = discover_benchmark_instances(benchmarks_root, problem_types=[ProblemType.TDVRP])
        (item,) = items
        assert item.benchmark_name == "Mamut2026"
        assert item.base_instance_name == BASE
        assert item.subinstance == "bpr-heavy"
        assert item.instance_name == f"{BASE}-bpr-heavy"
        assert item.place_slug == CITY
        assert item.num_customers == 2
        assert item.metric_variant is None

    def test_static_items_expose_metric_and_base(self, benchmarks_root):
        items = discover_benchmark_instances(
            benchmarks_root,
            problem_types=[ProblemType.CVRP],
            metric_variants=[MetricVariant.FASTEST],
        )
        (item,) = items
        assert item.base_instance_name == BASE
        assert item.subinstance is None
        assert item.metric_variant == MetricVariant.FASTEST

    def test_historic_layout_unaffected(self, benchmarks_root):
        items = discover_benchmark_instances(benchmarks_root, benchmark_names=["Sintef2008"])
        (item,) = items
        assert item.problem_type == ProblemType.VRPTW
        assert item.base_instance_name is None
        assert isinstance(item.load(), BenchmarkInstance)

    def test_instance_ids_are_unique(self, benchmarks_root):
        items = discover_benchmark_instances(benchmarks_root)
        ids = [item.instance_id for item in items]
        assert len(set(ids)) == len(ids)

    def test_collection_checkout_scanned_as_root(self, benchmarks_root):
        # A standalone collection checkout (marker at the scanned root itself)
        # must discover the same collection instances.
        items = discover_benchmark_instances(benchmarks_root / "Mamut2026")
        assert len(items) == 5
        assert {item.benchmark_name for item in items} == {"Mamut2026"}


class TestCollectionLayoutParser:
    def test_td_name_must_compose_base_and_sub(self):
        with pytest.raises(ValueError, match="does not equal"):
            parse_collection_layout(
                Path("TDVRP/lyon/n=10/base-a/sub-b/wrong-name.vrp.json"),
                Path("/x/wrong-name.vrp.json"),
                "Mamut2026",
            )

    def test_static_name_must_equal_base_dir(self):
        with pytest.raises(ValueError, match="does not equal"):
            parse_collection_layout(
                Path("CVRP/fastest/lyon/n=10/base-a/other.vrp.json"),
                Path("/x/other.vrp.json"),
                "Mamut2026",
            )

    def test_wrong_depth_rejected(self):
        with pytest.raises(ValueError, match="Unsupported collection"):
            parse_collection_layout(
                Path("CVRP/fastest/n=10/base/base.vrp.json"),
                Path("/x/base.vrp.json"),
                "Mamut2026",
            )


class TestSlimInstances:
    def test_load_dispatches_to_collection_models(self, benchmarks_root):
        collection = benchmarks_root / "Mamut2026"
        cvrp = load_benchmark_instance(
            collection / "CVRP" / "euclidean" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        )
        assert isinstance(cvrp, BenchmarkInstanceCVRPCollection)
        vrptw = load_benchmark_instance(
            collection / "VRPTW" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        )
        assert isinstance(vrptw, BenchmarkInstanceVRPTWCollection)
        assert vrptw.time_windows[1] == (100, 5000)

    def test_euclidean_hydration(self, benchmarks_root):
        path = benchmarks_root / "Mamut2026" / "CVRP" / "euclidean" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        matrix = resolve_arc_costs(instance, path)
        assert matrix[0][1] == round(math.hypot(3.0, 4.0), 3) == 5.0
        assert matrix[0][2] == 10.0
        assert matrix[1][1] == 0.0

    def test_distances_sidecar_hydration_with_sha(self, benchmarks_root):
        path = benchmarks_root / "Mamut2026" / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        matrix = resolve_arc_costs(instance, path)
        assert matrix[0][1] == 120.5
        assert matrix[2][1] == 130.75

    def test_sha_mismatch_raises(self, benchmarks_root, tmp_path):
        path = benchmarks_root / "Mamut2026" / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        bad = instance.model_copy(deep=True)
        bad.arc_costs_source.distances.sha256 = "0" * 64
        with pytest.raises(ValueError, match="sha256 mismatch"):
            resolve_arc_costs(bad, path)

    def test_metric_mismatch_raises(self, benchmarks_root):
        path = benchmarks_root / "Mamut2026" / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        bad = instance.model_copy(deep=True, update={"metric_variant": MetricVariant.SHORTEST})
        with pytest.raises(ValueError, match="does not match instance"):
            resolve_arc_costs(bad, path)

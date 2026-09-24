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
from pathlib import Path

from collection_utils import BASE, CITY, write_toy_collection


@pytest.fixture()
def benchmarks_root(tmp_path) -> Path:
    return write_toy_collection(tmp_path)


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
        assert item.benchmark_name == "Poryos2026"
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
        items = discover_benchmark_instances(benchmarks_root / "Poryos2026")
        assert len(items) == 5
        assert {item.benchmark_name for item in items} == {"Poryos2026"}


class TestCollectionLayoutParser:
    def test_td_name_must_compose_base_and_sub(self):
        with pytest.raises(ValueError, match="does not equal"):
            parse_collection_layout(
                Path("TDVRP/lyon/n=10/base-a/sub-b/wrong-name.vrp.json"),
                Path("/x/wrong-name.vrp.json"),
                "Poryos2026",
            )

    def test_static_name_must_equal_base_dir(self):
        with pytest.raises(ValueError, match="does not equal"):
            parse_collection_layout(
                Path("CVRP/fastest/lyon/n=10/base-a/other.vrp.json"),
                Path("/x/other.vrp.json"),
                "Poryos2026",
            )

    def test_wrong_depth_rejected(self):
        with pytest.raises(ValueError, match="Unsupported collection"):
            parse_collection_layout(
                Path("CVRP/fastest/n=10/base/base.vrp.json"),
                Path("/x/base.vrp.json"),
                "Poryos2026",
            )

    def test_vrptw_tw_set_suffix_accepted(self):
        layout = parse_collection_layout(
            Path("VRPTW/fastest/lyon/n=10/base-a/base-a-tw-tight.vrp.json"),
            Path("/x/base-a-tw-tight.vrp.json"),
            "Poryos2026",
        )
        assert layout.instance_name == "base-a-tw-tight"
        assert layout.base_instance_name == "base-a"
        assert layout.tw_set == "tight"

    def test_vrptw_bare_name_is_td_shared(self):
        layout = parse_collection_layout(
            Path("VRPTW/fastest/lyon/n=10/base-a/base-a.vrp.json"),
            Path("/x/base-a.vrp.json"),
            "Poryos2026",
        )
        assert layout.tw_set == "td-shared"

    def test_cvrp_rejects_tw_set_suffix(self):
        with pytest.raises(ValueError, match="does not equal"):
            parse_collection_layout(
                Path("CVRP/fastest/lyon/n=10/base-a/base-a-tw-tight.vrp.json"),
                Path("/x/base-a-tw-tight.vrp.json"),
                "Poryos2026",
            )

    def test_vrptw_empty_tw_tag_rejected(self):
        with pytest.raises(ValueError, match="does not equal"):
            parse_collection_layout(
                Path("VRPTW/fastest/lyon/n=10/base-a/base-a-tw-.vrp.json"),
                Path("/x/base-a-tw-.vrp.json"),
                "Poryos2026",
            )


class TestSlimInstances:
    def test_load_dispatches_to_collection_models(self, benchmarks_root):
        collection = benchmarks_root / "Poryos2026"
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
        path = benchmarks_root / "Poryos2026" / "CVRP" / "euclidean" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        matrix = resolve_arc_costs(instance, path)
        assert matrix[0][1] == round(math.hypot(3.0, 4.0), 3) == 5.0
        assert matrix[0][2] == 10.0
        assert matrix[1][1] == 0.0

    def test_distances_sidecar_hydration_with_sha(self, benchmarks_root):
        path = benchmarks_root / "Poryos2026" / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        matrix = resolve_arc_costs(instance, path)
        assert matrix[0][1] == 120.5
        assert matrix[2][1] == 130.75

    def test_sha_mismatch_raises(self, benchmarks_root, tmp_path):
        path = benchmarks_root / "Poryos2026" / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        bad = instance.model_copy(deep=True)
        bad.arc_costs_source.distances.sha256 = "0" * 64
        with pytest.raises(ValueError, match="sha256 mismatch"):
            resolve_arc_costs(bad, path)

    def test_metric_mismatch_raises(self, benchmarks_root):
        path = benchmarks_root / "Poryos2026" / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json"
        instance = load_benchmark_instance(path)
        bad = instance.model_copy(deep=True, update={"metric_variant": MetricVariant.SHORTEST})
        with pytest.raises(ValueError, match="does not match instance"):
            resolve_arc_costs(bad, path)

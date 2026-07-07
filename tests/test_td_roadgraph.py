"""Tests for the road-graph td model: sidecar, pinned algorithms, materialization, loader."""

from __future__ import annotations

import json

import pytest

from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.td import (
    ATFFormatError,
    InstanceRoadGraph,
    NDCPWLF,
    RoadGraphFormatError,
    TDRoadGraphRef,
    build_adjacency,
    compute_atf_sha256,
    compute_fastest_path_tree,
    compute_road_graph_sha256,
    ichoua_travel_time,
    load_instance_road_graph,
    load_td_instance,
    materialize_instance_atfs_roadgraph,
    save_instance_road_graph,
    simplify_ndcpwlf,
    td_instance_from_payload,
)

HORIZON = (0.0, 300.0)
BIN_EDGES = [0.0, 100.0, 200.0, 300.0]
EXTENSION_END = 900.0
SAMPLE_STEP = 2.0

# Diamond graph: depot vertex 0, customers at vertices 1 and 3. The fastest
# free-flow path 0 -> 3 goes through vertex 1 (200) rather than 2 (300).
EDGES = [
    (0, 1, 100.0, [1.0, 0.5, 1.0]),
    (0, 2, 150.0, [1.0, 1.0, 1.0]),
    (1, 0, 100.0, [1.0, 0.5, 1.0]),
    (1, 3, 100.0, [1.0, 0.5, 1.0]),
    (2, 0, 150.0, [1.0, 1.0, 1.0]),
    (2, 3, 150.0, [1.0, 1.0, 1.0]),
    (3, 1, 100.0, [1.0, 0.5, 1.0]),
    (3, 2, 150.0, [1.0, 1.0, 1.0]),
]


def make_road_graph(*, tolerance: float = 0.0, extension_end: float = EXTENSION_END) -> InstanceRoadGraph:
    return InstanceRoadGraph(
        base_name="mamut-toy",
        benchmark_name="Mamut2026",
        num_customers=2,
        horizon=HORIZON,
        extension_end=extension_end,
        bin_edges=list(BIN_EDGES),
        sample_step=SAMPLE_STEP,
        simplify_tolerance=tolerance,
        num_vertices=4,
        vertex_osm_ids=[11, 22, 33, 44],
        node_vertices=[0, 1, 3],
        edges=[(u, v, length, list(speeds)) for u, v, length, speeds in EDGES],
        generator={"name": "test-fixture", "model": "wave", "intensity": "heavy", "seed": 42},
    )


def road_instance_payload(road: InstanceRoadGraph, *, with_time_windows: bool = True) -> dict:
    payload = {
        "instance_name": "mamut-toy-n2-wave-heavy-poi",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Mamut2026",
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 10,
        "coordinates": [[0, 0], [100, 0], [200, 0]],
        "demands": [0, 4, 4],
        "service_times": [0, 10, 10],
        "depot": 0,
        "horizon": [0.0, 300.0],
        "td": {
            "model": "road-graph",
            "graph_path": "mamut-toy.road.json",
            "graph_sha256": compute_road_graph_sha256(road),
        },
        "metadata": {},
    }
    if with_time_windows:
        payload["time_windows"] = [[0, 300], [0, 300], [0, 300]]
    return payload


def write_road_instance_files(directory, *, gzip_sidecar: bool = False, with_time_windows: bool = True):
    road = make_road_graph()
    sidecar_name = "mamut-toy.road.json.gz" if gzip_sidecar else "mamut-toy.road.json"
    save_instance_road_graph(road, directory / sidecar_name)

    payload = road_instance_payload(road, with_time_windows=with_time_windows)
    payload["td"]["graph_path"] = sidecar_name
    instance = td_instance_from_payload(payload)
    payload["td"]["atf_sha256"] = compute_atf_sha256(materialize_instance_atfs_roadgraph(instance, road))
    instance_path = directory / "mamut-toy-n2-wave-heavy-poi.vrp.json"
    save_json_to_file(payload, instance_path)
    return instance_path


class TestSimplify:
    def make_wiggly(self) -> NDCPWLF:
        xs = [float(k) for k in range(21)]
        ys = []
        y = 0.0
        for k in range(21):
            y += 1.0 + (0.4 if k % 3 == 0 else 0.05 if k % 3 == 1 else 0.0)
            ys.append(y)
        return NDCPWLF(xs, ys)

    def test_keeps_endpoints_and_monotone_subset(self):
        f = self.make_wiggly()
        g = simplify_ndcpwlf(f, 0.5)
        assert g.xs[0] == f.xs[0] and g.xs[-1] == f.xs[-1]
        assert g.ys[0] == f.ys[0] and g.ys[-1] == f.ys[-1]
        assert g.num_breakpoints() < f.num_breakpoints()
        original = set(zip(f.xs, f.ys))
        assert all(point in original for point in zip(g.xs, g.ys))
        assert all(g.ys[k] >= g.ys[k - 1] for k in range(1, g.num_breakpoints()))

    def test_vertical_deviation_bounded(self):
        f = self.make_wiggly()
        tolerance = 0.3
        g = simplify_ndcpwlf(f, tolerance)
        for k in range(f.num_breakpoints()):
            assert abs(g.evaluate(f.xs[k]) - f.ys[k]) <= tolerance + 1e-12

    def test_deterministic(self):
        f = self.make_wiggly()
        a = simplify_ndcpwlf(f, 0.3)
        b = simplify_ndcpwlf(f, 0.3)
        assert a.xs == b.xs and a.ys == b.ys

    def test_zero_tolerance_drops_only_collinear(self):
        f = NDCPWLF([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0])  # first two pieces collinear
        g = simplify_ndcpwlf(f, 0.0)
        assert g.xs == [0.0, 2.0, 3.0]
        assert g.ys == [0.0, 2.0, 4.0]
        for x in [0.0, 0.5, 1.5, 2.5, 3.0]:
            assert g.evaluate(x) == f.evaluate(x)

    def test_interior_step_smoothed_within_tolerance(self):
        f = NDCPWLF([0.0, 1.0, 1.0, 1.0, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0])
        g = simplify_ndcpwlf(f, 10.0)
        # A step strictly inside a kept pair may be smoothed away entirely...
        assert g.xs == [0.0, 2.0] and g.ys == [0.0, 2.0]
        # ...with the vertical deviation still bounded by the tolerance.
        assert all(abs(g.evaluate(x) - y) <= 10.0 for x, y in zip(f.xs, f.ys))

    def test_pure_vertical_run_kept_whole(self):
        f = NDCPWLF([1.0, 1.0, 1.0], [0.0, 5.0, 10.0])
        g = simplify_ndcpwlf(f, 100.0)
        assert g.xs == f.xs and g.ys == f.ys

    def test_short_functions_untouched(self):
        f = NDCPWLF([0.0, 10.0], [5.0, 15.0])
        assert simplify_ndcpwlf(f, 1.0) is f


class TestPinnedDijkstra:
    def test_tie_break_is_canonical(self):
        # Two equal-cost paths 0->1->3 and 0->2->3; the pinned rule (heap
        # orders by (dist, vertex), strict-< updates) must pick the path
        # through vertex 1 and never revise it.
        road = InstanceRoadGraph(
            base_name="tie",
            benchmark_name="Mamut2026",
            num_customers=1,
            horizon=HORIZON,
            extension_end=EXTENSION_END,
            bin_edges=list(BIN_EDGES),
            sample_step=SAMPLE_STEP,
            simplify_tolerance=0.0,
            num_vertices=4,
            vertex_osm_ids=[1, 2, 3, 4],
            node_vertices=[0, 3],
            edges=[
                (0, 1, 100.0, [1.0, 1.0, 1.0]),
                (0, 2, 100.0, [1.0, 1.0, 1.0]),
                (1, 3, 100.0, [1.0, 1.0, 1.0]),
                (2, 3, 100.0, [1.0, 1.0, 1.0]),
                (3, 0, 100.0, [1.0, 1.0, 1.0]),
            ],
        )
        dist, pred = compute_fastest_path_tree(road, build_adjacency(road), 0)
        assert dist[3] == 200.0
        assert road.edges[pred[3]][0] == 1  # tree edge into 3 comes from vertex 1

    def test_free_flow_uses_max_bin_speed(self):
        road = make_road_graph()
        dist, pred = compute_fastest_path_tree(road, build_adjacency(road), 0)
        # Edge (0,1) free-flow weight = 100 / max(1.0, 0.5, 1.0) = 100.
        assert dist[1] == 100.0
        assert dist[3] == 200.0  # via vertex 1, not the 300 route via 2
        assert road.edges[pred[3]][0] == 1

    def test_unreachable_target_raises(self):
        road = make_road_graph()
        # Cut vertex 3 off: keep only edges leaving 3 so it stays present.
        road.edges = [e for e in road.edges if e[1] != 3]
        instance = td_instance_from_payload(road_instance_payload(make_road_graph()))
        with pytest.raises(RoadGraphFormatError, match="unreachable"):
            materialize_instance_atfs_roadgraph(instance, road)


class TestRoadGraphSidecar:
    def test_roundtrip_plain_and_gzip(self, tmp_path):
        road = make_road_graph()
        for name in ["g.road.json", "g.road.json.gz"]:
            save_instance_road_graph(road, tmp_path / name)
            loaded = load_instance_road_graph(tmp_path / name)
            assert compute_road_graph_sha256(loaded) == compute_road_graph_sha256(road)
            assert loaded.edges == road.edges
            assert loaded.node_vertices == road.node_vertices

    def test_sha_is_storage_form_independent(self, tmp_path):
        road = make_road_graph()
        save_instance_road_graph(road, tmp_path / "a.road.json")
        save_instance_road_graph(road, tmp_path / "b.road.json.gz")
        plain = load_instance_road_graph(tmp_path / "a.road.json")
        gzipped = load_instance_road_graph(tmp_path / "b.road.json.gz")
        assert compute_road_graph_sha256(plain) == compute_road_graph_sha256(gzipped)

    def test_int_inputs_are_coerced_to_canonical_floats(self):
        road = make_road_graph()
        as_ints = InstanceRoadGraph(
            base_name=road.base_name,
            benchmark_name=road.benchmark_name,
            num_customers=road.num_customers,
            horizon=(0, 300),
            extension_end=900,
            bin_edges=[0, 100, 200, 300],
            sample_step=2,
            simplify_tolerance=0,
            num_vertices=road.num_vertices,
            vertex_osm_ids=road.vertex_osm_ids,
            node_vertices=road.node_vertices,
            edges=[(u, v, int(length), [s for s in speeds]) for u, v, length, speeds in road.edges],
            generator=road.generator,
        )
        assert compute_road_graph_sha256(as_ints) == compute_road_graph_sha256(road)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda r: r.edges.__setitem__(0, (1, 1, 100.0, [1.0, 1.0, 1.0])), "self-loop"),
            (lambda r: r.edges.reverse(), "sorted strictly increasing"),
            (lambda r: r.edges.__setitem__(0, (0, 1, 100.0, [1.0, 0.0, 1.0])), "strictly positive"),
            (lambda r: r.edges.__setitem__(0, (0, 1, 100.0, [1.0, 1.0])), "expected one per bin"),
            (lambda r: r.edges.__setitem__(0, (0, 1, 0.0, [1.0, 1.0, 1.0])), "length"),
            (lambda r: setattr(r, "bin_edges", [0.0, 100.0, 250.0]), "span exactly the horizon"),
            (lambda r: setattr(r, "node_vertices", [0, 1, 1]), "distinct"),
            (lambda r: setattr(r, "vertex_osm_ids", [44, 22, 33, 11]), "strictly increasing"),
            (lambda r: setattr(r, "extension_end", 300.0), "extension_end"),
            (lambda r: setattr(r, "simplify_tolerance", -1.0), "simplify_tolerance"),
            (lambda r: setattr(r, "sample_step", 7.0), "tile the horizon"),
        ],
    )
    def test_validator_rejections(self, tmp_path, mutate, match):
        road = make_road_graph()
        mutate(road)
        with pytest.raises(RoadGraphFormatError, match=match):
            save_instance_road_graph(road, tmp_path / "bad.road.json")

    def test_bad_suffix_rejected(self, tmp_path):
        with pytest.raises(RoadGraphFormatError, match="must end with"):
            save_instance_road_graph(make_road_graph(), tmp_path / "g.json")


class TestMaterialization:
    def path_edges(self, source: int, target: int) -> list[tuple[float, list[float]]]:
        # Fixture knowledge: fastest paths in the diamond go through vertex 1.
        by_key = {(u, v): (length, speeds) for u, v, length, speeds in EDGES}
        chains = {
            (0, 1): [(0, 1)],
            (0, 3): [(0, 1), (1, 3)],
            (1, 0): [(1, 0)],
            (1, 3): [(1, 3)],
            (3, 0): [(3, 1), (1, 0)],
            (3, 1): [(3, 1)],
        }
        return [by_key[step] for step in chains[(source, target)]]

    def manual_arrival(self, source_vertex: int, target_vertex: int, departure: float) -> float:
        zones = [(BIN_EDGES[k], BIN_EDGES[k + 1]) for k in range(len(BIN_EDGES) - 1)]
        t = departure
        for length, speeds in self.path_edges(source_vertex, target_vertex):
            t = t + ichoua_travel_time(zones, speeds, length, t)
        return t

    def test_exact_against_manual_walk_at_grid_points(self):
        # With tolerance 0 the decimation drops only exactly-collinear
        # samples, so the ATF reproduces the exact edge-by-edge arrival at
        # every grid departure time — no accumulated error by construction.
        road = make_road_graph(tolerance=0.0)
        instance = td_instance_from_payload(road_instance_payload(road))
        atfs = materialize_instance_atfs_roadgraph(instance, road)
        node_to_vertex = {0: 0, 1: 1, 2: 3}
        for (i, j), atf in atfs.arcs.items():
            for k in range(151):
                departure = HORIZON[0] + k * SAMPLE_STEP
                expected = self.manual_arrival(node_to_vertex[i], node_to_vertex[j], departure)
                assert atf.evaluate(departure) == pytest.approx(expected, abs=1e-9)

    def test_numpy_and_reference_paths_bit_identical(self, monkeypatch):
        import mamut_routing_lib.td.roadgraph as roadgraph_module

        if roadgraph_module._np is None:
            pytest.skip("numpy not installed")
        road = make_road_graph(tolerance=0.5)
        instance = td_instance_from_payload(road_instance_payload(road))
        fast = compute_atf_sha256(materialize_instance_atfs_roadgraph(instance, road))
        monkeypatch.setattr(roadgraph_module, "_np", None)
        reference = compute_atf_sha256(materialize_instance_atfs_roadgraph(instance, road))
        assert fast == reference

    def test_arcs_complete_and_span_horizon(self):
        road = make_road_graph(tolerance=0.1)
        instance = td_instance_from_payload(road_instance_payload(road))
        atfs = materialize_instance_atfs_roadgraph(instance, road)
        assert set(atfs.arcs) == {(i, j) for i in range(3) for j in range(3) if i != j}
        for atf in atfs.arcs.values():
            assert atf.xs[0] == HORIZON[0]
            assert atf.xs[-1] == HORIZON[1]
            assert all(y >= x for x, y in zip(atf.xs, atf.ys))

    def test_time_dependence_is_present(self):
        road = make_road_graph()
        instance = td_instance_from_payload(road_instance_payload(road))
        atfs = materialize_instance_atfs_roadgraph(instance, road)
        atf = atfs.arcs[(0, 1)]
        off_peak = atf.evaluate(0.0) - 0.0
        peak = atf.evaluate(150.0) - 150.0
        assert peak > off_peak  # midday congestion on edge (0, 1)

    def test_tolerance_shrinks_breakpoints_with_bounded_deviation(self):
        exact = materialize_instance_atfs_roadgraph(
            td_instance_from_payload(road_instance_payload(make_road_graph(tolerance=0.0))),
            make_road_graph(tolerance=0.0),
        )
        coarse = materialize_instance_atfs_roadgraph(
            td_instance_from_payload(road_instance_payload(make_road_graph(tolerance=0.5))),
            make_road_graph(tolerance=0.5),
        )
        total_exact = sum(atf.num_breakpoints() for atf in exact.arcs.values())
        total_coarse = sum(atf.num_breakpoints() for atf in coarse.arcs.values())
        assert total_coarse <= total_exact
        for key in exact.arcs:
            for k in range(76):
                x = HORIZON[0] + k * (HORIZON[1] - HORIZON[0]) / 75
                # Loose engineering bound: per-step tolerance 0.5 over <= 2
                # compositions, slopes bounded by the 2x speed ratio.
                assert abs(coarse.arcs[key].evaluate(x) - exact.arcs[key].evaluate(x)) <= 2.5

    def test_materialization_is_deterministic(self):
        road = make_road_graph(tolerance=0.1)
        instance = td_instance_from_payload(road_instance_payload(road))
        first = compute_atf_sha256(materialize_instance_atfs_roadgraph(instance, road))
        second = compute_atf_sha256(materialize_instance_atfs_roadgraph(instance, road))
        assert first == second

    def test_extension_end_too_small_raises(self):
        road = make_road_graph(extension_end=310.0)
        instance = td_instance_from_payload(road_instance_payload(road))
        with pytest.raises(RoadGraphFormatError, match="extension_end too small"):
            materialize_instance_atfs_roadgraph(instance, road)

    def test_wrong_model_rejected(self):
        road = make_road_graph()
        payload = road_instance_payload(road)
        payload["td"] = {"model": "atf-ndcpwlf", "atf_path": "x.atf.json"}
        instance = td_instance_from_payload(payload)
        with pytest.raises(RoadGraphFormatError, match="expected road-graph"):
            materialize_instance_atfs_roadgraph(instance, road)


class TestModelRef:
    def test_discriminated_union_dispatch(self):
        road = make_road_graph()
        instance = td_instance_from_payload(road_instance_payload(road))
        assert isinstance(instance.td, TDRoadGraphRef)

    def test_graph_path_must_be_plain_relative(self):
        for bad in ["/abs/g.road.json", "../g.road.json", ""]:
            with pytest.raises(ValueError):
                TDRoadGraphRef(graph_path=bad)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            TDRoadGraphRef(graph_path="g.road.json", surprise=1)


class TestLoader:
    def test_load_roundtrip_with_sha_verification(self, tmp_path):
        instance_path = write_road_instance_files(tmp_path)
        loaded = load_td_instance(instance_path, verify_sha256=True)
        assert loaded.atf_path is None
        assert loaded.road_graph_path is not None
        assert loaded.road_graph_path.name == "mamut-toy.road.json"
        assert len(loaded.atfs.arcs) == 6
        assert loaded.atfs.generator == {"name": "road-graph-materializer", "version": 1}

    def test_load_gzip_sidecar(self, tmp_path):
        instance_path = write_road_instance_files(tmp_path, gzip_sidecar=True)
        loaded = load_td_instance(instance_path, verify_sha256=True)
        assert loaded.road_graph_path.name == "mamut-toy.road.json.gz"

    def test_tdvrp_twin_loads(self, tmp_path):
        instance_path = write_road_instance_files(tmp_path, with_time_windows=False)
        loaded = load_td_instance(instance_path, verify_sha256=True)
        assert not hasattr(loaded.instance, "time_windows")

    def test_graph_sha_mismatch_raises(self, tmp_path):
        instance_path = write_road_instance_files(tmp_path)
        payload = json.loads(instance_path.read_text())
        payload["td"]["graph_sha256"] = "0" * 64
        save_json_to_file(payload, instance_path)
        with pytest.raises(ATFFormatError, match="road-graph sha256 mismatch"):
            load_td_instance(instance_path, verify_sha256=True)

    def test_atf_sha_mismatch_raises(self, tmp_path):
        instance_path = write_road_instance_files(tmp_path)
        payload = json.loads(instance_path.read_text())
        payload["td"]["atf_sha256"] = "0" * 64
        save_json_to_file(payload, instance_path)
        with pytest.raises(ATFFormatError, match="materialized ATF sha256 mismatch"):
            load_td_instance(instance_path, verify_sha256=True)

    def test_skip_verification_still_loads(self, tmp_path):
        instance_path = write_road_instance_files(tmp_path)
        payload = json.loads(instance_path.read_text())
        payload["td"]["atf_sha256"] = "0" * 64
        save_json_to_file(payload, instance_path)
        loaded = load_td_instance(instance_path, verify_sha256=False)
        assert len(loaded.atfs.arcs) == 6

"""Tests for the geo (v3) and distances sidecars."""

from __future__ import annotations

import pytest

from mamut_routing_lib.distances import (
    DistancesFormatError,
    InstanceDistances,
    compute_distances_sha256,
    load_instance_distances,
    save_instance_distances,
)
from mamut_routing_lib.geo import (
    GeoFormatError,
    GeoNode,
    GeoRoadCache,
    InstanceGeo,
    compute_geo_sha256,
    load_instance_geo,
    save_instance_geo,
)


def make_geo(*, with_cache: bool = True) -> InstanceGeo:
    nodes = [
        GeoNode(instance_node_id=0, poi_lon=4.84, poi_lat=45.76, enu_x=0.0, enu_y=0.0,
                demand=0, source_tag="depot", graph_vertex_id=7),
        GeoNode(instance_node_id=1, poi_lon=4.85, poi_lat=45.76, enu_x=780.2, enu_y=-3.5,
                demand=4, source_tag="poi", graph_vertex_id=12),
        GeoNode(instance_node_id=2, poi_lon=4.84, poi_lat=45.77, enu_x=1.5, enu_y=1113.0,
                demand=7, source_tag="uniform", graph_vertex_id=31),
    ]
    cache = None
    if with_cache:
        cache = GeoRoadCache(
            vertex_lonlat=[(4.84, 45.76), (4.845, 45.762), (4.85, 45.76), (4.84, 45.77)],
            paths={
                "fastest": {
                    "0-1": [0, 1, 2], "1-0": [2, 1, 0],
                    "0-2": [0, 3], "2-0": [3, 0],
                    "1-2": [2, 1, 0, 3], "2-1": [3, 0, 1, 2],
                },
                "shortest": {
                    "0-1": [0, 2], "1-0": [2, 0],
                    "0-2": [0, 3], "2-0": [3, 0],
                    "1-2": [2, 0, 3], "2-1": [3, 0, 2],
                },
            },
        )
    return InstanceGeo(
        base_name="poryos-toy",
        benchmark_name="Poryos2026",
        city="toyville",
        method="hyb",
        source_osm_file="toyville.osm.pbf",
        reference_lla={"lat": 45.76, "lon": 4.84, "alt": 0.0},
        map_options={"only_intersections": False, "trim_to_connected_graph": True},
        nodes=nodes,
        road_cache=cache,
        generator={"name": "test-fixture"},
    )


class TestGeoSidecar:
    def test_roundtrip_plain_and_gzip(self, tmp_path):
        geo = make_geo()
        for name in ["b.geo.json", "b.geo.json.gz"]:
            save_instance_geo(geo, tmp_path / name)
            loaded = load_instance_geo(tmp_path / name)
            assert compute_geo_sha256(loaded) == compute_geo_sha256(geo)
            assert loaded.nodes == geo.nodes
            assert loaded.road_cache.paths == geo.road_cache.paths
            assert loaded.road_cache.vertex_lonlat == geo.road_cache.vertex_lonlat

    def test_sha_is_storage_form_independent(self, tmp_path):
        geo = make_geo()
        save_instance_geo(geo, tmp_path / "a.geo.json")
        save_instance_geo(geo, tmp_path / "b.geo.json.gz")
        assert compute_geo_sha256(load_instance_geo(tmp_path / "a.geo.json")) == compute_geo_sha256(
            load_instance_geo(tmp_path / "b.geo.json.gz")
        )

    def test_no_cache_roundtrip(self, tmp_path):
        geo = make_geo(with_cache=False)
        save_instance_geo(geo, tmp_path / "b.geo.json.gz")
        loaded = load_instance_geo(tmp_path / "b.geo.json.gz")
        assert loaded.road_cache is None
        assert compute_geo_sha256(loaded) == compute_geo_sha256(geo)

    def test_path_decoding(self):
        geo = make_geo()
        polyline = geo.road_cache.path_lonlat("fastest", 0, 1)
        assert polyline == [(4.84, 45.76), (4.845, 45.762), (4.85, 45.76)]
        with pytest.raises(GeoFormatError, match="no cached"):
            geo.road_cache.path_lonlat("fastest", 2, 2)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda g: setattr(g.nodes[1], "instance_node_id", 5), "sorted by instance_node_id"),
            (lambda g: setattr(g.nodes[0], "poi_lon", 999.0), "WGS84"),
            (lambda g: g.road_cache.paths["fastest"].__setitem__("0-9", [0, 1]), "out of node range"),
            (lambda g: g.road_cache.paths["fastest"].__setitem__("0-2", [0, 99]), "out of vertex range"),
            (lambda g: g.road_cache.paths["fastest"].__setitem__("0-2", [0]), "at least two points"),
            (lambda g: g.road_cache.paths.__setitem__("euclidean", {"0-1": [0, 1]}), "not in"),
            (lambda g: g.reference_lla.pop("lat"), "reference_lla"),
        ],
    )
    def test_validator_rejections(self, tmp_path, mutate, match):
        geo = make_geo()
        mutate(geo)
        with pytest.raises(GeoFormatError, match=match):
            save_instance_geo(geo, tmp_path / "bad.geo.json")

    def test_bad_suffix_rejected(self, tmp_path):
        with pytest.raises(GeoFormatError, match="must end with"):
            save_instance_geo(make_geo(), tmp_path / "b.json")

    def test_canonical_bytes_are_path_order_independent(self):
        geo_a = make_geo()
        geo_b = make_geo()
        # Same content in scrambled insertion order must hash identically.
        fastest = geo_b.road_cache.paths["fastest"]
        geo_b.road_cache.paths["fastest"] = dict(sorted(fastest.items(), reverse=True))
        assert compute_geo_sha256(geo_a) == compute_geo_sha256(geo_b)


def make_distances() -> InstanceDistances:
    return InstanceDistances(
        base_name="poryos-toy",
        benchmark_name="Poryos2026",
        metric="fastest",
        num_customers=2,
        values=[[0.0, 100.0, 200.0], [100.0, 0.0, 100.0], [200.0, 100.0, 0.0]],
        generator={"name": "test-fixture"},
    )


class TestDistancesSidecar:
    def test_roundtrip_plain_and_gzip(self, tmp_path):
        distances = make_distances()
        for name in ["b.distances-fastest.json", "b.distances-fastest.json.gz"]:
            save_instance_distances(distances, tmp_path / name)
            loaded = load_instance_distances(tmp_path / name)
            assert compute_distances_sha256(loaded) == compute_distances_sha256(distances)
            assert loaded.values == distances.values

    def test_infix_required(self, tmp_path):
        with pytest.raises(DistancesFormatError, match="distances-"):
            save_instance_distances(make_distances(), tmp_path / "b.json")

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda d: d.values.pop(), "rows"),
            (lambda d: d.values[0].pop(), "entries"),
            (lambda d: d.values[0].__setitem__(0, 1.0), "diagonal"),
            (lambda d: d.values[0].__setitem__(1, -3.0), "strictly positive"),
        ],
    )
    def test_validator_rejections(self, tmp_path, mutate, match):
        distances = make_distances()
        mutate(distances)
        with pytest.raises(DistancesFormatError, match=match):
            save_instance_distances(distances, tmp_path / "bad.distances-fastest.json")

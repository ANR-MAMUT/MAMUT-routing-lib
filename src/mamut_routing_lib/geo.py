"""Geo sidecar of a collection base instance (format ``mamut-geo`` v3).

One geo sidecar per base instance (``<base>.geo.json[.gz]``), shared by every
problem-type variant of the base. It carries everything the site/workbench
needs to draw the instance on a map: per-node geodetic + local ENU positions,
the geodetic reference frame, and (for n <= 100) a complete road-path cache
over all ordered node pairs for the ``fastest`` and ``shortest`` metrics in an
**indexed encoding**: one geo-local ``vertex_lonlat`` table (the union of all
path vertices) plus per-metric maps from ``"i-j"`` arc keys to lists of
indices into that table. Measured at n=100 this is ~5x smaller gzipped than
repeating coordinate polylines per pair. The sidecar is static: it never
changes when BKS routes change (v3 retires the on-demand ``road_cache``
growth of the v1/v2 ``meta.json``).

Purely informative for solvers: the checker never reads it. Hashed over its
uncompressed canonical JSON bytes like every other sidecar, gzip ``mtime=0``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GEO_FORMAT = "mamut-geo"
GEO_FORMAT_VERSION = 3
GEO_PLAIN_SUFFIX = ".geo.json"
GEO_GZIP_SUFFIX = ".geo.json.gz"

#: Metrics a road cache may carry paths for (euclidean previews are straight
#: lines by definition and never cached).
GEO_ROAD_CACHE_METRICS = ("fastest", "shortest")


class GeoFormatError(ValueError):
    """Raised when a geo sidecar violates the canonical format."""


@dataclass
class GeoNode:
    """One instance node on the map (depot included, ``instance_node_id`` 0-based)."""

    instance_node_id: int
    poi_lon: float
    poi_lat: float
    enu_x: float
    enu_y: float
    demand: int
    source_tag: str
    graph_vertex_id: int | None = None

    def __post_init__(self) -> None:
        self.instance_node_id = int(self.instance_node_id)
        self.poi_lon = float(self.poi_lon)
        self.poi_lat = float(self.poi_lat)
        self.enu_x = float(self.enu_x)
        self.enu_y = float(self.enu_y)
        self.demand = int(self.demand)
        self.source_tag = str(self.source_tag)
        if self.graph_vertex_id is not None:
            self.graph_vertex_id = int(self.graph_vertex_id)


@dataclass
class GeoRoadCache:
    """Indexed road-path cache: paths as index lists over a local vertex table.

    ``paths[metric]["i-j"]`` is the vertex-index polyline of the cached road
    path from instance node ``i`` to instance node ``j`` (ordered pair,
    indices into ``vertex_lonlat``, at least two points). A complete cache
    covers all ordered pairs of the instance for every metric it carries;
    completeness is a family policy (Poryos2026: n <= 100 only), not a format
    invariant.
    """

    vertex_lonlat: list[tuple[float, float]]
    paths: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertex_lonlat = [(float(lon), float(lat)) for lon, lat in self.vertex_lonlat]
        self.paths = {
            str(metric): {str(key): [int(v) for v in path] for key, path in entries.items()}
            for metric, entries in self.paths.items()
        }

    def path_lonlat(self, metric: str, from_node: int, to_node: int) -> list[tuple[float, float]]:
        """Decode one cached path back to a lon/lat polyline."""
        entries = self.paths.get(metric)
        if entries is None:
            raise GeoFormatError(f"no cached paths for metric {metric!r}")
        key = f"{from_node}-{to_node}"
        path = entries.get(key)
        if path is None:
            raise GeoFormatError(f"no cached {metric} path for arc {key}")
        return [self.vertex_lonlat[v] for v in path]


@dataclass
class InstanceGeo:
    """In-memory content of a geo sidecar."""

    base_name: str
    benchmark_name: str
    city: str
    method: str
    source_osm_file: str
    reference_lla: dict[str, float]
    map_options: dict[str, Any]
    nodes: list[GeoNode]
    road_cache: GeoRoadCache | None = None
    generator: dict[str, Any] = field(default_factory=dict)
    format_version: int = GEO_FORMAT_VERSION

    def num_customers(self) -> int:
        return len(self.nodes) - 1


def _validate_geo(geo: InstanceGeo) -> None:
    if not geo.base_name:
        raise GeoFormatError("base_name must be non-empty")
    if len(geo.nodes) < 2:
        raise GeoFormatError("nodes must contain the depot and at least one customer")
    for expected, node in enumerate(geo.nodes):
        if node.instance_node_id != expected:
            raise GeoFormatError(
                f"nodes must be sorted by instance_node_id starting at 0; "
                f"found {node.instance_node_id} at position {expected}"
            )
        if not (-180.0 <= node.poi_lon <= 180.0 and -90.0 <= node.poi_lat <= 90.0):
            raise GeoFormatError(
                f"node {expected} poi position ({node.poi_lon}, {node.poi_lat}) out of WGS84 range"
            )
    for key in ("lat", "lon"):
        if key not in geo.reference_lla:
            raise GeoFormatError(f"reference_lla must carry {key!r}")
    if geo.road_cache is not None:
        cache = geo.road_cache
        if not cache.vertex_lonlat:
            raise GeoFormatError("road_cache.vertex_lonlat must be non-empty")
        for v, (lon, lat) in enumerate(cache.vertex_lonlat):
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                raise GeoFormatError(f"road_cache.vertex_lonlat[{v}] = ({lon}, {lat}) out of WGS84 range")
        num_nodes = len(geo.nodes)
        num_vertices = len(cache.vertex_lonlat)
        for metric, entries in cache.paths.items():
            if metric not in GEO_ROAD_CACHE_METRICS:
                raise GeoFormatError(
                    f"road_cache metric {metric!r} not in {GEO_ROAD_CACHE_METRICS}"
                )
            for key, path in entries.items():
                parts = key.split("-")
                if len(parts) != 2:
                    raise GeoFormatError(f"road_cache path key {key!r} must be 'i-j'")
                i, j = int(parts[0]), int(parts[1])
                if not (0 <= i < num_nodes and 0 <= j < num_nodes) or i == j:
                    raise GeoFormatError(f"road_cache path key {key!r} out of node range")
                if len(path) < 2:
                    raise GeoFormatError(f"road_cache path {key!r} must have at least two points")
                for v in path:
                    if not 0 <= v < num_vertices:
                        raise GeoFormatError(
                            f"road_cache path {key!r} index {v} out of vertex range 0..{num_vertices - 1}"
                        )


def _path_key_order(key: str) -> tuple[int, int]:
    i, j = key.split("-")
    return int(i), int(j)


def geo_to_canonical_json_bytes(geo: InstanceGeo) -> bytes:
    """Serialize to the canonical JSON bytes (the input of the geo sha256).

    Fixed key order, one node / one cache vertex / one cached path per line,
    floats via Python's shortest round-trip repr, gzip-independent. Path maps
    are ordered by ``(i, j)``.
    """
    header_lines = [
        "{",
        f'    "format": {json.dumps(GEO_FORMAT)},',
        f'    "format_version": {json.dumps(geo.format_version)},',
        f'    "base_name": {json.dumps(geo.base_name)},',
        f'    "benchmark_name": {json.dumps(geo.benchmark_name)},',
        f'    "city": {json.dumps(geo.city)},',
        f'    "method": {json.dumps(geo.method)},',
        f'    "source_osm_file": {json.dumps(geo.source_osm_file)},',
        f'    "reference_lla": {json.dumps(geo.reference_lla, sort_keys=True)},',
        f'    "map_options": {json.dumps(geo.map_options, sort_keys=True)},',
        f'    "generator": {json.dumps(geo.generator, sort_keys=True)},',
        '    "nodes": [',
    ]
    node_lines = ",\n".join(
        "        "
        + json.dumps(
            {
                "instance_node_id": node.instance_node_id,
                "poi_lon": node.poi_lon,
                "poi_lat": node.poi_lat,
                "enu_x": node.enu_x,
                "enu_y": node.enu_y,
                "demand": node.demand,
                "source_tag": node.source_tag,
                "graph_vertex_id": node.graph_vertex_id,
            }
        )
        for node in geo.nodes
    )
    prefix = "\n".join(header_lines) + "\n" + node_lines + "\n"
    if geo.road_cache is None:
        return (prefix + '    ],\n    "road_cache": null\n}\n').encode("utf-8")
    cache = geo.road_cache
    cache_lines = ['    ],', '    "road_cache": {', '        "vertex_lonlat": [']
    cache_lines.append(
        ",\n".join("            " + json.dumps([lon, lat]) for lon, lat in cache.vertex_lonlat)
    )
    cache_lines.append('        ],')
    cache_lines.append('        "paths": {')
    metric_blocks = []
    for metric in sorted(cache.paths):
        entries = cache.paths[metric]
        entry_lines = ",\n".join(
            f"                {json.dumps(key)}: {json.dumps(entries[key])}"
            for key in sorted(entries, key=_path_key_order)
        )
        metric_blocks.append(
            f"            {json.dumps(metric)}: {{\n" + entry_lines + "\n            }"
        )
    cache_lines.append(",\n".join(metric_blocks))
    cache_lines.append("        }")
    cache_lines.append("    }")
    return (prefix + "\n".join(cache_lines) + "\n}\n").encode("utf-8")


def compute_geo_sha256(geo: InstanceGeo) -> str:
    return hashlib.sha256(geo_to_canonical_json_bytes(geo)).hexdigest()


def save_instance_geo(geo: InstanceGeo, path: str | Path) -> None:
    """Write the sidecar; gzip iff the path ends with ``.geo.json.gz`` (``mtime=0``)."""
    _validate_geo(geo)
    target = Path(path)
    data = geo_to_canonical_json_bytes(geo)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name.endswith(GEO_GZIP_SUFFIX):
        target.write_bytes(gzip.compress(data, mtime=0))
    elif target.name.endswith(GEO_PLAIN_SUFFIX):
        target.write_bytes(data)
    else:
        raise GeoFormatError(f"geo path must end with {GEO_PLAIN_SUFFIX} or {GEO_GZIP_SUFFIX}: {target.name}")


def load_instance_geo(path: str | Path) -> InstanceGeo:
    source = Path(path)
    raw = source.read_bytes()
    if source.name.endswith(GEO_GZIP_SUFFIX):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != GEO_FORMAT:
        raise GeoFormatError(f"unexpected format marker: {payload.get('format')!r}")
    if payload.get("format_version") != GEO_FORMAT_VERSION:
        raise GeoFormatError(f"unsupported format_version: {payload.get('format_version')!r}")
    cache_payload = payload.get("road_cache")
    road_cache = None
    if cache_payload is not None:
        road_cache = GeoRoadCache(
            vertex_lonlat=[(p[0], p[1]) for p in cache_payload["vertex_lonlat"]],
            paths={
                metric: dict(entries)
                for metric, entries in cache_payload.get("paths", {}).items()
            },
        )
    geo = InstanceGeo(
        base_name=str(payload["base_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        city=str(payload["city"]),
        method=str(payload["method"]),
        source_osm_file=str(payload["source_osm_file"]),
        reference_lla=dict(payload["reference_lla"]),
        map_options=dict(payload["map_options"]),
        nodes=[GeoNode(**node) for node in payload["nodes"]],
        road_cache=road_cache,
        generator=dict(payload.get("generator", {})),
    )
    _validate_geo(geo)
    return geo

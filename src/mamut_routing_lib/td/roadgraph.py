"""Road-network engine for the ``road-graph`` td model (format v2).

Format v2 splits the v1 monolithic sidecar in two, mirroring the base /
subinstance structure of the Mamut2026 family:

1. The **road-graph sidecar** (``<base>.road.json[.gz]``, format
   ``mamut-road-graph`` v2), one per base instance: the trimmed road subgraph
   every traffic subinstance lives on — directed edges with a physical length
   and a static free-flow ``speed_limit`` (the pinned Dijkstra weight is
   ``length / speed_limit``, so the trim and the pinned paths are properties
   of the base, independent of traffic), ``vertex_lonlat`` WGS84 coordinates
   (route polylines and traffic heatmaps are derivable from published data
   alone), and the instance-node -> graph-vertex mapping.

2. The **traffic overlay sidecar**
   (``<base>.traffic-<model>-<intensity>.json[.gz]``, format
   ``mamut-traffic-overlay``), one per subinstance: per-edge piecewise-
   constant speeds over the horizon bins, rows aligned with the road
   sidecar's edge order, strictly positive (FIFO by construction) and clamped
   at the edge's free-flow limit (overlays are slowdowns by contract).

Both are hashed over their uncompressed canonical JSON bytes like every other
TD sidecar. The pinned deterministic materialization is unchanged from the
M12.1 sampling spec:

- a tie-break-pinned Dijkstra over static free-flow times
  (``compute_fastest_path_tree``) fixing one canonical fastest path per
  ordered vertex pair, shared by all subinstances of the base;
- exact per-edge arrival functions from the overlay speeds (``build_arc_atf``
  reused from the IGP engine with the bins as speed zones, extended to
  ``extension_end``);
- **exact grid sampling** along the fastest-path tree: a fixed departure grid
  (``sample_step`` spacing over the horizon) is propagated down the tree by
  evaluating each edge ATF at the parent's arrival values, so every sample is
  the *exact* arrival of the exact edge-by-edge composition — pointwise error
  cannot accumulate along the path;
- deterministic decimation of the sampled arc function: one simultaneous drop
  of interior points exactly collinear with their original neighbours, then
  Douglas-Peucker simplification with ``simplify_tolerance``
  (``simplify_ndcpwlf`` spec). Kept points are a subset of the exact samples,
  so the result is non-decreasing (FIFO) and ``ys >= xs`` structurally, and
  it spans exactly the horizon.

Every step is plain IEEE-754 double arithmetic on the stored floats with no
epsilon comparisons, so the result is a pure function of the two sidecars'
content. An optional numpy fast path evaluates the same formulas elementwise
and is required to be bit-identical to the pure-Python reference
(gate-tested); materialization therefore does not depend on whether numpy is
installed.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mamut_routing_lib.td.igp import _gc_paused, build_arc_atf
from mamut_routing_lib.td.models import (
    ROAD_GRAPH_FORMAT,
    ROAD_GRAPH_FORMAT_VERSION,
    TRAFFIC_OVERLAY_FORMAT,
    TRAFFIC_OVERLAY_FORMAT_VERSION,
    AnyTDBenchmarkInstance,
    TDRoadGraphRef,
)
from mamut_routing_lib.td.pwlf import NDCPWLF

try:  # optional fast path; results are gate-tested bit-identical
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is a de-facto dependency
    _np = None

if TYPE_CHECKING:
    from mamut_routing_lib.td.artifacts import InstanceATFs

ROAD_PLAIN_SUFFIX = ".road.json"
ROAD_GZIP_SUFFIX = ".road.json.gz"
TRAFFIC_PLAIN_SUFFIX = ".json"
TRAFFIC_GZIP_SUFFIX = ".json.gz"
TRAFFIC_INFIX = ".traffic-"

#: Fixed header constant of materialized sidecars. Materialization is defined
#: by the road-graph + traffic sidecars plus the TD benchmark standard, not by
#: the tool that generated the instance (tool provenance lives in the instance
#: ``metadata`` and the sidecar ``generator``) -- this is what makes
#: ``atf_sha256`` reproducible from the published data alone.
ROAD_MATERIALIZER_GENERATOR: dict[str, Any] = {"name": "road-graph-materializer", "version": 2}


class RoadGraphFormatError(ValueError):
    """Raised when a road-graph or traffic-overlay sidecar violates the canonical format."""


@dataclass
class InstanceRoadGraph:
    """In-memory content of a road-graph sidecar (format v2).

    Vertices are ``0 .. num_vertices - 1``; ``vertex_osm_ids[v]`` records the
    originating OSM node id of vertex ``v`` (strictly increasing — vertices
    are numbered in ascending OSM-id order, informative only);
    ``vertex_lonlat[v]`` is its WGS84 ``(lon, lat)`` position (informative
    only: rendering and heatmaps, never read by the materialization).
    ``edges`` holds ``[u, v, length_m, speed_limit]`` entries sorted strictly
    increasing by ``(u, v)`` (directed, no parallel edges); ``speed_limit`` is
    the static free-flow limit (m-per-time-unit in the instance's units,
    strictly positive) defining the pinned Dijkstra weight
    ``length / speed_limit``. ``node_vertices[i]`` is the graph vertex of
    instance node ``i`` (depot first, all distinct). ``sample_step`` is the
    departure-grid spacing and ``simplify_tolerance`` the decimation
    tolerance of the pinned materialization; both are canonical (hashed)
    materialization parameters, repeated in the instance td block and
    cross-checked at load.
    """

    base_name: str
    benchmark_name: str
    num_customers: int
    horizon: tuple[float, float]
    extension_end: float
    bin_edges: list[float]
    sample_step: float
    simplify_tolerance: float
    num_vertices: int
    vertex_osm_ids: list[int]
    vertex_lonlat: list[tuple[float, float]]
    node_vertices: list[int]
    edges: list[tuple[int, int, float, float]]
    generator: dict[str, Any] = field(default_factory=dict)
    format_version: int = ROAD_GRAPH_FORMAT_VERSION

    def __post_init__(self) -> None:
        # Coerce numerics so canonical serialization never leaks python ints
        # into float positions (``86400`` vs ``86400.0`` would break the
        # write/reload sha256 round-trip).
        self.horizon = (float(self.horizon[0]), float(self.horizon[1]))
        self.extension_end = float(self.extension_end)
        self.bin_edges = [float(b) for b in self.bin_edges]
        self.sample_step = float(self.sample_step)
        self.simplify_tolerance = float(self.simplify_tolerance)
        self.num_customers = int(self.num_customers)
        self.num_vertices = int(self.num_vertices)
        self.vertex_osm_ids = [int(v) for v in self.vertex_osm_ids]
        self.vertex_lonlat = [(float(lon), float(lat)) for lon, lat in self.vertex_lonlat]
        self.node_vertices = [int(v) for v in self.node_vertices]
        self.edges = [
            (int(u), int(v), float(length), float(speed_limit))
            for u, v, length, speed_limit in self.edges
        ]


@dataclass
class TrafficOverlay:
    """In-memory content of a traffic-overlay sidecar.

    ``edge_speeds[k]`` is the per-bin speed row of edge ``k`` of the base's
    road-graph sidecar (same order, same count); every speed is strictly
    positive (FIFO by construction) and at most the edge's static
    ``speed_limit`` (validated against the road graph: overlays are slowdowns
    by contract). ``bin_edges`` must equal the road sidecar's bit-exactly.
    """

    base_name: str
    benchmark_name: str
    traffic_model: str
    intensity: str
    bin_edges: list[float]
    edge_speeds: list[list[float]]
    generator: dict[str, Any] = field(default_factory=dict)
    format_version: int = TRAFFIC_OVERLAY_FORMAT_VERSION

    def __post_init__(self) -> None:
        self.bin_edges = [float(b) for b in self.bin_edges]
        self.edge_speeds = [[float(s) for s in row] for row in self.edge_speeds]

    def num_edges(self) -> int:
        return len(self.edge_speeds)


def departure_grid(horizon: tuple[float, float], sample_step: float) -> list[float]:
    """The pinned departure grid: ``start + k * sample_step`` up to the horizon end.

    ``sample_step`` must tile the horizon exactly (the computed endpoint must
    equal the horizon end bit-for-bit), so the grid spans the horizon and the
    materialized arc functions inherit exact horizon endpoints.
    """
    start, end = horizon
    if sample_step <= 0:
        raise RoadGraphFormatError(f"sample_step must be strictly positive, got {sample_step}")
    count = round((end - start) / sample_step)
    if count < 1 or start + count * sample_step != end:
        raise RoadGraphFormatError(
            f"sample_step {sample_step} must tile the horizon [{start}, {end}] exactly"
        )
    return [start + k * sample_step for k in range(count + 1)]


def _validate_road_graph(road: InstanceRoadGraph) -> None:
    if road.horizon[0] >= road.horizon[1]:
        raise RoadGraphFormatError(f"horizon must be a non-empty interval, got {road.horizon}")
    if road.extension_end <= road.horizon[1]:
        raise RoadGraphFormatError(
            f"extension_end {road.extension_end} must lie strictly past the horizon end {road.horizon[1]}"
        )
    if road.simplify_tolerance < 0:
        raise RoadGraphFormatError(f"simplify_tolerance must be >= 0, got {road.simplify_tolerance}")
    departure_grid(road.horizon, road.sample_step)  # validates sample_step
    if len(road.bin_edges) < 2:
        raise RoadGraphFormatError("bin_edges must define at least one bin")
    if road.bin_edges[0] != road.horizon[0] or road.bin_edges[-1] != road.horizon[1]:
        raise RoadGraphFormatError(
            f"bin_edges [{road.bin_edges[0]}, {road.bin_edges[-1]}] must span exactly the horizon "
            f"[{road.horizon[0]}, {road.horizon[1]}]"
        )
    for k in range(1, len(road.bin_edges)):
        if road.bin_edges[k] <= road.bin_edges[k - 1]:
            raise RoadGraphFormatError(f"bin_edges must be strictly increasing (violated at index {k})")
    if road.num_customers <= 0:
        raise RoadGraphFormatError(f"num_customers must be positive, got {road.num_customers}")
    if road.num_vertices <= 0:
        raise RoadGraphFormatError(f"num_vertices must be positive, got {road.num_vertices}")
    if len(road.vertex_osm_ids) != road.num_vertices:
        raise RoadGraphFormatError(
            f"expected {road.num_vertices} vertex_osm_ids, found {len(road.vertex_osm_ids)}"
        )
    for v in range(1, road.num_vertices):
        if road.vertex_osm_ids[v] <= road.vertex_osm_ids[v - 1]:
            raise RoadGraphFormatError(
                "vertex_osm_ids must be strictly increasing (vertices are numbered in ascending OSM-id order)"
            )
    if len(road.vertex_lonlat) != road.num_vertices:
        raise RoadGraphFormatError(
            f"expected {road.num_vertices} vertex_lonlat entries, found {len(road.vertex_lonlat)}"
        )
    for v, (lon, lat) in enumerate(road.vertex_lonlat):
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise RoadGraphFormatError(f"vertex_lonlat[{v}] = ({lon}, {lat}) out of WGS84 range")
    expected_nodes = road.num_customers + 1
    if len(road.node_vertices) != expected_nodes:
        raise RoadGraphFormatError(
            f"expected {expected_nodes} node_vertices (num_customers + 1), found {len(road.node_vertices)}"
        )
    seen_nodes: set[int] = set()
    for i, vertex in enumerate(road.node_vertices):
        if not 0 <= vertex < road.num_vertices:
            raise RoadGraphFormatError(f"node_vertices[{i}] = {vertex} out of range 0..{road.num_vertices - 1}")
        if vertex in seen_nodes:
            raise RoadGraphFormatError(f"node_vertices must be distinct; vertex {vertex} appears twice")
        seen_nodes.add(vertex)
    if not road.edges:
        raise RoadGraphFormatError("edges must be non-empty")
    previous: tuple[int, int] | None = None
    for index, (u, v, length, speed_limit) in enumerate(road.edges):
        if not (0 <= u < road.num_vertices and 0 <= v < road.num_vertices):
            raise RoadGraphFormatError(f"edge {index} endpoints ({u}, {v}) out of range")
        if u == v:
            raise RoadGraphFormatError(f"edge {index} is a self-loop at vertex {u}")
        key = (u, v)
        if previous is not None and key <= previous:
            raise RoadGraphFormatError(
                f"edges must be sorted strictly increasing by (u, v); ({u}, {v}) after {previous}"
            )
        previous = key
        if length <= 0:
            raise RoadGraphFormatError(f"edge {index} ({u}, {v}) length {length} must be strictly positive")
        if speed_limit <= 0:
            raise RoadGraphFormatError(
                f"edge {index} ({u}, {v}) speed_limit {speed_limit} must be strictly positive"
            )


def _validate_traffic_overlay(overlay: TrafficOverlay) -> None:
    if len(overlay.bin_edges) < 2:
        raise RoadGraphFormatError("bin_edges must define at least one bin")
    for k in range(1, len(overlay.bin_edges)):
        if overlay.bin_edges[k] <= overlay.bin_edges[k - 1]:
            raise RoadGraphFormatError(f"bin_edges must be strictly increasing (violated at index {k})")
    if not overlay.edge_speeds:
        raise RoadGraphFormatError("edge_speeds must be non-empty")
    num_bins = len(overlay.bin_edges) - 1
    for index, row in enumerate(overlay.edge_speeds):
        if len(row) != num_bins:
            raise RoadGraphFormatError(
                f"edge_speeds[{index}] has {len(row)} entries, expected one per bin ({num_bins})"
            )
        for b, speed in enumerate(row):
            if speed <= 0:
                raise RoadGraphFormatError(
                    f"edge_speeds[{index}][{b}] = {speed} must be strictly positive (FIFO)"
                )
    if not overlay.traffic_model:
        raise RoadGraphFormatError("traffic_model must be non-empty")
    if not overlay.intensity:
        raise RoadGraphFormatError("intensity must be non-empty")


def validate_overlay_against_road(overlay: TrafficOverlay, road: InstanceRoadGraph) -> None:
    """The alignment contract between a traffic overlay and its base road graph."""
    if overlay.base_name != road.base_name:
        raise RoadGraphFormatError(
            f"overlay base_name {overlay.base_name!r} does not match road graph {road.base_name!r}"
        )
    if overlay.benchmark_name != road.benchmark_name:
        raise RoadGraphFormatError(
            f"overlay benchmark_name {overlay.benchmark_name!r} does not match "
            f"road graph {road.benchmark_name!r}"
        )
    if overlay.bin_edges != road.bin_edges:
        raise RoadGraphFormatError("overlay bin_edges do not equal the road graph's bit-exactly")
    if len(overlay.edge_speeds) != len(road.edges):
        raise RoadGraphFormatError(
            f"overlay has {len(overlay.edge_speeds)} edge rows, road graph has {len(road.edges)} edges"
        )
    for index, ((u, v, _, speed_limit), row) in enumerate(zip(road.edges, overlay.edge_speeds)):
        for b, speed in enumerate(row):
            if speed > speed_limit:
                raise RoadGraphFormatError(
                    f"edge {index} ({u}, {v}) speed[{b}] = {speed} exceeds the free-flow "
                    f"speed_limit {speed_limit} (overlays are slowdowns by contract)"
                )


def road_graph_to_canonical_json_bytes(road: InstanceRoadGraph) -> bytes:
    """Serialize to the canonical JSON bytes (the input of the graph sha256).

    Fixed key order, one edge / one vertex coordinate pair per line, floats
    via Python's shortest round-trip repr, gzip-independent.
    """
    header_lines = [
        "{",
        f'    "format": {json.dumps(ROAD_GRAPH_FORMAT)},',
        f'    "format_version": {json.dumps(road.format_version)},',
        f'    "base_name": {json.dumps(road.base_name)},',
        f'    "benchmark_name": {json.dumps(road.benchmark_name)},',
        f'    "num_customers": {json.dumps(road.num_customers)},',
        f'    "horizon": {json.dumps(list(road.horizon))},',
        f'    "extension_end": {json.dumps(road.extension_end)},',
        f'    "bin_edges": {json.dumps(road.bin_edges)},',
        f'    "sample_step": {json.dumps(road.sample_step)},',
        f'    "simplify_tolerance": {json.dumps(road.simplify_tolerance)},',
        f'    "num_vertices": {json.dumps(road.num_vertices)},',
        f'    "vertex_osm_ids": {json.dumps(road.vertex_osm_ids)},',
        f'    "node_vertices": {json.dumps(road.node_vertices)},',
        f'    "generator": {json.dumps(road.generator, sort_keys=True)},',
        '    "vertex_lonlat": [',
    ]
    lonlat_body = ",\n".join(
        "        " + json.dumps([lon, lat]) for lon, lat in road.vertex_lonlat
    )
    edge_header = '    ],\n    "edges": ['
    edges_body = ",\n".join(
        "        " + json.dumps([u, v, length, speed_limit])
        for u, v, length, speed_limit in road.edges
    )
    text = (
        "\n".join(header_lines)
        + "\n" + lonlat_body + "\n" + edge_header + "\n" + edges_body + "\n    ]\n}\n"
    )
    return text.encode("utf-8")


def traffic_overlay_to_canonical_json_bytes(overlay: TrafficOverlay) -> bytes:
    """Serialize to the canonical JSON bytes (the input of the traffic sha256)."""
    header_lines = [
        "{",
        f'    "format": {json.dumps(TRAFFIC_OVERLAY_FORMAT)},',
        f'    "format_version": {json.dumps(overlay.format_version)},',
        f'    "base_name": {json.dumps(overlay.base_name)},',
        f'    "benchmark_name": {json.dumps(overlay.benchmark_name)},',
        f'    "traffic_model": {json.dumps(overlay.traffic_model)},',
        f'    "intensity": {json.dumps(overlay.intensity)},',
        f'    "bin_edges": {json.dumps(overlay.bin_edges)},',
        f'    "generator": {json.dumps(overlay.generator, sort_keys=True)},',
        '    "edge_speeds": [',
    ]
    body = ",\n".join("        " + json.dumps(row) for row in overlay.edge_speeds)
    text = "\n".join(header_lines) + "\n" + body + "\n    ]\n}\n"
    return text.encode("utf-8")


def compute_road_graph_sha256(road: InstanceRoadGraph) -> str:
    return hashlib.sha256(road_graph_to_canonical_json_bytes(road)).hexdigest()


def compute_traffic_overlay_sha256(overlay: TrafficOverlay) -> str:
    return hashlib.sha256(traffic_overlay_to_canonical_json_bytes(overlay)).hexdigest()


def save_instance_road_graph(road: InstanceRoadGraph, path: str | Path) -> None:
    """Write the sidecar; gzip iff the path ends with ``.road.json.gz`` (``mtime=0``)."""
    _validate_road_graph(road)
    target = Path(path)
    data = road_graph_to_canonical_json_bytes(road)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name.endswith(ROAD_GZIP_SUFFIX):
        target.write_bytes(gzip.compress(data, mtime=0))
    elif target.name.endswith(ROAD_PLAIN_SUFFIX):
        target.write_bytes(data)
    else:
        raise RoadGraphFormatError(
            f"road-graph path must end with {ROAD_PLAIN_SUFFIX} or {ROAD_GZIP_SUFFIX}: {target.name}"
        )


def load_instance_road_graph(path: str | Path) -> InstanceRoadGraph:
    source = Path(path)
    raw = source.read_bytes()
    if source.name.endswith(ROAD_GZIP_SUFFIX):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != ROAD_GRAPH_FORMAT:
        raise RoadGraphFormatError(f"unexpected format marker: {payload.get('format')!r}")
    if payload.get("format_version") != ROAD_GRAPH_FORMAT_VERSION:
        raise RoadGraphFormatError(f"unsupported format_version: {payload.get('format_version')!r}")
    road = InstanceRoadGraph(
        base_name=str(payload["base_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        num_customers=int(payload["num_customers"]),
        horizon=(payload["horizon"][0], payload["horizon"][1]),
        extension_end=payload["extension_end"],
        bin_edges=list(payload["bin_edges"]),
        sample_step=payload["sample_step"],
        simplify_tolerance=payload["simplify_tolerance"],
        num_vertices=int(payload["num_vertices"]),
        vertex_osm_ids=list(payload["vertex_osm_ids"]),
        vertex_lonlat=[(p[0], p[1]) for p in payload["vertex_lonlat"]],
        node_vertices=list(payload["node_vertices"]),
        edges=[(e[0], e[1], e[2], e[3]) for e in payload["edges"]],
        generator=dict(payload.get("generator", {})),
    )
    _validate_road_graph(road)
    return road


def save_traffic_overlay(overlay: TrafficOverlay, path: str | Path) -> None:
    """Write the overlay; gzip iff the path ends with ``.json.gz`` (``mtime=0``).

    The conventional name is ``<base>.traffic-<model>-<intensity>.json[.gz]``;
    the ``.traffic-`` infix is required so overlays are never confused with
    other sidecars.
    """
    _validate_traffic_overlay(overlay)
    target = Path(path)
    if TRAFFIC_INFIX not in target.name:
        raise RoadGraphFormatError(f"traffic-overlay path must contain {TRAFFIC_INFIX!r}: {target.name}")
    data = traffic_overlay_to_canonical_json_bytes(overlay)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name.endswith(TRAFFIC_GZIP_SUFFIX):
        target.write_bytes(gzip.compress(data, mtime=0))
    elif target.name.endswith(TRAFFIC_PLAIN_SUFFIX):
        target.write_bytes(data)
    else:
        raise RoadGraphFormatError(
            f"traffic-overlay path must end with {TRAFFIC_PLAIN_SUFFIX} or {TRAFFIC_GZIP_SUFFIX}: {target.name}"
        )


def load_traffic_overlay(path: str | Path) -> TrafficOverlay:
    source = Path(path)
    raw = source.read_bytes()
    if source.name.endswith(TRAFFIC_GZIP_SUFFIX):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != TRAFFIC_OVERLAY_FORMAT:
        raise RoadGraphFormatError(f"unexpected format marker: {payload.get('format')!r}")
    if payload.get("format_version") != TRAFFIC_OVERLAY_FORMAT_VERSION:
        raise RoadGraphFormatError(f"unsupported format_version: {payload.get('format_version')!r}")
    overlay = TrafficOverlay(
        base_name=str(payload["base_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        traffic_model=str(payload["traffic_model"]),
        intensity=str(payload["intensity"]),
        bin_edges=list(payload["bin_edges"]),
        edge_speeds=[list(row) for row in payload["edge_speeds"]],
        generator=dict(payload.get("generator", {})),
    )
    _validate_traffic_overlay(overlay)
    return overlay


def _drop_exactly_collinear(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """One simultaneous pass dropping interior points exactly collinear with
    their original neighbours (cross-multiplied slope equality on the stored
    floats — a deterministic exact-float test, not an epsilon)."""
    n = len(xs)
    if n <= 2:
        return xs, ys
    kept_xs = [xs[0]]
    kept_ys = [ys[0]]
    for k in range(1, n - 1):
        left_dx = xs[k] - xs[k - 1]
        right_dx = xs[k + 1] - xs[k]
        if (ys[k] - ys[k - 1]) * right_dx == (ys[k + 1] - ys[k]) * left_dx:
            continue
        kept_xs.append(xs[k])
        kept_ys.append(ys[k])
    kept_xs.append(xs[n - 1])
    kept_ys.append(ys[n - 1])
    return kept_xs, kept_ys


def _simplify_points(xs: list[float], ys: list[float], tolerance: float) -> tuple[list[float], list[float]]:
    """Deterministic Douglas-Peucker on breakpoint lists (see ``simplify_ndcpwlf``)."""
    n = len(xs)
    if n <= 2:
        return xs, ys
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        x_lo, x_hi = xs[lo], xs[hi]
        if x_lo == x_hi:
            # A pure vertical run: chord interpolation is undefined, keep it whole.
            for k in range(lo + 1, hi):
                keep[k] = True
            continue
        y_lo, y_hi = ys[lo], ys[hi]
        inv_span = 1.0 / (x_hi - x_lo)
        best_dev = -1.0
        best_k = -1
        for k in range(lo + 1, hi):
            t = (xs[k] - x_lo) * inv_span
            deviation = abs(ys[k] - (y_lo + t * (y_hi - y_lo)))
            if deviation > best_dev:
                best_dev = deviation
                best_k = k
        if best_dev > tolerance:
            keep[best_k] = True
            stack.append((lo, best_k))
            stack.append((best_k, hi))
    if all(keep):
        return xs, ys
    kept_xs = [xs[k] for k in range(n) if keep[k]]
    kept_ys = [ys[k] for k in range(n) if keep[k]]
    return kept_xs, kept_ys


def simplify_ndcpwlf(f: NDCPWLF, tolerance: float) -> NDCPWLF:
    """Deterministic Douglas-Peucker simplification of an NDCPWLF.

    Keeps a subset of the original breakpoints (endpoints always kept), so
    the result is non-decreasing and continuous by construction and every
    kept breakpoint keeps ``ys >= xs``. Interior breakpoints are dropped when
    their vertical deviation from the chord of the enclosing kept pair is
    ``<= tolerance``; the split point is the deviation argmax, ties broken by
    the smallest index. A vertical step strictly inside a kept pair may be
    smoothed like any other breakpoint (its deviation is bounded the same
    way); only when the kept pair itself is a pure vertical run (equal
    ``x``) is chord interpolation undefined and the run kept whole.
    """
    if tolerance < 0:
        raise RoadGraphFormatError(f"tolerance must be >= 0, got {tolerance}")
    if f.num_breakpoints() <= 2:
        return f
    xs, ys = _simplify_points(f.xs, f.ys, tolerance)
    if xs is f.xs:
        return f
    return NDCPWLF(xs, ys, validate=False)


def compute_fastest_path_tree(
    road: InstanceRoadGraph,
    adjacency: list[list[int]],
    source_vertex: int,
) -> tuple[list[float], list[int]]:
    """Pinned Dijkstra over static free-flow times from ``source_vertex``.

    Free-flow edge weight is ``length / speed_limit``: a property of the
    base's road graph, independent of any traffic overlay. Determinism is
    pinned end to end: the heap orders ``(distance, vertex)`` tuples (total
    order, so equal-distance pops resolve by vertex id), outgoing edges relax
    in edge-list order, and labels update on strict ``<`` only — the first
    fastest path found is canonical. Returns ``(dist, pred_edge)`` where
    ``pred_edge[v]`` is the index into ``road.edges`` of the tree edge
    entering ``v`` (``-1`` at the source and at unreachable vertices).
    """
    edges = road.edges
    dist = [float("inf")] * road.num_vertices
    pred_edge = [-1] * road.num_vertices
    dist[source_vertex] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source_vertex)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for edge_index in adjacency[u]:
            _, v, length, speed_limit = edges[edge_index]
            candidate = d + length / speed_limit
            if candidate < dist[v]:
                dist[v] = candidate
                pred_edge[v] = edge_index
                heapq.heappush(heap, (candidate, v))
    return dist, pred_edge


def build_adjacency(road: InstanceRoadGraph) -> list[list[int]]:
    """Outgoing edge indices per vertex, in edge-list (sorted (u, v)) order."""
    adjacency: list[list[int]] = [[] for _ in range(road.num_vertices)]
    for index, (u, _, _, _) in enumerate(road.edges):
        adjacency[u].append(index)
    return adjacency


def free_flow_node_times(road: InstanceRoadGraph) -> list[list[float]]:
    """Free-flow fastest travel times between all instance nodes.

    One pinned Dijkstra per node; entry ``[i][j]`` is the free-flow time from
    node ``i`` to node ``j`` (0.0 on the diagonal). This is the reference the
    published ``distances-fastest`` sidecar must match (after the family's
    rounding convention); this is the generation gate of the Mamut2026 collection.
    """
    adjacency = build_adjacency(road)
    matrix: list[list[float]] = []
    for source_node in range(road.num_customers + 1):
        dist, _ = compute_fastest_path_tree(road, adjacency, road.node_vertices[source_node])
        row = [dist[road.node_vertices[target]] for target in range(road.num_customers + 1)]
        for target, value in enumerate(row):
            if value == float("inf"):
                raise RoadGraphFormatError(
                    f"node {target} unreachable from node {source_node} at free flow"
                )
        matrix.append(row)
    return matrix


class _EdgeEvaluator:
    """Evaluates one edge ATF at many points, reference or numpy fast path.

    Both paths compute, per query ``x``: locate ``i = bisect_left(xs, x)``;
    return ``ys[i]`` when ``xs[i] == x``, else interpolate
    ``y_lo + t * (y_hi - y_lo)`` with ``t = (x - x_lo) / (x_hi - x_lo)``.
    The numpy path applies the identical formula elementwise on float64, so
    the two are bit-identical (gate-tested in the suite).
    """

    __slots__ = ("atf", "np_xs", "np_ys")

    def __init__(self, atf: NDCPWLF) -> None:
        self.atf = atf
        if _np is not None:
            self.np_xs = _np.asarray(atf.xs, dtype=_np.float64)
            self.np_ys = _np.asarray(atf.ys, dtype=_np.float64)

    def evaluate_many(self, values):
        if _np is not None:
            queries = _np.asarray(values, dtype=_np.float64)
            idx = _np.searchsorted(self.np_xs, queries, side="left")
            x_at = self.np_xs[idx]
            exact = x_at == queries
            safe_prev = _np.maximum(idx, 1) - 1
            x_lo = self.np_xs[safe_prev]
            y_lo = self.np_ys[safe_prev]
            x_hi = x_at
            y_hi = self.np_ys[idx]
            span = x_hi - x_lo
            span = _np.where(span == 0.0, 1.0, span)  # masked by `exact` below
            t = (queries - x_lo) / span
            interpolated = y_lo + t * (y_hi - y_lo)
            return _np.where(exact, self.np_ys[idx], interpolated)
        atf = self.atf
        return [atf.evaluate(v) for v in values]


def materialize_instance_atfs_roadgraph(
    instance: AnyTDBenchmarkInstance,
    road: InstanceRoadGraph,
    overlay: TrafficOverlay,
) -> "InstanceATFs":
    """Build the canonical complete-graph ``InstanceATFs`` of a road-graph instance.

    Pinned fastest paths come from the road graph's static free-flow limits
    (shared by every subinstance of the base); per-edge arrival functions come
    from the overlay's bin speeds. One pinned Dijkstra per instance node, then
    per-arc ATFs by exact grid sampling: the departure grid is propagated down
    the fastest-path tree (arrival vectors memoized per tree vertex), each
    target's samples are decimated (exact-collinear drop, then Douglas-Peucker
    with the pinned tolerance), and the kept subset of exact samples becomes
    the arc ATF.
    """
    from mamut_routing_lib.td.artifacts import InstanceATFs

    td = instance.td
    if not isinstance(td, TDRoadGraphRef):
        raise RoadGraphFormatError(f"instance td model is {td.model!r}, expected road-graph")
    _validate_road_graph(road)
    _validate_traffic_overlay(overlay)
    validate_overlay_against_road(overlay, road)
    if road.num_customers != instance.num_customers:
        raise RoadGraphFormatError(
            f"road-graph num_customers {road.num_customers} does not match "
            f"instance {instance.num_customers}"
        )
    instance_horizon = (float(instance.horizon[0]), float(instance.horizon[1]))
    if road.horizon != instance_horizon:
        raise RoadGraphFormatError(
            f"road-graph horizon {road.horizon} does not match instance horizon {instance_horizon}"
        )
    if float(td.sample_step) != road.sample_step:
        raise RoadGraphFormatError(
            f"td block sample_step {td.sample_step} does not match road sidecar {road.sample_step}"
        )
    if float(td.simplify_tolerance) != road.simplify_tolerance:
        raise RoadGraphFormatError(
            f"td block simplify_tolerance {td.simplify_tolerance} does not match "
            f"road sidecar {road.simplify_tolerance}"
        )

    zones = [(road.bin_edges[k], road.bin_edges[k + 1]) for k in range(len(road.bin_edges) - 1)]
    extended_horizon = (road.horizon[0], road.extension_end)
    tolerance = road.simplify_tolerance
    adjacency = build_adjacency(road)
    node_vertices = road.node_vertices
    num_nodes = road.num_customers + 1
    grid = departure_grid(road.horizon, road.sample_step)
    grid_root = _np.asarray(grid, dtype=_np.float64) if _np is not None else list(grid)

    evaluators: dict[int, _EdgeEvaluator] = {}

    def edge_evaluator(edge_index: int) -> _EdgeEvaluator:
        evaluator = evaluators.get(edge_index)
        if evaluator is None:
            _, _, length, _ = road.edges[edge_index]
            speeds = overlay.edge_speeds[edge_index]
            evaluator = _EdgeEvaluator(build_arc_atf(zones, speeds, length, extended_horizon))
            evaluators[edge_index] = evaluator
        return evaluator

    arcs: dict[tuple[int, int], NDCPWLF] = {}
    with _gc_paused():
        for source_node in range(num_nodes):
            source_vertex = node_vertices[source_node]
            _, pred_edge = compute_fastest_path_tree(road, adjacency, source_vertex)
            arrivals = {source_vertex: grid_root}
            for target_node in range(num_nodes):
                if target_node == source_node:
                    continue
                target_vertex = node_vertices[target_node]
                chain: list[int] = []
                vertex = target_vertex
                while vertex not in arrivals:
                    edge_index = pred_edge[vertex]
                    if edge_index < 0:
                        raise RoadGraphFormatError(
                            f"node {target_node} (OSM {road.vertex_osm_ids[target_vertex]}) "
                            f"unreachable from node {source_node} "
                            f"(OSM {road.vertex_osm_ids[source_vertex]})"
                        )
                    chain.append(vertex)
                    vertex = road.edges[edge_index][0]
                for vertex in reversed(chain):
                    edge_index = pred_edge[vertex]
                    parent = road.edges[edge_index][0]
                    values = edge_evaluator(edge_index).evaluate_many(arrivals[parent])
                    last = float(values[-1])
                    if last > road.extension_end:
                        raise RoadGraphFormatError(
                            f"arrival {last} exceeds extension_end {road.extension_end} "
                            f"— extension_end too small"
                        )
                    arrivals[vertex] = values
                samples = arrivals[target_vertex]
                ys = [float(v) for v in samples] if _np is not None else list(samples)
                xs, ys = _drop_exactly_collinear(grid, ys)
                xs, ys = _simplify_points(xs, ys, tolerance)
                arcs[(source_node, target_node)] = NDCPWLF(list(xs), list(ys), validate=False)
            # Free per-source memo before the next Dijkstra; keeps peak memory
            # to one shortest-path tree of arrival vectors at a time.
            del arrivals
    return InstanceATFs(
        instance_name=instance.instance_name,
        benchmark_name=instance.benchmark_name.value,
        horizon=road.horizon,
        num_customers=road.num_customers,
        arcs=arcs,
        generator=dict(ROAD_MATERIALIZER_GENERATOR),
    )


def materialize_selected_atfs_roadgraph(
    instance: AnyTDBenchmarkInstance,
    road: InstanceRoadGraph,
    overlay: TrafficOverlay,
    selected_arcs: set[tuple[int, int]],
) -> dict[tuple[int, int], NDCPWLF]:
    """Materialize only selected complete-graph arcs with canonical semantics.

    This is the sparse counterpart of :func:`materialize_instance_atfs_roadgraph`.
    It uses the same pinned paths, departure grid, edge evaluation and
    simplification, but avoids constructing the quadratic matrix when a
    generation audit needs only the arcs of a route certificate.
    """
    td = instance.td
    if not isinstance(td, TDRoadGraphRef):
        raise RoadGraphFormatError(f"instance td model is {td.model!r}, expected road-graph")
    _validate_road_graph(road)
    _validate_traffic_overlay(overlay)
    validate_overlay_against_road(overlay, road)
    if road.num_customers != instance.num_customers:
        raise RoadGraphFormatError("road-graph customer count does not match instance")
    num_nodes = road.num_customers + 1
    for source, target in selected_arcs:
        if not (0 <= source < num_nodes and 0 <= target < num_nodes) or source == target:
            raise RoadGraphFormatError(f"invalid selected arc {(source, target)}")

    zones = [(road.bin_edges[k], road.bin_edges[k + 1]) for k in range(len(road.bin_edges) - 1)]
    extended_horizon = (road.horizon[0], road.extension_end)
    adjacency = build_adjacency(road)
    grid = departure_grid(road.horizon, road.sample_step)
    grid_root = _np.asarray(grid, dtype=_np.float64) if _np is not None else list(grid)
    evaluators: dict[int, _EdgeEvaluator] = {}

    def edge_evaluator(edge_index: int) -> _EdgeEvaluator:
        evaluator = evaluators.get(edge_index)
        if evaluator is None:
            _, _, length, _ = road.edges[edge_index]
            evaluator = _EdgeEvaluator(
                build_arc_atf(zones, overlay.edge_speeds[edge_index], length, extended_horizon)
            )
            evaluators[edge_index] = evaluator
        return evaluator

    targets_by_source: dict[int, list[int]] = {}
    for source, target in sorted(selected_arcs):
        targets_by_source.setdefault(source, []).append(target)
    arcs: dict[tuple[int, int], NDCPWLF] = {}
    with _gc_paused():
        for source_node, targets in targets_by_source.items():
            source_vertex = road.node_vertices[source_node]
            _, pred_edge = compute_fastest_path_tree(road, adjacency, source_vertex)
            arrivals = {source_vertex: grid_root}
            for target_node in targets:
                target_vertex = road.node_vertices[target_node]
                chain: list[int] = []
                vertex = target_vertex
                while vertex not in arrivals:
                    edge_index = pred_edge[vertex]
                    if edge_index < 0:
                        raise RoadGraphFormatError(
                            f"node {target_node} unreachable from node {source_node}"
                        )
                    chain.append(vertex)
                    vertex = road.edges[edge_index][0]
                for vertex in reversed(chain):
                    edge_index = pred_edge[vertex]
                    parent = road.edges[edge_index][0]
                    values = edge_evaluator(edge_index).evaluate_many(arrivals[parent])
                    if float(values[-1]) > road.extension_end:
                        raise RoadGraphFormatError("selected-arc arrival exceeds extension_end")
                    arrivals[vertex] = values
                ys = [float(value) for value in arrivals[target_vertex]]
                xs, ys = _drop_exactly_collinear(grid, ys)
                xs, ys = _simplify_points(xs, ys, road.simplify_tolerance)
                arcs[(source_node, target_node)] = NDCPWLF(list(xs), list(ys), validate=False)
    return arcs

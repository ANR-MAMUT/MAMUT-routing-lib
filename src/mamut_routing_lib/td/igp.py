"""IGP (Ichoua-Gendreau-Potvin 2003) engine for the ``igp-profile`` td model.

Three responsibilities:

1. The exact IGP -> arrival-time NDCPWLF consolidation (``ichoua_travel_time``
   + ``build_arc_atf``) -- the robust-marching construction previously
   maintained in the tdvrptw-workspace migration scripts (``igp_atf.py``),
   promoted here as the single source of truth. Breakpoints sit exactly where
   the departure or the arrival crosses a speed-zone boundary, each
   re-evaluated with the exact forward Ichoua loop so there is no drift; a
   degenerate-boundary skip and an ulp-scale monotone clamp harden the
   marching (byte-identical to the strict construction wherever that one
   validates, e.g. all of Dabia2013).

2. The arc-category sidecar (``<Base>.igp.json[.gz]``, format
   ``mamut-td-igp-categories``): an explicit symmetric category matrix stored
   as digit strings, hashed over its uncompressed canonical JSON bytes like
   every other TD sidecar.

3. The deterministic materialization of the canonical ``InstanceATFs`` from
   an ``igp-profile`` instance: Euclidean distances ``sqrt(dx*dx + dy*dy)``
   from the instance coordinates, one ``build_arc_atf`` per arc of the
   complete graph, and a fixed materializer ``generator`` constant so the
   resulting ``atf_sha256`` is reproducible from the published data alone.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mamut_routing_lib.td.models import (
    IGP_CATEGORIES_FORMAT,
    IGP_CATEGORIES_FORMAT_VERSION,
    AnyTDBenchmarkInstance,
    TDIGPProfileRef,
)
from mamut_routing_lib.td.pwlf import NDCPWLF

if TYPE_CHECKING:
    from mamut_routing_lib.td.artifacts import InstanceATFs

IGP_PLAIN_SUFFIX = ".igp.json"
IGP_GZIP_SUFFIX = ".igp.json.gz"

#: Fixed header constant of materialized sidecars. Materialization is defined
#: by the instance td block plus the TD benchmark standard, not by the tool
#: that generated the instance, so the constant never carries tool provenance
#: (that lives in the instance ``metadata``) -- this is what makes
#: ``atf_sha256`` reproducible from the published data alone.
IGP_MATERIALIZER_GENERATOR: dict[str, Any] = {"name": "igp-profile-materializer", "version": 1}


class IGPFormatError(ValueError):
    """Raised when an IGP categories sidecar violates the canonical format."""


def ichoua_travel_time(zones, speeds, distance, t0):
    """Exact forward IGP loop; last-zone speed extended beyond the horizon."""
    k = 0
    while k < len(zones) - 1 and t0 > zones[k][1]:
        k += 1
    t = t0
    d = distance
    tt = t + d / speeds[k]
    while k < len(zones) - 1 and tt > zones[k][1]:
        d = d - speeds[k] * (zones[k][1] - t)
        t = zones[k][1]
        if d <= 0.0:
            break
        k += 1
        tt = t + d / speeds[k]
    return tt - t0


def build_arc_atf(zones, speeds, distance, horizon):
    """Exact PWL arrival function: breakpoints at every zone-boundary crossing.

    Zone bounds are coerced to floats: canonical ATF breakpoints must be
    serialized as JSON floats (``1080.0``), and integer zone bounds would
    otherwise leak python ints into ``xs`` and break the write/reload sha256
    round-trip (``1080`` vs ``1080.0``).
    """
    zones = [(float(a), float(b)) for a, b in zones]
    t_end = float(horizon[1])
    if distance == 0.0:
        return NDCPWLF([float(horizon[0]), t_end], [float(horizon[0]), t_end])

    def zone_right(x):
        # Zone governing the speed immediately AFTER instant x (forward march).
        k = 0
        while k < len(zones) - 1 and x >= zones[k][1]:
            k += 1
        return k

    xs, ys = [], []
    t = horizon[0]
    while True:
        a = t + ichoua_travel_time(zones, speeds, distance, t)
        if ys and a < ys[-1]:
            a = ys[-1]  # monotone clamp (ulp-scale rounding only)
        xs.append(t)
        ys.append(a)
        if t >= t_end:
            break
        kd = zone_right(t)
        next_dep = zones[kd][1] if kd < len(zones) - 1 else t_end
        # First arrival-boundary crossing strictly after t. The arrival moves
        # at rate speeds[kd]/speeds[ka]; a candidate that fails to advance
        # (previous event one ulp below the boundary) is degenerate -- its
        # breakpoint is the one just emitted -- so target the next boundary.
        ka = zone_right(a)
        next_arr = t_end
        while ka < len(zones) - 1:
            candidate = t + (zones[ka][1] - a) * speeds[ka] / speeds[kd]
            if candidate > t:
                next_arr = candidate
                break
            ka += 1
        t_next = min(next_dep, next_arr, t_end)
        if t_next <= t:
            raise RuntimeError(f"marching stalled at t={t} (distance {distance})")
        t = t_next
    return NDCPWLF(xs, ys)


@dataclass
class InstanceCategories:
    """In-memory content of an IGP categories sidecar.

    ``categories`` holds ``num_customers + 1`` digit strings of the same
    length; ``categories[i][j]`` is the category of arc ``(i, j)``. The matrix
    is symmetric with a fixed ``'0'`` diagonal (self-arcs do not exist; the
    value is never read).
    """

    base_name: str
    benchmark_name: str
    num_customers: int
    num_categories: int
    categories: list[str]
    generator: dict[str, Any] = field(default_factory=dict)
    format_version: int = IGP_CATEGORIES_FORMAT_VERSION

    def category(self, i: int, j: int) -> int:
        return int(self.categories[i][j])


def _validate_categories(cats: InstanceCategories) -> None:
    num_vertices = cats.num_customers + 1
    if cats.num_categories <= 0 or cats.num_categories > 10:
        raise IGPFormatError(f"num_categories must be in 1..10 (digit encoding), got {cats.num_categories}")
    if len(cats.categories) != num_vertices:
        raise IGPFormatError(
            f"expected {num_vertices} category rows (num_customers + 1), found {len(cats.categories)}"
        )
    for i, row in enumerate(cats.categories):
        if len(row) != num_vertices:
            raise IGPFormatError(f"category row {i} has length {len(row)}, expected {num_vertices}")
        if not row.isdigit():
            raise IGPFormatError(f"category row {i} contains non-digit characters")
    for i in range(num_vertices):
        if cats.categories[i][i] != "0":
            raise IGPFormatError(f"diagonal entry ({i}, {i}) must be '0', got {cats.categories[i][i]!r}")
        for j in range(i + 1, num_vertices):
            c = cats.categories[i][j]
            if int(c) >= cats.num_categories:
                raise IGPFormatError(
                    f"category {c} at ({i}, {j}) out of range for num_categories={cats.num_categories}"
                )
            if cats.categories[j][i] != c:
                raise IGPFormatError(
                    f"category matrix must be symmetric: ({i}, {j}) = {c}, "
                    f"({j}, {i}) = {cats.categories[j][i]}"
                )


def categories_to_canonical_json_bytes(cats: InstanceCategories) -> bytes:
    """Serialize to the canonical JSON bytes (the input of ``categories_sha256``).

    Fixed key order, one category row per line, gzip-independent.
    """
    header_lines = [
        "{",
        f'    "format": {json.dumps(IGP_CATEGORIES_FORMAT)},',
        f'    "format_version": {json.dumps(cats.format_version)},',
        f'    "base_name": {json.dumps(cats.base_name)},',
        f'    "benchmark_name": {json.dumps(cats.benchmark_name)},',
        f'    "num_customers": {json.dumps(cats.num_customers)},',
        f'    "num_categories": {json.dumps(cats.num_categories)},',
        f'    "generator": {json.dumps(cats.generator, sort_keys=True)},',
        '    "categories": [',
    ]
    body = ",\n".join("        " + json.dumps(row) for row in cats.categories)
    text = "\n".join(header_lines) + "\n" + body + "\n    ]\n}\n"
    return text.encode("utf-8")


def compute_categories_sha256(cats: InstanceCategories) -> str:
    return hashlib.sha256(categories_to_canonical_json_bytes(cats)).hexdigest()


def save_instance_categories(cats: InstanceCategories, path: str | Path) -> None:
    """Write the sidecar; gzip iff the path ends with ``.igp.json.gz`` (``mtime=0``)."""
    _validate_categories(cats)
    target = Path(path)
    data = categories_to_canonical_json_bytes(cats)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name.endswith(IGP_GZIP_SUFFIX):
        target.write_bytes(gzip.compress(data, mtime=0))
    elif target.name.endswith(IGP_PLAIN_SUFFIX):
        target.write_bytes(data)
    else:
        raise IGPFormatError(f"categories path must end with {IGP_PLAIN_SUFFIX} or {IGP_GZIP_SUFFIX}: {target.name}")


def load_instance_categories(path: str | Path) -> InstanceCategories:
    source = Path(path)
    raw = source.read_bytes()
    if source.name.endswith(IGP_GZIP_SUFFIX):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != IGP_CATEGORIES_FORMAT:
        raise IGPFormatError(f"unexpected format marker: {payload.get('format')!r}")
    if payload.get("format_version") != IGP_CATEGORIES_FORMAT_VERSION:
        raise IGPFormatError(f"unsupported format_version: {payload.get('format_version')!r}")
    cats = InstanceCategories(
        base_name=str(payload["base_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        num_customers=int(payload["num_customers"]),
        num_categories=int(payload["num_categories"]),
        categories=[str(row) for row in payload["categories"]],
        generator=dict(payload.get("generator", {})),
    )
    _validate_categories(cats)
    return cats


def euclidean_distance(a, b) -> float:
    """Canonical distance of the igp-profile model: ``sqrt(dx*dx + dy*dy)``.

    IEEE-754 binary64 with correctly rounded ``sqrt`` -- fully deterministic,
    and bit-identical to the stored Sintef2008 ``arc_costs`` on integer G&H
    coordinates.
    """
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def materialize_instance_atfs(
    instance: AnyTDBenchmarkInstance,
    categories: InstanceCategories,
) -> "InstanceATFs":
    """Build the canonical complete-graph ``InstanceATFs`` of an igp-profile instance."""
    from mamut_routing_lib.td.artifacts import InstanceATFs

    td = instance.td
    if not isinstance(td, TDIGPProfileRef):
        raise IGPFormatError(f"instance td model is {td.model!r}, expected igp-profile")
    if categories.num_customers != instance.num_customers:
        raise IGPFormatError(
            f"categories num_customers {categories.num_customers} does not match "
            f"instance {instance.num_customers}"
        )
    if categories.num_categories != td.num_categories():
        raise IGPFormatError(
            f"categories num_categories {categories.num_categories} does not match "
            f"the {td.num_categories()} speed rows of the td block"
        )

    horizon = (float(instance.horizon[0]), float(instance.horizon[1]))
    zones = [(float(a), float(b)) for a, b in td.time_periods]
    speeds = [[float(v) for v in row] for row in td.speeds]
    coordinates = instance.coordinates
    num_vertices = instance.num_customers + 1

    arcs: dict[tuple[int, int], NDCPWLF] = {}
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j:
                continue
            distance = euclidean_distance(coordinates[i], coordinates[j])
            arcs[(i, j)] = build_arc_atf(zones, speeds[categories.category(i, j)], distance, horizon)

    return InstanceATFs(
        instance_name=instance.instance_name,
        benchmark_name=instance.benchmark_name.value,
        horizon=horizon,
        num_customers=instance.num_customers,
        arcs=arcs,
        generator=dict(IGP_MATERIALIZER_GENERATOR),
    )

"""Distance-matrix sidecar of a collection base instance (``mamut-distances``).

One sidecar per (base, metric): the full ``(n+1) x (n+1)`` matrix of the
``fastest`` (free-flow travel times) or ``shortest`` (path lengths) metric,
values rounded to the family's precision (Poryos2026: 3 decimals). Slim
CVRP/VRPTW instances reference it by sha-pinned collection-relative path
instead of embedding the matrix; the ``fastest`` sidecar of a base must equal
the free-flow node-to-node times of its road-graph sidecar after the same
rounding (generation gate). Hashed over its uncompressed canonical JSON
bytes, gzip ``mtime=0``, like every other sidecar.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mamut_routing_lib.exact_types import binary64_operand, nonnegative_integer
from mamut_routing_lib.json_utils import strict_json_loads

DISTANCES_FORMAT = "mamut-distances"
DISTANCES_FORMAT_VERSION = 1
DISTANCES_PLAIN_SUFFIX = ".json"
DISTANCES_GZIP_SUFFIX = ".json.gz"
DISTANCES_INFIX = ".distances-"


class DistancesFormatError(ValueError):
    """Raised when a distances sidecar violates the canonical format."""


@dataclass
class InstanceDistances:
    """In-memory content of a distances sidecar."""

    base_name: str
    benchmark_name: str
    metric: str
    num_customers: int
    values: list[list[float]]
    generator: dict[str, Any] = field(default_factory=dict)
    format_version: int = DISTANCES_FORMAT_VERSION

    def __post_init__(self) -> None:
        try:
            self.num_customers = nonnegative_integer(self.num_customers, "num_customers")
            self.values = [[binary64_operand(v) for v in row] for row in self.values]
        except (TypeError, ValueError) as exc:
            raise DistancesFormatError(f"invalid distance matrix input: {exc}") from exc


def _validate_distances(distances: InstanceDistances) -> None:
    if type(distances.format_version) is not int or distances.format_version != DISTANCES_FORMAT_VERSION:
        raise DistancesFormatError(f"unsupported format_version: {distances.format_version!r}")
    if not distances.base_name:
        raise DistancesFormatError("base_name must be non-empty")
    if not distances.metric:
        raise DistancesFormatError("metric must be non-empty")
    if type(distances.num_customers) is not int or distances.num_customers < 0:
        raise DistancesFormatError("num_customers must be a nonnegative integer")
    expected = distances.num_customers + 1
    if len(distances.values) != expected:
        raise DistancesFormatError(
            f"values must have {expected} rows (num_customers + 1), found {len(distances.values)}"
        )
    for i, row in enumerate(distances.values):
        if len(row) != expected:
            raise DistancesFormatError(f"values row {i} has {len(row)} entries, expected {expected}")
        for j, value in enumerate(row):
            try:
                finite = type(value) in (float, int) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise DistancesFormatError(f"entry [{i}][{j}] must be a finite binary64 number")
            if i == j:
                if value != 0.0:
                    raise DistancesFormatError(f"diagonal entry [{i}][{j}] must be 0.0, got {value}")
            elif value <= 0:
                raise DistancesFormatError(f"entry [{i}][{j}] = {value} must be strictly positive")


def distances_to_canonical_json_bytes(distances: InstanceDistances) -> bytes:
    """Serialize to the canonical JSON bytes (the input of the distances sha256).

    Fixed key order, one matrix row per line, floats via Python's shortest
    round-trip repr, gzip-independent.
    """
    _validate_distances(distances)
    header_lines = [
        "{",
        f'    "format": {json.dumps(DISTANCES_FORMAT)},',
        f'    "format_version": {json.dumps(distances.format_version)},',
        f'    "base_name": {json.dumps(distances.base_name)},',
        f'    "benchmark_name": {json.dumps(distances.benchmark_name)},',
        f'    "metric": {json.dumps(distances.metric)},',
        f'    "num_customers": {json.dumps(distances.num_customers)},',
        f'    "generator": {json.dumps(distances.generator, sort_keys=True)},',
        '    "values": [',
    ]
    body = ",\n".join("        " + json.dumps(row) for row in distances.values)
    text = "\n".join(header_lines) + "\n" + body + "\n    ]\n}\n"
    return text.encode("utf-8")


def compute_distances_sha256(distances: InstanceDistances) -> str:
    return hashlib.sha256(distances_to_canonical_json_bytes(distances)).hexdigest()


def save_instance_distances(distances: InstanceDistances, path: str | Path) -> None:
    """Write the sidecar; gzip iff the path ends with ``.json.gz`` (``mtime=0``).

    The conventional name is ``<base>.distances-<metric>.json[.gz]``; the
    ``.distances-`` infix is required so the sidecar is never confused with
    other artifact kinds.
    """
    target = Path(path)
    if DISTANCES_INFIX not in target.name:
        raise DistancesFormatError(f"distances path must contain {DISTANCES_INFIX!r}: {target.name}")
    data = distances_to_canonical_json_bytes(distances)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name.endswith(DISTANCES_GZIP_SUFFIX):
        target.write_bytes(gzip.compress(data, mtime=0))
    elif target.name.endswith(DISTANCES_PLAIN_SUFFIX):
        target.write_bytes(data)
    else:
        raise DistancesFormatError(
            f"distances path must end with {DISTANCES_PLAIN_SUFFIX} or {DISTANCES_GZIP_SUFFIX}: {target.name}"
        )


def load_instance_distances(path: str | Path) -> InstanceDistances:
    source = Path(path)
    raw = source.read_bytes()
    if source.name.endswith(DISTANCES_GZIP_SUFFIX):
        raw = gzip.decompress(raw)
    try:
        payload = strict_json_loads(raw)
    except ValueError as exc:
        raise DistancesFormatError(f"invalid distances JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DistancesFormatError("expected a distances JSON object")
    if payload.get("format") != DISTANCES_FORMAT:
        raise DistancesFormatError(f"unexpected format marker: {payload.get('format')!r}")
    if type(payload.get("format_version")) is not int or payload["format_version"] != DISTANCES_FORMAT_VERSION:
        raise DistancesFormatError(f"unsupported format_version: {payload.get('format_version')!r}")
    distances = InstanceDistances(
        base_name=str(payload["base_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        metric=str(payload["metric"]),
        num_customers=payload["num_customers"],
        values=payload["values"],
        generator=dict(payload.get("generator", {})),
    )
    _validate_distances(distances)
    return distances

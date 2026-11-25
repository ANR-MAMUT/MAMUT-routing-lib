"""Verify and fingerprint the complete TD materialization while retaining only requested arcs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from gzip import open as gzip_open
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
from struct import pack

from mamut_routing_lib.json_utils import strict_json_loads
from mamut_routing_lib.sidecars import validate_relative_sidecar_path
from mamut_routing_lib.td.artifacts import (
    InstanceATFs, _canonical_atf_arc, atfs_to_canonical_json_bytes, compute_atf_sha256, load_instance_atfs, load_td_benchmark_instance,
)
from mamut_routing_lib.td.models import ATF_FORMAT, ATF_FORMAT_VERSION
from mamut_routing_lib.td.igp import (
    IGP_MATERIALIZER_GENERATOR, compute_categories_sha256, load_instance_categories, materialize_selected_atfs_igp,
)
from mamut_routing_lib.td.pwlf import NDCPWLF
from mamut_routing_lib.td.roadgraph import (
    ROAD_MATERIALIZER_GENERATOR, compute_road_graph_sha256, compute_traffic_overlay_sha256,
    load_instance_road_graph, load_traffic_overlay, materialize_selected_atfs_roadgraph, validate_overlay_against_road,
)


@dataclass
class TDMaterialization:
    operands: InstanceATFs
    canonical_sha256: str
    atf_stream_sha256: str
    sidecars: list[dict]
    num_arcs: int
    num_knots: int


def _match(actual: str, expected: str | None, subject: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{subject}: expected {expected}, computed {actual}")


def _binary64(value: int | float) -> float:
    if type(value) not in (int, float):
        raise ValueError("ATF operand must be a finite JSON number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("ATF operand overflows binary64") from error
    if not math.isfinite(value):
        raise ValueError("ATF operand must be finite")
    return value if value else 0.0


def _digest_arcs(operands: InstanceATFs, arcs: Iterable, selected: set | None):
    n = operands.num_vertices()
    lo, hi = map(_binary64, operands.horizon)
    if not (0 <= lo < hi):
        raise ValueError(f"invalid horizon {operands.horizon}")
    digest = sha256(f"MAMUT-ATF-v1\nbinary64\n{n}\n".encode("ascii") + pack(">dd", lo, hi))
    count = 0
    num_knots = 0
    rows = iter(arcs)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = next(rows, None)
            if row is None or len(row) != 4:
                raise ValueError(f"missing arc ({i}, {j})")
            a, b, xs, ys = row
            if type(a) is not int or type(b) is not int or (a, b) != (i, j):
                raise ValueError(f"expected arc ({i}, {j}), got ({a}, {b})")
            if not isinstance(xs, (list, tuple)) or not isinstance(ys, (list, tuple)):
                raise ValueError(f"invalid knot arrays on ({i}, {j})")
            if len(xs) != len(ys) or len(xs) < 2 or xs[0] != lo or xs[-1] != hi:
                raise ValueError(f"invalid knot arrays on ({i}, {j})")
            coords = []
            previous_x = previous_y = lo
            for x, y in zip(xs, ys):
                x, y = _binary64(x), _binary64(y)
                if x < previous_x or y < previous_y or y < x:
                    raise ValueError(f"invalid knot on ({i}, {j})")
                coords.extend((x, y))
                previous_x, previous_y = x, y
            digest.update(f"{i},{j},{len(xs)}\n".encode("ascii"))
            digest.update(pack(f">{len(coords)}d", *coords))
            if selected is None or (i, j) in selected:
                operands.arcs[i, j] = NDCPWLF(list(xs), list(ys))
            count += 1
            num_knots += len(xs)
    if next(rows, None) is not None or (selected is not None and selected != operands.arcs.keys()):
        raise ValueError("extra or missing requested arc")
    return digest.hexdigest(), count, num_knots


def _stored_records(stream, canonical, instance, operands) -> Iterator:
    """Parse the library's canonical JSON format, which puts one complete arc on each line."""
    header = bytearray()
    while True:
        line = stream.readline(65537)
        if not line or not line.endswith(b"\n") or len(header) + len(line) > 65536:
            raise ValueError("noncanonical ATF header")
        canonical.update(line)
        header.extend(line)
        if line == b'    "arcs": [\n':
            break
    info = strict_json_loads(bytes(header) + b"]}")
    if type(info.get("num_customers")) is not int or type(info.get("format_version")) is not int:
        raise ValueError("stored ATF dimensions and version must be integers")
    if list(map(_binary64, info["horizon"])) != list(instance.horizon):
        raise ValueError("stored ATF horizon differs from descriptor")
    expected = {"format": ATF_FORMAT, "format_version": ATF_FORMAT_VERSION, "instance_name": instance.instance_name,
                "benchmark_name": instance.benchmark_name.value, "num_customers": instance.num_customers,
                "horizon": list(instance.horizon)}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"stored ATF {key} differs from descriptor")
    operands.generator = info.get("generator", {})
    last_record = False
    for line in stream:
        canonical.update(line)
        if line == b"    ]\n":
            if not last_record and instance.num_customers:
                raise ValueError("missing last ATF arc or unexpected trailing comma")
            break
        if line == b"\n" and instance.num_customers == 0:
            continue
        if last_record or not line.startswith(b"        ["):
            raise ValueError("malformed canonical ATF arc record")
        body = line.strip()
        last_record = not body.endswith(b",")
        yield strict_json_loads(body.removesuffix(b","))
    tail = stream.read()
    canonical.update(tail)
    if tail != b"}\n":
        raise ValueError("unexpected stored ATF footer")


def _generated_records(operands, materialize, canonical, block_arcs):
    empty = atfs_to_canonical_json_bytes(operands)
    header, sep, footer = empty.rpartition(b"\n    ]\n}\n")
    if not sep or footer:
        raise ValueError("unrecognized canonical ATF encoding")
    canonical.update(header)
    n = operands.num_vertices()
    rows_per_block = max(1, block_arcs // max(1, n - 1))
    first = True
    for start in range(0, n, rows_per_block):
        keys = {(i, j) for i in range(start, min(start + rows_per_block, n)) for j in range(n) if j != i}
        block = materialize(keys)
        for (i, j), arc in sorted(block.items()):
            row = _canonical_atf_arc(i, j, arc)
            canonical.update((b"" if first else b",\n") + b"        " + json.dumps(row).encode("ascii"))
            first = False
            yield row
        del block, keys
    canonical.update(sep)


def _sidecar_record(root: Path, name: str) -> dict:
    path = root / name
    decompressed = path.suffix == ".gz"
    with (gzip_open if decompressed else open)(path, "rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"path": name, "sha256": digest, "scope": "decompressed" if decompressed else "file"}


def stream_td_materialization(instance_path: Path, family_root: Path, selected_arcs=(), *,
                              atf_cache_root: Path | None = None, block_arcs: int = 8192) -> TDMaterialization:
    """Check the published ATF pin and hash every directed arc before returning sparse route operands.

    A cache holds generated Blauth ATFs under the sidecar path from the descriptor. Both a
    cached and an in-tree ATF must reproduce the descriptor's full canonical SHA-256.
    """
    if isinstance(block_arcs, bool) or not isinstance(block_arcs, int) or block_arcs < 1:
        raise ValueError("block_arcs must be a positive integer")
    instance = load_td_benchmark_instance(instance_path)
    td = instance.td
    selected = None if selected_arcs is None else set(selected_arcs)
    operands = InstanceATFs(instance.instance_name, instance.benchmark_name.value, tuple(instance.horizon),
                            instance.num_customers, {})
    canonical = sha256()
    if td.model == "atf-ndcpwlf":
        path = family_root / validate_relative_sidecar_path(td.atf_path)
        stored = path.is_file()
        if not stored and atf_cache_root is not None:
            path = atf_cache_root / validate_relative_sidecar_path(td.atf_path)
        try:
            with (gzip_open if path.suffix == ".gz" else open)(path, "rb") as stream:
                records = _stored_records(stream, canonical, instance, operands)
                digest, count, knots = _digest_arcs(operands, records, selected)
            raw_sha = canonical.hexdigest()
            if raw_sha != td.atf_sha256:
                raise ValueError("the input bytes require canonical serialization")
            canonical_sha = raw_sha
        except (ValueError, KeyError, TypeError):
            # Ordinary ATF JSON may reorder object members or use different whitespace.
            # Only the published canonical byte layout admits the line-wise fast path.
            full = load_instance_atfs(path)
            if (full.instance_name != instance.instance_name or full.benchmark_name != instance.benchmark_name.value
                    or full.num_customers != instance.num_customers or tuple(full.horizon) != tuple(instance.horizon)):
                raise ValueError("stored ATF identity or horizon differs from descriptor")
            operands.generator = full.generator
            operands.arcs.clear()
            rows = ((i, j, arc.xs, arc.ys) for (i, j), arc in sorted(full.arcs.items()))
            digest, count, knots = _digest_arcs(operands, rows, selected)
            canonical_sha = compute_atf_sha256(full)
            with (gzip_open if path.suffix == ".gz" else open)(path, "rb") as stream:
                raw_sha = file_digest(stream, "sha256").hexdigest()
        sidecars = ([{"path": td.atf_path, "sha256": raw_sha,
                      "scope": "decompressed" if path.suffix == ".gz" else "file"}] if stored else [])
    else:
        if td.model == "igp-profile":
            cats = load_instance_categories(family_root / validate_relative_sidecar_path(td.categories_path))
            _match(compute_categories_sha256(cats), td.categories_sha256, td.categories_path)
            if cats.benchmark_name != instance.benchmark_name.value or cats.num_categories != td.num_categories():
                raise ValueError("category table does not describe this TD instance")
            factory = partial(materialize_selected_atfs_igp, instance, cats)
            operands.generator = dict(IGP_MATERIALIZER_GENERATOR)
            names = [td.categories_path]
        elif td.model == "road-graph":
            road = load_instance_road_graph(family_root / validate_relative_sidecar_path(td.graph.path))
            overlay = load_traffic_overlay(family_root / validate_relative_sidecar_path(td.traffic.path))
            _match(compute_road_graph_sha256(road), td.graph.sha256, td.graph.path)
            _match(compute_traffic_overlay_sha256(overlay), td.traffic.sha256, td.traffic.path)
            validate_overlay_against_road(overlay, road)
            if road.benchmark_name != instance.benchmark_name.value:
                raise ValueError("road graph does not describe this family")
            if (tuple(road.horizon) != tuple(instance.horizon) or road.sample_step != td.sample_step
                    or road.simplify_tolerance != td.simplify_tolerance):
                raise ValueError("graph and descriptor materialization parameters differ")
            factory = partial(materialize_selected_atfs_roadgraph, instance, road, overlay)
            operands.generator = dict(ROAD_MATERIALIZER_GENERATOR)
            names = [td.graph.path, td.traffic.path]
        else:
            raise ValueError(f"unsupported TD model {td.model}")
        sidecars = [_sidecar_record(family_root, name) for name in names]
        records = _generated_records(operands, factory, canonical, block_arcs)
        digest, count, knots = _digest_arcs(operands, records, selected)
        canonical_sha = canonical.hexdigest()
    _match(canonical_sha, td.atf_sha256, str(instance_path))
    return TDMaterialization(operands, canonical_sha, digest, sidecars, count, knots)

"""IO for TD benchmark artifacts: ATF sidecar files and TD instance loading.

The ATF sidecar (``<Instance>.atf.json`` or ``<Instance>.atf.json.gz``) is the
canonical ground truth of a TD instance: one arrival-time NDCPWLF per arc of
the complete customer-based graph. Plain JSON is used for small instances
(debuggable at a glance), gzip for large ones; the loader accepts both. The
gzip form is written with ``mtime=0`` so bytes are deterministic, and the
``atf_sha256`` recorded in instance files always hashes the *uncompressed*
canonical JSON bytes, making it storage-form independent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mamut_routing_lib.json_utils import load_json_from_file
from mamut_routing_lib.td.models import (
    ATF_FORMAT,
    ATF_FORMAT_VERSION,
    AnyTDBenchmarkInstance,
    BenchmarkInstanceTDVRP,
    BenchmarkInstanceTDVRPTW,
    TDIGPProfileRef,
)
from mamut_routing_lib.td.pwlf import NDCPWLF, PWLFError

ATF_PLAIN_SUFFIX = ".atf.json"
ATF_GZIP_SUFFIX = ".atf.json.gz"


class ATFFormatError(ValueError):
    """Raised when an ATF sidecar file violates the canonical format."""


@dataclass
class InstanceATFs:
    """In-memory content of an ATF sidecar file."""

    instance_name: str
    benchmark_name: str
    horizon: tuple[float, float]
    num_customers: int
    arcs: dict[tuple[int, int], NDCPWLF]
    generator: dict[str, Any] = field(default_factory=dict)
    format_version: int = ATF_FORMAT_VERSION

    def num_vertices(self) -> int:
        return self.num_customers + 1


def _validate_arc_function(i: int, j: int, atf: NDCPWLF, horizon: tuple[float, float]) -> None:
    if atf.num_breakpoints() < 2:
        raise ATFFormatError(f"arc ({i}, {j}): ATF must have at least 2 breakpoints")
    if atf.xs[0] != horizon[0] or atf.xs[-1] != horizon[1]:
        raise ATFFormatError(
            f"arc ({i}, {j}): ATF domain [{atf.xs[0]}, {atf.xs[-1]}] must span the horizon "
            f"[{horizon[0]}, {horizon[1]}]"
        )
    for k in range(atf.num_breakpoints()):
        if atf.ys[k] < atf.xs[k]:
            raise ATFFormatError(
                f"arc ({i}, {j}): arrival time {atf.ys[k]} before departure time {atf.xs[k]} "
                f"at breakpoint {k} (negative travel time)"
            )


def _atfs_from_payload(payload: dict[str, Any], *, validate_complete: bool = True) -> InstanceATFs:
    if payload.get("format") != ATF_FORMAT:
        raise ATFFormatError(f"unexpected format marker: {payload.get('format')!r}")
    if payload.get("format_version") != ATF_FORMAT_VERSION:
        raise ATFFormatError(f"unsupported format_version: {payload.get('format_version')!r}")

    horizon_raw = payload["horizon"]
    horizon = (float(horizon_raw[0]), float(horizon_raw[1]))
    num_customers = int(payload["num_customers"])
    num_vertices = num_customers + 1

    arcs: dict[tuple[int, int], NDCPWLF] = {}
    previous: tuple[int, int] | None = None
    for entry in payload["arcs"]:
        if len(entry) != 4:
            raise ATFFormatError(f"arc entry must be [i, j, xs, ys], got {entry!r}")
        i, j, xs, ys = entry
        if not (0 <= i < num_vertices and 0 <= j < num_vertices) or i == j:
            raise ATFFormatError(f"invalid arc indices ({i}, {j})")
        key = (int(i), int(j))
        if previous is not None and key <= previous:
            raise ATFFormatError(f"arcs must be sorted lexicographically; ({i}, {j}) after {previous}")
        previous = key
        try:
            atf = NDCPWLF([float(x) for x in xs], [float(y) for y in ys])
        except PWLFError as error:
            raise ATFFormatError(f"arc ({i}, {j}): {error}") from error
        _validate_arc_function(key[0], key[1], atf, horizon)
        arcs[key] = atf

    if validate_complete:
        expected = num_vertices * (num_vertices - 1)
        if len(arcs) != expected:
            raise ATFFormatError(
                f"expected a complete graph with {expected} arcs over {num_vertices} vertices, "
                f"found {len(arcs)}"
            )

    return InstanceATFs(
        instance_name=str(payload["instance_name"]),
        benchmark_name=str(payload["benchmark_name"]),
        horizon=horizon,
        num_customers=num_customers,
        arcs=arcs,
        generator=dict(payload.get("generator", {})),
    )


def atfs_to_canonical_json_bytes(atfs: InstanceATFs) -> bytes:
    """Serialize to the canonical JSON bytes (the input of ``atf_sha256``).

    Fixed key order, one arc per line, floats via Python's shortest
    round-trip repr.
    """
    header_lines = [
        "{",
        f'    "format": {json.dumps(ATF_FORMAT)},',
        f'    "format_version": {json.dumps(atfs.format_version)},',
        f'    "instance_name": {json.dumps(atfs.instance_name)},',
        f'    "benchmark_name": {json.dumps(atfs.benchmark_name)},',
        f'    "horizon": {json.dumps(list(atfs.horizon))},',
        f'    "num_customers": {json.dumps(atfs.num_customers)},',
        f'    "generator": {json.dumps(atfs.generator, sort_keys=True)},',
        '    "arcs": [',
    ]
    arc_lines = []
    for (i, j) in sorted(atfs.arcs):
        atf = atfs.arcs[(i, j)]
        arc_lines.append("        " + json.dumps([i, j, list(atf.xs), list(atf.ys)]))
    body = ",\n".join(arc_lines)
    text = "\n".join(header_lines) + "\n" + body + "\n    ]\n}\n"
    return text.encode("utf-8")


def compute_atf_sha256(atfs: InstanceATFs) -> str:
    return hashlib.sha256(atfs_to_canonical_json_bytes(atfs)).hexdigest()


def save_instance_atfs(atfs: InstanceATFs, path: str | Path) -> None:
    """Write the sidecar; gzip iff the path ends with ``.atf.json.gz``."""
    target = Path(path)
    data = atfs_to_canonical_json_bytes(atfs)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name.endswith(ATF_GZIP_SUFFIX):
        target.write_bytes(gzip.compress(data, mtime=0))
    elif target.name.endswith(ATF_PLAIN_SUFFIX):
        target.write_bytes(data)
    else:
        raise ATFFormatError(f"ATF path must end with {ATF_PLAIN_SUFFIX} or {ATF_GZIP_SUFFIX}: {target.name}")


def load_instance_atfs(path: str | Path, *, validate_complete: bool = True) -> InstanceATFs:
    source = Path(path)
    raw = source.read_bytes()
    if source.name.endswith(ATF_GZIP_SUFFIX):
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    return _atfs_from_payload(payload, validate_complete=validate_complete)


def get_atf_path_for_instance(instance_path: str | Path, atf_path: str | None = None) -> Path:
    """Resolve the sidecar path next to the instance file.

    When ``atf_path`` (from the instance's ``td`` block) is given it is
    resolved relative to the instance directory; otherwise the conventional
    sibling names are probed (plain first, then gzip).
    """
    base = Path(instance_path)
    if atf_path is not None:
        return base.parent / atf_path
    stem = base.name.removesuffix(".vrp.json")
    plain = base.parent / f"{stem}{ATF_PLAIN_SUFFIX}"
    if plain.exists():
        return plain
    return base.parent / f"{stem}{ATF_GZIP_SUFFIX}"


def load_td_benchmark_instance(instance_path: str | Path) -> AnyTDBenchmarkInstance:
    """Load and validate the instance file only (no sidecar)."""
    payload = load_json_from_file(instance_path)
    return td_instance_from_payload(payload)


def td_instance_from_payload(payload: dict[str, Any]) -> AnyTDBenchmarkInstance:
    if "time_windows" in payload:
        return BenchmarkInstanceTDVRPTW(**payload)
    return BenchmarkInstanceTDVRP(**payload)


@dataclass
class LoadedTDInstance:
    """A TD instance paired with its arrival-time functions.

    ``atf_path`` is the on-disk ATF sidecar for ``atf-ndcpwlf`` instances and
    ``None`` for ``igp-profile`` instances (whose ATFs are materialized);
    ``categories_path`` is the exact mirror of that situation.
    """

    instance: AnyTDBenchmarkInstance
    atfs: InstanceATFs
    instance_path: Path
    atf_path: Path | None
    categories_path: Path | None = None


def _load_td_instance_igp(
    source: Path,
    instance: AnyTDBenchmarkInstance,
    *,
    verify_sha256: bool,
) -> LoadedTDInstance:
    """igp-profile branch: load categories, materialize the canonical ATFs."""
    from mamut_routing_lib.td.igp import (
        compute_categories_sha256,
        load_instance_categories,
        materialize_instance_atfs,
    )

    td = instance.td
    categories_path = source.parent / td.categories_path
    categories = load_instance_categories(categories_path)
    if categories.num_customers != instance.num_customers:
        raise ATFFormatError(
            f"categories num_customers {categories.num_customers} does not match "
            f"{instance.num_customers}"
        )
    if categories.benchmark_name != instance.benchmark_name.value:
        raise ATFFormatError(
            f"categories benchmark_name {categories.benchmark_name!r} does not match "
            f"{instance.benchmark_name.value!r}"
        )
    if verify_sha256 and td.categories_sha256 is not None:
        digest = compute_categories_sha256(categories)
        if digest != td.categories_sha256:
            raise ATFFormatError(
                f"categories sha256 mismatch: computed {digest}, instance declares {td.categories_sha256}"
            )
    atfs = materialize_instance_atfs(instance, categories)
    if verify_sha256 and td.atf_sha256 is not None:
        digest = compute_atf_sha256(atfs)
        if digest != td.atf_sha256:
            raise ATFFormatError(
                f"materialized ATF sha256 mismatch: computed {digest}, instance declares {td.atf_sha256}"
            )
    return LoadedTDInstance(
        instance=instance,
        atfs=atfs,
        instance_path=source,
        atf_path=None,
        categories_path=categories_path,
    )


def load_td_instance(
    instance_path: str | Path,
    *,
    verify_sha256: bool = True,
    validate_complete: bool = True,
) -> LoadedTDInstance:
    """Load instance + sidecar, checking their mutual consistency.

    For ``igp-profile`` instances the ATFs are materialized deterministically
    from the td block and the categories sidecar; ``verify_sha256`` then
    covers both ``categories_sha256`` (cheap) and ``atf_sha256`` (a full
    canonical serialization of the materialized ATFs -- minutes at n=1000;
    pass ``verify_sha256=False`` in solver hot paths, materialization is
    deterministic either way).
    """
    source = Path(instance_path)
    instance = load_td_benchmark_instance(source)
    if isinstance(instance.td, TDIGPProfileRef):
        return _load_td_instance_igp(source, instance, verify_sha256=verify_sha256)
    atf_path = get_atf_path_for_instance(source, instance.td.atf_path)
    atfs = load_instance_atfs(atf_path, validate_complete=validate_complete)

    if atfs.instance_name != instance.instance_name:
        raise ATFFormatError(
            f"sidecar instance_name {atfs.instance_name!r} does not match {instance.instance_name!r}"
        )
    if atfs.num_customers != instance.num_customers:
        raise ATFFormatError(
            f"sidecar num_customers {atfs.num_customers} does not match {instance.num_customers}"
        )
    if (float(instance.horizon[0]), float(instance.horizon[1])) != atfs.horizon:
        raise ATFFormatError(
            f"sidecar horizon {atfs.horizon} does not match instance horizon {tuple(instance.horizon)}"
        )
    if verify_sha256 and instance.td.atf_sha256 is not None:
        digest = compute_atf_sha256(atfs)
        if digest != instance.td.atf_sha256:
            raise ATFFormatError(
                f"sidecar sha256 mismatch: computed {digest}, instance declares {instance.td.atf_sha256}"
            )
    return LoadedTDInstance(instance=instance, atfs=atfs, instance_path=source, atf_path=atf_path)

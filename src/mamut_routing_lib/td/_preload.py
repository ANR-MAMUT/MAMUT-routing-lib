"""Bind a caller's mutable TD materialization to the inputs at its descriptor path."""

from hashlib import sha256
from pathlib import Path
from struct import Struct, error as PackError, pack
from weakref import WeakValueDictionary

from mamut_routing_lib.sidecars import sidecar_base_dir
from mamut_routing_lib.td.artifacts import LoadedTDInstance, load_td_instance
from mamut_routing_lib.td.models import TDIGPProfileRef, TDRoadGraphRef

_verified: WeakValueDictionary = WeakValueDictionary()
_arc_header = Struct("=qqQQ")


def _source_revision(descriptor, path, atf_cache=None):
    td = descriptor.td
    base = sidecar_base_dir(path)
    if isinstance(td, TDIGPProfileRef):
        paths = [base / td.categories_path]
    elif isinstance(td, TDRoadGraphRef):
        paths = [base / td.graph.path, base / td.traffic.path]
    else:
        source = base / td.atf_path
        if not source.is_file() and atf_cache is not None and td.atf_sha256:
            source = Path(atf_cache) / source.name
        paths = [source]
    versions = []
    for source in paths:
        stat = source.stat()
        versions.append((source.resolve(), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return path.resolve(), sha256(descriptor.model_dump_json().encode()).digest(), tuple(versions)


def _arc_digest(atfs):
    # Use the numerical arrays directly. Rebuilding canonical ATF JSON would
    # serialize the entire graph even on a repeated check of a verified preload.
    digest = sha256()
    try:
        for (i, j), arc in atfs.arcs.items():
            digest.update(_arc_header.pack(i, j, len(arc.xs), len(arc.ys)))
            digest.update(pack(f"={len(arc.xs)}d", *arc.xs))
            digest.update(pack(f"={len(arc.ys)}d", *arc.ys))
    except (TypeError, OverflowError, PackError) as exc:
        raise ValueError("Invalid numerical operand in supplied TD arrival functions") from exc
    return digest.digest()


def load_verified_td(descriptor, path):
    before = _source_revision(descriptor, path)
    result = load_td_instance(path)
    if result.instance == descriptor and before == _source_revision(descriptor, path):
        _verified[id(result), before, _arc_digest(result.atfs)] = result
    return result


def require_td_preload(supplied: LoadedTDInstance, descriptor, path: Path):
    if Path(supplied.instance_path).resolve() != path.resolve() or supplied.instance != descriptor:
        raise ValueError(f"Supplied TD descriptor or path differs from {path}")
    held = supplied.atfs
    if (held.instance_name != descriptor.instance_name or held.benchmark_name != descriptor.benchmark_name.value
            or held.num_customers != descriptor.num_customers or tuple(held.horizon) != tuple(descriptor.horizon)):
        raise ValueError(f"Supplied TD materialization identifies a different instance at {path}")

    cache_dir = Path(supplied.atf_path).parent if supplied.atf_path is not None else None
    revision = _source_revision(descriptor, path, cache_dir)
    key = id(supplied), revision, _arc_digest(held)
    if _verified.get(key) is supplied:
        return

    from mamut_routing_lib.td.selected_arcs import load_td_instance_for_routes

    # Verify every supplied arc, including those absent from the submission.
    # A sparse IGP or road-graph preload only needs its supplied arcs rebuilt.
    td = descriptor.td
    generated = isinstance(td, (TDIGPProfileRef, TDRoadGraphRef))
    if generated and len(held.arcs) == descriptor.num_customers * (descriptor.num_customers + 1):
        published = load_td_instance(path)
    else:
        routes = [[v for v in edge if v != descriptor.depot] for edge in held.arcs] if generated else []
        published = load_td_instance_for_routes(path, routes, atf_dir=cache_dir, verify_sidecar_sha256=True)
    if any(published.atfs.arcs.get(edge) != arc for edge, arc in held.arcs.items()):
        raise ValueError(f"Supplied TD arrival functions differ from the materialization at {path}")
    if revision != _source_revision(descriptor, path, cache_dir):
        raise ValueError(f"TD source files changed during verification at {path}")
    _verified[key] = supplied

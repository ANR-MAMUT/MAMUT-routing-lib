"""Load a TD instance with only the arcs that a set of routes uses.

Large road-graph and igp-profile instances are expensive to materialize on the
complete graph. ``load_td_instance_for_routes`` materializes the arcs of the
given routes only, with the same materializers as the full load, so each arc is
bit-identical to its full-load value. ``route_arcs`` lists those arcs.
"""

from __future__ import annotations

from pathlib import Path

from mamut_routing_lib.td.artifacts import (
    ATFFormatError,
    InstanceATFs,
    LoadedTDInstance,
    get_atf_path_for_instance,
    load_instance_atfs,
    load_td_benchmark_instance,
)
from mamut_routing_lib.td.models import TDIGPProfileRef, TDRoadGraphRef

__all__ = ["load_td_instance_for_routes", "route_arcs"]


def route_arcs(routes: list[list[int]], depot: int = 0) -> set[tuple[int, int]]:
    """Every arc a route list uses, depot to depot."""
    arcs: set[tuple[int, int]] = set()
    for route in routes:
        previous = depot
        for vertex in route:
            arcs.add((previous, vertex))
            previous = vertex
        arcs.add((previous, depot))
    return arcs


def load_td_instance_for_routes(
    instance_path: str | Path,
    routes: list[list[int]],
    *,
    collection_root: str | Path | None = None,
    atf_dir: str | Path | None = None,
    verify_sidecar_sha256: bool = False,
) -> LoadedTDInstance:
    """Load a TD instance with the ATFs of the arcs ``routes`` use only.

    The arcs are bit-identical to the full load (same materializers), but the
    complete-graph ``atf_sha256`` pin cannot be checked on a partial set.
    ``verify_sidecar_sha256`` checks the pins of the sidecars actually read
    (ATF file, categories, road graph and traffic overlay). ``atf_dir`` is a
    fallback directory for ``atf-ndcpwlf`` sidecars that are pinned but not
    committed (Blauth2024 n >= 1000). Raises ``FileNotFoundError`` when a
    sidecar is missing.
    """
    source = Path(instance_path)
    instance = load_td_benchmark_instance(source)
    td = instance.td
    selected = route_arcs(routes, instance.depot)
    horizon = (float(instance.horizon[0]), float(instance.horizon[1]))

    if isinstance(td, TDIGPProfileRef):
        from mamut_routing_lib.td.igp import (
            compute_categories_sha256,
            load_instance_categories,
            materialize_selected_atfs_igp,
        )

        from mamut_routing_lib.sidecars import sidecar_base_dir

        categories_path = sidecar_base_dir(source, collection_root) / td.categories_path
        categories = load_instance_categories(categories_path)
        if verify_sidecar_sha256 and td.categories_sha256 is not None:
            if compute_categories_sha256(categories) != td.categories_sha256:
                raise ATFFormatError(f"categories sha256 mismatch for {source}")
        arcs = materialize_selected_atfs_igp(instance, categories, selected)
        return LoadedTDInstance(
            instance=instance,
            atfs=InstanceATFs(instance.instance_name, instance.benchmark_name.value, horizon, instance.num_customers, arcs),
            instance_path=source,
            atf_path=None,
            categories_path=categories_path,
        )

    if isinstance(td, TDRoadGraphRef):
        from mamut_routing_lib.sidecars import require_collection_root
        from mamut_routing_lib.td.roadgraph import (
            compute_road_graph_sha256,
            compute_traffic_overlay_sha256,
            load_instance_road_graph,
            load_traffic_overlay,
            materialize_selected_atfs_roadgraph,
        )

        root = require_collection_root(source, Path(collection_root) if collection_root is not None else None)
        road_graph_path = root / td.graph.path
        traffic_path = root / td.traffic.path
        road = load_instance_road_graph(road_graph_path)
        overlay = load_traffic_overlay(traffic_path)
        if verify_sidecar_sha256:
            if td.graph.sha256 is not None and compute_road_graph_sha256(road) != td.graph.sha256:
                raise ATFFormatError(f"road-graph sha256 mismatch for {source}")
            if td.traffic.sha256 is not None and compute_traffic_overlay_sha256(overlay) != td.traffic.sha256:
                raise ATFFormatError(f"traffic-overlay sha256 mismatch for {source}")
        arcs = materialize_selected_atfs_roadgraph(instance, road, overlay, selected)
        return LoadedTDInstance(
            instance=instance,
            atfs=InstanceATFs(instance.instance_name, instance.benchmark_name.value, horizon, instance.num_customers, arcs),
            instance_path=source,
            atf_path=None,
            road_graph_path=road_graph_path,
            traffic_path=traffic_path,
            collection_root=root,
        )

    atf_path = get_atf_path_for_instance(source, td.atf_path, collection_root=collection_root)
    if not atf_path.is_file() and atf_dir is not None:
        atf_path = Path(atf_dir) / Path(td.atf_path).name
    if not atf_path.is_file():
        raise FileNotFoundError(f"ATF sidecar not found: {atf_path}")
    atfs = load_instance_atfs(atf_path)
    if verify_sidecar_sha256 and td.atf_sha256 is not None:
        from mamut_routing_lib.td.artifacts import compute_atf_sha256

        if compute_atf_sha256(atfs) != td.atf_sha256:
            raise ATFFormatError(f"ATF sidecar sha256 mismatch for {source}")
    return LoadedTDInstance(instance=instance, atfs=atfs, instance_path=source, atf_path=atf_path)

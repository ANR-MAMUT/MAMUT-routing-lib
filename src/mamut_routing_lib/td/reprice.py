"""Re-price stored TD BKS under the current checker contract.

A change of the TD route fold (see ``td.checker.TD_CHECKER_CONTRACT``) moves
the float value of some stored BKS by a few ulps, and occasionally by more
(Rifki2020 under ``td-fold/1``). The store refuses a stored BKS whose cost no
longer matches the checker, so every stored TD BKS has to be re-priced once
after such a change. This module does it file by file, routes untouched:

- the instance is loaded with only the arcs its BKS routes use
  (``load_td_instance_for_routes``), which makes large road-graph and
  igp-profile instances cheap;
- the routes are priced by ``check_td_solution`` under the BKS objective;
- a file is rewritten only when a stored checker output changed: ``cost``,
  ``metadata.validated_cost`` (when present), ``metadata.route_durations`` and
  ``metadata.route_departure_times`` (stored route order);
- a moved ``cost`` is recorded in ``metadata.repriced``, and an optimality
  stamp gets ``proven_optimum`` set to the new cost plus a note; a stamped BKS
  that moves by more than ``OPTIMALITY_COST_TOLERANCE`` is refused, because the
  proof would no longer describe the stored solution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_lib.artifacts import load_bks, save_bks
from mamut_routing_lib.models import BenchmarkBKS
from mamut_routing_lib.td.artifacts import (
    ATFFormatError,
    InstanceATFs,
    LoadedTDInstance,
    get_atf_path_for_instance,
    load_instance_atfs,
    load_td_benchmark_instance,
)
from mamut_routing_lib.td.bks import OPTIMALITY_COST_TOLERANCE
from mamut_routing_lib.td.checker import TD_CHECKER_CONTRACT, check_td_solution
from mamut_routing_lib.td.models import TDIGPProfileRef, TDRoadGraphRef

#: Label written to ``metadata.repriced.checker``.
REPRICE_CHECKER_LABEL = f"mamut-routing-lib td checker, contract {TD_CHECKER_CONTRACT} (lib >= 0.12.0)"

#: Objectives stored by the TD BKS store.
TD_BKS_SUFFIXES = (".bks.Duration.json", ".bks.FleetCostDuration.json")


@dataclass
class RepriceResult:
    """Outcome for one BKS file.

    ``action`` is ``unchanged``, ``repriced`` (written), ``would-reprice``
    (dry run), ``skipped-missing-sidecar``, ``infeasible``, ``stamp-conflict``
    or ``error``.
    """

    bks_path: str
    action: str
    previous_cost: float | None = None
    new_cost: float | None = None
    cost_ulps: int | None = None
    changed_fields: list[str] = field(default_factory=list)
    stamped: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def instance_path_for_bks(bks_path: str | Path) -> Path:
    """``<dir>/<base>.bks.<Objective>.json`` -> ``<dir>/<base>.vrp.json``."""
    path = Path(bks_path)
    base, sep, _ = path.name.partition(".bks.")
    if not sep:
        raise ValueError(f"not a BKS file name: {path.name}")
    return path.with_name(f"{base}.vrp.json")


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

        categories_path = source.parent / td.categories_path
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

    atf_path = get_atf_path_for_instance(source, td.atf_path)
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


def _ulps(old: float, new: float) -> int:
    if old == new:
        return 0
    return max(1, round(abs(new - old) / math.ulp(max(abs(old), abs(new)))))


def _stamp_note(previous: float, new: float, date: str) -> str:
    return (
        f"Re-priced on {date} from {previous!r} to {new!r} by {REPRICE_CHECKER_LABEL} "
        "(exact vertex ready-time transforms and slope-one arc interpolation). Routes unchanged. "
        "The proof was obtained under the pre-0.12 fold (td-fold/1); dual_bound is the prover's value "
        "under that arithmetic. Re-certification under td-fold/2 is pending."
    )


def reprice_td_bks(
    bks_path: str | Path,
    *,
    dry_run: bool = False,
    collection_root: str | Path | None = None,
    atf_dir: str | Path | None = None,
    verify_sidecar_sha256: bool = False,
    note_date: str | None = None,
) -> RepriceResult:
    """Re-price one stored TD BKS file (module docstring). Never raises for data problems."""
    path = Path(bks_path)
    date = note_date or datetime.now(UTC).date().isoformat()
    try:
        bks = load_bks(path)
        loaded = load_td_instance_for_routes(
            instance_path_for_bks(path),
            bks.routes,
            collection_root=collection_root,
            atf_dir=atf_dir,
            verify_sidecar_sha256=verify_sidecar_sha256,
        )
    except FileNotFoundError as exc:
        return RepriceResult(str(path), "skipped-missing-sidecar", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - report every other load failure per file
        return RepriceResult(str(path), "error", detail=f"{type(exc).__name__}: {exc}")

    stamped = "optimality" in bks.metadata
    try:
        check = check_td_solution(loaded, bks.model_copy(update={"cost": None}), bks.objective_function)
    except Exception as exc:  # noqa: BLE001
        return RepriceResult(str(path), "error", previous_cost=bks.cost, stamped=stamped, detail=f"{type(exc).__name__}: {exc}")
    if not check.is_valid():
        return RepriceResult(
            str(path), "infeasible", previous_cost=bks.cost, stamped=stamped, detail=check.error_message
        )

    new_cost = check.routing_cost
    durations = [evaluation.duration for evaluation in check.route_evaluations]
    departures = [evaluation.departure_time for evaluation in check.route_evaluations]
    metadata = dict(bks.metadata)
    changed: list[str] = []
    if bks.cost != new_cost:
        changed.append("cost")
    if "validated_cost" in metadata and metadata["validated_cost"] != new_cost:
        changed.append("metadata.validated_cost")
        metadata["validated_cost"] = new_cost
    if metadata.get("route_durations") != durations:
        changed.append("metadata.route_durations")
        metadata["route_durations"] = durations
    if metadata.get("route_departure_times") != departures:
        changed.append("metadata.route_departure_times")
        metadata["route_departure_times"] = departures

    result = RepriceResult(
        str(path),
        "unchanged",
        previous_cost=bks.cost,
        new_cost=new_cost,
        cost_ulps=_ulps(bks.cost, new_cost) if bks.cost is not None else None,
        changed_fields=changed,
        stamped=stamped,
    )
    if not changed:
        return result

    if "cost" in changed:
        metadata["repriced"] = {
            "previous_cost": bks.cost,
            "checker": REPRICE_CHECKER_LABEL,
            "contract": TD_CHECKER_CONTRACT,
            "date": date,
        }
        if stamped:
            optimality = dict(metadata["optimality"])
            proven = optimality.get("proven_optimum")
            if proven is not None and abs(proven - new_cost) > OPTIMALITY_COST_TOLERANCE:
                result.action = "stamp-conflict"
                result.detail = f"stamped proven_optimum {proven!r} vs re-priced cost {new_cost!r}"
                return result
            if proven is not None:
                optimality["proven_optimum"] = new_cost
            note = _stamp_note(bks.cost, new_cost, date)
            optimality["note"] = f"{optimality['note']} {note}" if optimality.get("note") else note
            metadata["optimality"] = optimality

    if dry_run:
        result.action = "would-reprice"
        return result
    repriced = BenchmarkBKS(**{**bks.model_dump(mode="json"), "cost": new_cost, "metadata": metadata})
    save_bks(repriced, path)
    result.action = "repriced"
    return result


def iter_td_bks_files(paths: list[str | Path]) -> list[Path]:
    """TD BKS files under ``paths`` (files are taken as given), sorted, without duplicates."""
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.add(path)
            continue
        for suffix in TD_BKS_SUFFIXES:
            for candidate in path.rglob(f"*{suffix}"):
                if ".mamut-staging" in candidate.parts:
                    continue
                found.add(candidate)
    return sorted(found)

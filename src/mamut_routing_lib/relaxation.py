"""Relaxation twins: instance pairs where one problem relaxes the other.

Two twin conventions exist in the benchmark tree:

- **Time-dependent.** ``…/TDVRPTW/…/<name>.vrp.json`` and ``…/TDVRP/…/<name>.vrp.json``
  (same relative path otherwise) with identical td block, coordinates,
  demands, capacity, service times, fleet and horizon. The TDVRP twin drops
  the time windows (no waiting, no due dates), so with FIFO arrival-time
  functions any TDVRPTW route is feasible on it with a duration no larger.
- **Static collections.** ``<root>/VRPTW/<metric>/<city>/n=<N>/<base>/<base>[-tw-<set>].vrp.json``
  and ``<root>/CVRP/<metric>/<city>/n=<N>/<base>/<base>.vrp.json`` sharing the
  arc-cost source, demands, capacity and fleet: every VRPTW solution is a CVRP
  solution of the same cost.

A published BKS of the relaxation must therefore never be worse than the BKS
of its restriction under the same objective (cost, or ``(routes, cost)`` for
the hierarchical objective). ``check_relaxation_pair`` screens a pair on
stored values and, on request, prices the restriction's routes on the
relaxation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mamut_routing_lib.checker import exact_cost_value, get_objective_tuple
from mamut_routing_lib.enums import ObjectiveFunction

_TD_STRICT = "TDVRPTW"
_TD_RELAXED = "TDVRP"
_TD_SHARED_FIELDS = (
    "td",
    "coordinates",
    "demands",
    "vehicle_capacity",
    "service_times",
    "num_vehicles",
    "num_customers",
    "horizon",
    "depot",
    "fleet_fixed_cost",
)
_STATIC_SHARED_FIELDS = (
    "arc_costs_source",
    "arc_costs",
    "demands",
    "vehicle_capacity",
    "num_vehicles",
    "num_customers",
    "depot",
)


def relaxation_twin(instance_path: str | Path) -> Path | None:
    """The relaxation twin path of a TDVRPTW or collection VRPTW instance, or ``None``.

    Only the path convention is applied; the twin may not exist.
    """
    path = Path(instance_path)
    parts = list(path.parts)
    if parts.count(_TD_STRICT) == 1:
        parts[parts.index(_TD_STRICT)] = _TD_RELAXED
        return Path(*parts)
    if "VRPTW" in parts:
        index = len(parts) - 1 - parts[::-1].index("VRPTW")
        tail = parts[index + 1 :]
        # <metric>/<city>/n=<N>/<base>/<file>
        if len(tail) == 5 and tail[2].startswith("n="):
            base = tail[3]
            if tail[4] == f"{base}.vrp.json" or tail[4].startswith(f"{base}-tw-"):
                return Path(*parts[:index], "CVRP", *tail[:4], f"{base}.vrp.json")
    return None


def structural_relaxation_issues(strict_path: str | Path, relaxed_path: str | Path) -> list[str]:
    """Reasons the pair is not a relaxation (empty when it is)."""
    strict = json.loads(Path(strict_path).read_text(encoding="utf-8"))
    relaxed = json.loads(Path(relaxed_path).read_text(encoding="utf-8"))
    issues: list[str] = []
    if "td" in strict:
        fields = _TD_SHARED_FIELDS
        if "time_windows" in relaxed:
            issues.append("relaxed twin has time windows")
        # The TDVRP twin departs and returns within the horizon, so the
        # restriction's depot window must lie inside it. Customer windows may
        # reach past the horizon (Vu2020 has some): the return-by-horizon cut
        # binds on both sides, so they do not break the relaxation.
        horizon = strict.get("horizon")
        windows = strict.get("time_windows") or []
        depot = strict.get("depot", 0)
        if horizon and len(windows) > depot:
            earliest, latest = windows[depot]
            if earliest < horizon[0] or latest > horizon[1]:
                issues.append("the depot window leaves the horizon")
    else:
        fields = _STATIC_SHARED_FIELDS
        if "time_windows" in relaxed:
            issues.append("relaxed twin has time windows")
    for name in fields:
        if strict.get(name) != relaxed.get(name):
            issues.append(f"{name} differs")
    return issues


@dataclass
class RelaxationCheck:
    """Result of screening one twin pair (see ``check_relaxation_pair``)."""

    strict_bks: str
    relaxed_bks: str
    objective_function: str
    strict_cost: float | int | None
    relaxed_cost: float | int | None
    priced_cost: float | int | None = None
    inverted: bool = False
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bks_for(instance_path: Path, objective: str) -> Path:
    base = instance_path.name.removesuffix(".vrp.json")
    return instance_path.with_name(f"{base}.bks.{objective}.json")


def check_relaxation_pair(strict_bks_path: str | Path, *, priced: bool = False) -> RelaxationCheck | None:
    """Screen the twin pair of a restriction's BKS; ``None`` when it has no twin BKS.

    ``inverted`` is set when the relaxation's stored BKS is strictly worse
    than the restriction's (exact decimal costs, compared through
    ``get_objective_tuple``) or, with ``priced``, worse than the restriction's
    routes priced on the relaxation.
    """
    strict_bks = Path(strict_bks_path)
    base, _, rest = strict_bks.name.partition(".bks.")
    objective = rest.removesuffix(".json")
    strict_instance = strict_bks.with_name(f"{base}.vrp.json")
    relaxed_instance = relaxation_twin(strict_instance)
    if relaxed_instance is None or not relaxed_instance.is_file():
        return None
    relaxed_bks = _bks_for(relaxed_instance, objective)
    if not relaxed_bks.is_file():
        return None
    strict_payload = json.loads(strict_bks.read_text(encoding="utf-8"))
    relaxed_payload = json.loads(relaxed_bks.read_text(encoding="utf-8"))
    result = RelaxationCheck(
        strict_bks=str(strict_bks),
        relaxed_bks=str(relaxed_bks),
        objective_function=objective,
        strict_cost=strict_payload.get("cost"),
        relaxed_cost=relaxed_payload.get("cost"),
        issues=structural_relaxation_issues(strict_instance, relaxed_instance),
    )
    if result.issues or result.strict_cost is None or result.relaxed_cost is None:
        return result
    objective_function = ObjectiveFunction(objective)
    strict_routes = strict_payload["routes"]
    relaxed_key = get_objective_tuple(
        relaxed_payload["routes"], exact_cost_value(result.relaxed_cost), objective_function
    )
    if relaxed_key > get_objective_tuple(strict_routes, exact_cost_value(result.strict_cost), objective_function):
        result.inverted = True
    if priced:
        result.priced_cost = _price_on_relaxation(relaxed_instance, strict_routes, objective)
        if result.priced_cost is not None and relaxed_key > get_objective_tuple(
            strict_routes, exact_cost_value(result.priced_cost), objective_function
        ):
            result.inverted = True
    return result


def _price_on_relaxation(relaxed_instance: Path, routes: list[list[int]], objective: str) -> float | int | None:
    from mamut_routing_lib.models import BenchmarkSolution

    solution = BenchmarkSolution(instance_name="relaxation-twin", routes=routes)
    payload = json.loads(relaxed_instance.read_text(encoding="utf-8"))
    if "td" in payload:
        from mamut_routing_lib.td.checker import check_td_solution
        from mamut_routing_lib.td.reprice import load_td_instance_for_routes

        loaded = load_td_instance_for_routes(relaxed_instance, routes)
        check = check_td_solution(loaded, solution, ObjectiveFunction(objective))
    else:
        from mamut_routing_lib.artifacts import hydrate_collection_instance, load_benchmark_instance
        from mamut_routing_lib.checker import check_solution

        instance = hydrate_collection_instance(load_benchmark_instance(relaxed_instance), relaxed_instance)
        check = check_solution(instance, solution)
    return check.routing_cost if check.is_valid() else None

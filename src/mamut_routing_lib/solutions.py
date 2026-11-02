"""Solutions coming from outside the library: parsing, id normalisation, checking, submitting.

Formats read by :func:`parse_solution_text` / :func:`read_solution_file`:

- CVRPLIB ``.sol``: ``Route #k: a b c`` lines (``Route k:`` is accepted too)
  and an optional ``Cost <value>`` line. Costs may be written in exponent form
  (Java's ``Double.toString`` prints ``1.2345E7``; HGS-CVRP prints six
  significant digits, ``1.23457e+06``).
- JSON: ``{"routes": [[...]], "cost": ...}`` (a lib solution or BKS file) or a
  bare array of routes.
- LKH-3 ``MTSP_SOLUTION_FILE``: a ``<name>, Cost: <penalty>_<cost>`` header and
  one ``1 a b c 1 (#3)  Cost: <cost>`` line per vehicle, node ids 1-based with
  the depot as node 1.

Stop ids are then mapped onto library ids (depot 0, customers 1..n) by
:func:`normalize_routes`: a file is either understood exactly or rejected,
never silently reinterpreted into a different instance.

``check_routes`` prices a route set with the reference checker (static or
time-dependent); ``submit_bks`` stores it through the BKS stores, which only
ever replace a BKS with a strictly better, re-validated solution.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType

__all__ = [
    "IdConvention",
    "ParsedSolution",
    "SolutionFormatError",
    "check_routes",
    "is_time_dependent",
    "normalize_routes",
    "parse_lkh_mtsp_solution",
    "parse_solution_text",
    "preview_bks_update",
    "read_solution_file",
    "read_solution_text",
    "submit_bks",
    "write_cvrplib_solution",
]

IdConvention = Literal["auto", "one-based", "zero-based"]

_ROUTE_LINE = re.compile(r"^\s*Route\s*#?\s*\d+\s*:\s*(.*)$", re.IGNORECASE)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_COST_LINE = re.compile(rf"^\s*cost\s*[:=]?\s*({_NUMBER})\s*$", re.IGNORECASE)
_LKH_HEADER = re.compile(rf"Cost:\s*({_NUMBER})_({_NUMBER})")
_LKH_ROUTE = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*\(#\s*(\d+)\)\s*Cost:\s*(" + _NUMBER + r")\s*$")


class SolutionFormatError(ValueError):
    """A solution text that cannot be read, or whose stop ids do not fit the instance."""


@dataclass(frozen=True)
class ParsedSolution:
    """Routes in library ids, the declared cost (if any) and the detected format."""

    routes: list[list[int]]
    declared_cost: float | None
    format: str
    penalty: float | None = None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_solution_text(text: str, *, filename: str = "") -> tuple[list[list[int]], float | None]:
    """Routes (raw stop ids) and declared cost from a CVRPLIB ``.sol`` or a JSON solution.

    Accepts ``Route #k: a b c`` lines with an optional ``Cost <value>`` line, or
    JSON as either ``{"routes": [[...]]}`` or a bare array of routes. The stop ids
    are returned as written; :func:`normalize_routes` maps them onto the instance.
    """
    stripped = text.strip()
    if not stripped:
        raise SolutionFormatError("The solution is empty.")

    looks_json = stripped[0] in "[{" or filename.lower().endswith(".json")
    if looks_json:
        try:
            payload = json.loads(stripped)
        except ValueError as error:
            raise SolutionFormatError(f"Not valid JSON: {error}") from error
        raw_routes = payload if isinstance(payload, list) else payload.get("routes")
        if not isinstance(raw_routes, list):
            raise SolutionFormatError("JSON solution has no 'routes' array.")
        declared = None if isinstance(payload, list) else payload.get("cost")
        routes = []
        for route in raw_routes:
            if not isinstance(route, list):
                raise SolutionFormatError("Every entry of 'routes' must be a list of stops.")
            try:
                routes.append([int(stop) for stop in route])
            except (TypeError, ValueError) as error:
                raise SolutionFormatError(f"Route stops must be integers: {error}") from error
        return routes, (float(declared) if declared is not None else None)

    routes = []
    declared_cost: float | None = None
    for line in stripped.splitlines():
        match = _ROUTE_LINE.match(line)
        if match:
            try:
                routes.append([int(part) for part in match.group(1).split()])
            except ValueError as error:
                raise SolutionFormatError(f"Unreadable stop id in '{line.strip()}': {error}") from error
            continue
        cost_match = _COST_LINE.match(line)
        if cost_match:
            declared_cost = float(cost_match.group(1))
    if not routes:
        raise SolutionFormatError(
            "No 'Route #k: ...' lines found. Provide a CVRPLIB .sol file or JSON routes."
        )
    return routes, declared_cost


def parse_lkh_mtsp_solution(text: str) -> tuple[list[list[int]], float | None, float | None]:
    """Routes (1-based node ids, depot node 1 removed), penalty and cost of an LKH-3 MTSP solution file.

    LKH-3 writes each vehicle as ``1 a b c 1 (#3)  Cost: X`` and empty vehicles as
    ``1 1 (#0)  Cost: 0`` (with ``MTSP_MIN_SIZE = 0``); empty routes are dropped. A
    non-zero penalty means LKH-3 found no feasible solution.
    """
    lines = text.strip().splitlines()
    if not lines:
        raise SolutionFormatError("The solution is empty.")
    penalty = cost = None
    header = _LKH_HEADER.search(lines[0])
    if header:
        penalty, cost = float(header.group(1)), float(header.group(2))
    depot_id: int | None = None
    routes: list[list[int]] = []
    for line in lines[1:]:
        match = _LKH_ROUTE.match(line)
        if not match:
            continue
        nodes = [int(part) for part in match.group(1).split()]
        if depot_id is None:
            depot_id = nodes[0]
        customers = [node for node in nodes if node != depot_id]
        if customers:
            routes.append(customers)
    if depot_id is None:
        raise SolutionFormatError("No LKH-3 tour lines ('1 a b 1 (#k)  Cost: x') found.")
    return routes, penalty, cost


def _looks_like_lkh(text: str) -> bool:
    head = text.lstrip()[:400]
    return "salesmen are:" in head or bool(_LKH_HEADER.search(head.splitlines()[0] if head else ""))


def normalize_routes(
    routes: list[list[int]],
    num_customers: int,
    *,
    depot: int = 0,
    convention: IdConvention = "auto",
) -> list[list[int]]:
    """Map an external stop-id convention onto library ids (depot 0, customers 1..n).

    External files disagree on whether customers are numbered 1..n (CVRPLIB, the
    common case) or 0..n-1, and some include the depot at the route ends. With
    ``convention="auto"`` the two are told apart *before* any depot stripping, because
    a literal 0 is a depot marker under one and a customer under the other: only a
    file that covers 0..n-1 exactly once, zero included, is read as 0-based;
    everything else is read as 1..n with 0 meaning the depot.
    """
    if depot != 0:
        raise SolutionFormatError(
            f"Solution files number customers around a depot at index 0; this instance has its depot at "
            f"{depot}, so external stop ids cannot be mapped safely."
        )
    raw_stops = [stop for route in routes for stop in route]
    if not raw_stops:
        raise SolutionFormatError("The solution contains no customer stops.")

    if convention == "zero-based":
        zero_based = True
    elif convention == "one-based":
        zero_based = False
    else:
        zero_based = (
            0 in raw_stops
            and len(raw_stops) == len(set(raw_stops))
            and set(raw_stops) == set(range(num_customers))
        )
    if zero_based:
        normalized = [[stop + 1 for stop in route] for route in routes]
    else:
        # 0 is the depot written at the route boundaries: drop it, ids are 1..n.
        normalized = [[stop for stop in route if stop != 0] for route in routes]

    flat = [stop for route in normalized if route for stop in route]
    if not flat:
        raise SolutionFormatError("The solution contains no customer stops.")
    low, high = min(flat), max(flat)
    if low < 1 or high > num_customers:
        raise SolutionFormatError(
            f"Stop ids run {min(raw_stops)}..{max(raw_stops)}, which fits neither "
            f"1..{num_customers} nor 0..{num_customers - 1}; this solution is not "
            f"for an instance with {num_customers} customers."
        )

    seen = Counter(flat)
    duplicates = sorted(stop for stop, count in seen.items() if count > 1)
    if duplicates:
        raise SolutionFormatError(
            f"Customer(s) {', '.join(str(value) for value in duplicates[:10])} visited more than once."
        )
    missing = sorted(set(range(1, num_customers + 1)) - set(flat))
    if missing:
        raise SolutionFormatError(
            f"{len(missing)} customer(s) never visited (first missing: "
            f"{', '.join(str(value) for value in missing[:10])})."
        )
    return [route for route in normalized if route]


def read_solution_text(
    text: str,
    num_customers: int,
    *,
    filename: str = "",
    depot: int = 0,
    convention: IdConvention = "auto",
) -> ParsedSolution:
    """Parse any supported format and normalise its ids onto an instance with ``num_customers``."""
    if not filename.lower().endswith(".json") and _looks_like_lkh(text):
        routes, penalty, cost = parse_lkh_mtsp_solution(text)
        if penalty:
            raise SolutionFormatError(f"LKH-3 reports a penalty of {penalty:g}: the solution is infeasible.")
        # LKH node k is library node k-1 (the depot is node 1).
        library = normalize_routes([[node - 1 for node in route] for route in routes], num_customers, depot=depot, convention="one-based")
        return ParsedSolution(library, cost, "lkh-mtsp", penalty)
    routes, declared = parse_solution_text(text, filename=filename)
    library = normalize_routes(routes, num_customers, depot=depot, convention=convention)
    fmt = "json" if text.lstrip()[:1] in "[{" or filename.lower().endswith(".json") else "cvrplib-sol"
    return ParsedSolution(library, declared, fmt)


def read_solution_file(path: str | Path, instance: Any, *, convention: IdConvention = "auto") -> ParsedSolution:
    """Read a solution file for ``instance`` (any supported format), ids normalised."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SolutionFormatError(f"{target} is not a text file: {error}") from error
    return read_solution_text(
        text,
        int(instance.num_customers),
        filename=target.name,
        depot=int(getattr(instance, "depot", 0) or 0),
        convention=convention,
    )


def write_cvrplib_solution(routes: list[list[int]], cost: int | float | None = None) -> str:
    """CVRPLIB ``.sol`` text of library routes (customers 1..n), ``Cost`` line when given."""
    lines = [f"Route #{index}: {' '.join(str(stop) for stop in route)}" for index, route in enumerate(routes, start=1)]
    if cost is not None:
        value = float(cost)
        lines.append(f"Cost {int(value)}" if value.is_integer() else f"Cost {value!r}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def is_time_dependent(instance: Any) -> bool:
    from mamut_routing_lib.artifacts import instance_problem_type

    return instance_problem_type(instance) in (ProblemType.TDVRP, ProblemType.TDVRPTW)


def check_routes(
    instance: Any,
    instance_path: str | Path,
    routes: list[list[int]],
    objective_function: ObjectiveFunction | None = None,
    *,
    declared_cost: int | float | None = None,
) -> Any:
    """Check ``routes`` (library ids, depot implicit) on the instance at ``instance_path``.

    Returns a ``SolutionCheckResult`` (static) or ``TDSolutionCheckResult`` (TD); both expose
    ``status``, ``routing_cost``, ``num_routes``, ``error_message`` and ``is_valid()``. A
    ``declared_cost`` makes the checker refuse any deviation (``objective_value_mismatch``).
    """
    from mamut_routing_lib.artifacts import get_instance_identifier
    from mamut_routing_lib.models import BenchmarkSolution

    path = Path(instance_path)
    try:
        candidate = BenchmarkSolution(
            instance_name=get_instance_identifier(instance),
            routes=[[int(stop) for stop in route] for route in routes],
            cost=declared_cost,
            metadata={},
        )
    except ValueError as error:  # pydantic: a route visits a customer twice, or a bad id
        return _invalid(instance, error)
    if is_time_dependent(instance):
        from mamut_routing_lib.td import check_td_solution, load_td_instance

        objective = objective_function or ObjectiveFunction.DURATION
        loaded = load_td_instance(path)
        return check_td_solution(loaded, candidate, objective_function=objective)

    from mamut_routing_lib.artifacts import hydrate_collection_instance
    from mamut_routing_lib.checker import check_solution

    checkable = hydrate_collection_instance(instance, path)
    return check_solution(checkable, candidate)


def _invalid(instance: Any, error: Exception) -> Any:
    """The checker's rejection for a route set the solution model refuses to build."""
    from mamut_routing_lib.checker import SolutionCheckResult, SolutionCheckStatus

    message = str(error)
    if "elementary" in message:
        status = SolutionCheckStatus.CUSTOMER_SERVED_MULTIPLE_TIMES
        message = "a route visits a customer more than once"
    else:
        status = SolutionCheckStatus.INVALID_CUSTOMER_INDEX
    if is_time_dependent(instance):
        from mamut_routing_lib.td.checker import TDSolutionCheckResult

        return TDSolutionCheckResult.make_invalid(status, message)
    return SolutionCheckResult.make_invalid(status, message)


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


def submit_bks(
    instance_path: str | Path,
    routes: list[list[int]],
    *,
    authors: str,
    objective_function: ObjectiveFunction | None = None,
    metadata: Mapping[str, Any] | None = None,
    instance: Any = None,
) -> Any:
    """Store ``routes`` as the BKS of the instance if they strictly improve it.

    Static instances go through ``bks.save_solution_as_bks_if_improved`` (exact decimal
    comparison), TD instances through ``td.save_td_solution_as_bks_if_improved``. The
    objective defaults to the family's published one (``objectives.base_objective``).
    Returns the store's ``BKSUpdateResult``; raises ``ValueError`` for an invalid solution
    or empty ``authors``.
    """
    from mamut_routing_lib.artifacts import get_instance_identifier, load_benchmark_instance
    from mamut_routing_lib.models import BenchmarkSolution
    from mamut_routing_lib.objectives import base_objective

    if not isinstance(authors, str) or not authors.strip():
        raise ValueError("authors must be a non-empty string (who found the solution)")
    path = Path(instance_path)
    loaded = instance if instance is not None else load_benchmark_instance(path)
    objective = objective_function or base_objective(loaded)
    solution = BenchmarkSolution(
        instance_name=get_instance_identifier(loaded),
        routes=[[int(stop) for stop in route] for route in routes],
        cost=None,
        metadata={},
    )
    if is_time_dependent(loaded):
        from mamut_routing_lib.td import load_td_instance, save_td_solution_as_bks_if_improved

        return save_td_solution_as_bks_if_improved(
            load_td_instance(path),
            solution,
            authors=authors,
            metadata=dict(metadata or {}),
            objective_function=objective,
        )
    from mamut_routing_lib.bks import save_solution_as_bks_if_improved

    return save_solution_as_bks_if_improved(path, objective, solution, authors=authors, metadata=dict(metadata or {}))


def preview_bks_update(
    instance_path: str | Path,
    routes: list[list[int]],
    *,
    objective_function: ObjectiveFunction | None = None,
    instance: Any = None,
) -> dict[str, Any]:
    """What ``submit_bks`` would do, without writing: ``action`` is ``would_create``,
    ``would_replace``, ``would_keep`` (``tie`` when equal) or ``invalid``.

    Uses the stores' own comparisons: exact decimal costs for static instances, the TD
    checker's costs for time-dependent ones.
    """
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_benchmark_instance, load_bks
    from mamut_routing_lib.checker import get_objective_tuple
    from mamut_routing_lib.objectives import base_objective

    path = Path(instance_path)
    loaded = instance if instance is not None else load_benchmark_instance(path)
    objective = objective_function or base_objective(loaded)
    check = check_routes(loaded, path, routes, objective)
    bks_path = get_bks_path_for_instance(path, objective)
    summary: dict[str, Any] = {
        "objective_function": objective.value,
        "path": str(bks_path),
        "candidate_cost": check.routing_cost,
        "candidate_num_routes": check.num_routes,
        "existing_cost": None,
        "existing_num_routes": None,
        "tie": False,
    }
    if not check.is_valid():
        return {**summary, "action": "invalid", "error": check.error_message or str(check.status)}
    if not bks_path.exists():
        return {**summary, "action": "would_create"}
    existing = load_bks(bks_path)
    summary["existing_cost"] = existing.cost
    summary["existing_num_routes"] = existing.num_routes
    if is_time_dependent(loaded):
        from mamut_routing_lib.checker import is_better_solution

        candidate_cost = float(check.routing_cost)
        existing_cost = float(existing.cost) if existing.cost is not None else float("inf")
        better = is_better_solution(routes, candidate_cost, existing.routes, existing_cost, objective)
        tie = get_objective_tuple(routes, candidate_cost, objective) == get_objective_tuple(
            existing.routes, existing_cost, objective
        )
    else:
        from mamut_routing_lib.artifacts import hydrate_collection_instance
        from mamut_routing_lib.checker import compute_exact_solution_cost, is_better_exact

        checkable = hydrate_collection_instance(loaded, path)
        candidate_exact = compute_exact_solution_cost(checkable, routes)
        existing_exact = compute_exact_solution_cost(checkable, existing.routes)
        better = is_better_exact(routes, candidate_exact, existing.routes, existing_exact, objective)
        tie = get_objective_tuple(routes, candidate_exact, objective) == get_objective_tuple(
            existing.routes, existing_exact, objective
        )
    if better:
        return {**summary, "action": "would_replace"}
    return {**summary, "action": "would_keep", "tie": bool(tie)}

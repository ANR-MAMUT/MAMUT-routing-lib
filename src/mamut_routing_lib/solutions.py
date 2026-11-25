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

Reading a large distance matrix, or a time-dependent instance's travel-time
functions, is the expensive part of a check. ``load_checkable`` does it once;
``check_routes``, ``preview_bks_update`` and ``submit_bks`` take the result as
``checkable``, so a caller that checks and then stores reads the data one time.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Mapping
from weakref import WeakValueDictionary

from pydantic import TypeAdapter

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.json_utils import strict_json_loads
from mamut_routing_lib.models import BenchmarkBKS, reject_pre_013_optimality_claim
from mamut_routing_lib.exact_types import ExactBackend, ExactRouteError

__all__ = [
    "STORE_METADATA_KEYS",
    "IdConvention",
    "ParsedSolution",
    "SolutionFormatError",
    "check_routes",
    "is_time_dependent",
    "load_checkable",
    "normalize_routes",
    "parse_lkh_mtsp_solution",
    "parse_solution_text",
    "preview_bks_update",
    "read_solution_file",
    "read_solution_text",
    "resolve_convention",
    "solver_submission_metadata",
    "submit_bks",
    "write_cvrplib_solution",
]

IdConvention = Literal["auto", "one-based", "zero-based"]

#: BKS metadata that a submitter cannot supply: what the stores write (``authors`` comes from the
#: ``authors`` argument), what the checker tools write (``optimality``, ``repriced``, the route
#: timing fields), and the statements about validation that only a maintainer's tooling makes.
STORE_METADATA_KEYS = frozenset(
    {
        "authors",
        "date",
        "validated_cost",
        "validated_num_routes",
        "route_durations",
        "route_departure_times",
        "optimality",
        "repriced",
        "validated_by",
        "checker",
        "checker_commit",
        "checker_contract",
        "standard",
        "exact_evaluation",
        "verification_report",
    }
)

_ROUTE_LINE = re.compile(r"^\s*Route\s*#?\s*\d+\s*:\s*(.*)$", re.IGNORECASE)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_COST_LINE = re.compile(r"^\s*cost\b\s*[:=]?\s*(.*?)\s*$", re.IGNORECASE)
_LKH_HEADER = re.compile(r"Cost:\s*([^\s_]+)_([^\s]+)", re.IGNORECASE)
_LKH_ROUTE = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*\(#\s*(\d+)\)\s*Cost:\s*(.*?)\s*$", re.IGNORECASE)


class SolutionFormatError(ValueError):
    """A solution text that cannot be read, or whose stop ids do not fit the instance."""


@dataclass(frozen=True)
class ParsedSolution:
    """Routes in library ids, the declared cost (if any) and the detected format."""

    routes: list[list[int]]
    declared_cost: float | None
    format: str
    penalty: float | None = None
    #: How the file's stop ids were read: ``one-based`` (customers 1..n, 0 = depot) or ``zero-based``.
    ids: str = "one-based"
    #: The ``metadata`` object of a JSON solution file (``None`` for other formats).
    metadata: dict[str, Any] | None = None
    marked_input: BenchmarkBKS | None = None
    reported_cost_token: str | None = None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


# A route set has at most two arcs per customer. Its integer weights cannot
# exceed the largest binary64 operand divided by the smallest positive one.
# Larger written exponents remain compact Decimal diagnostics.
_MAX_REPORT_EXPONENT = Decimal(2 * sys.maxsize * int(sys.float_info.max) * math.ulp(0.0).as_integer_ratio()[1]).adjusted()


_ReportMode = bool | Literal["text"]


class _JsonNumberToken(str):
    """Distinguish JSON numbers from quoted strings while retaining their original spelling."""


def _reported_number(token: str | int | float | Decimal, exact: _ReportMode) -> int | float | Decimal | str:
    if not exact:
        return _finite_cost(token)
    if isinstance(token, str) and not re.fullmatch(_NUMBER, token):
        raise SolutionFormatError(f"Expected a finite numerical cost, got {token!r}.")
    try:
        number = Decimal(token)
    except InvalidOperation as error:
        raise SolutionFormatError("Unrepresentable decimal cost.") from error
    if not number.is_finite():
        raise SolutionFormatError("Expected a finite numerical cost.")
    if exact == "text":
        return str(token)
    if number.is_zero():
        return 0
    if number.adjusted() <= _MAX_REPORT_EXPONENT and number == number.to_integral_value():
        return int(number)
    return number


def _report_json_integer(token: str) -> int | Decimal:
    # Route identifiers are integers. A vastly larger literal can only be
    # useful as a cost diagnostic, retained at its written precision.
    return int(Decimal(token)) if len(token.lstrip("-")) <= _MAX_REPORT_EXPONENT + 1 else Decimal(token)


def parse_solution_text(
    text: str, *, filename: str = "", exact_report: _ReportMode = False,
) -> tuple[list[list[int]], int | float | Decimal | str | None]:
    """Routes (raw stop ids) and declared cost from a CVRPLIB ``.sol`` or a JSON solution.

    Accepts ``Route #k: a b c`` lines with an optional ``Cost <value>`` line, or
    JSON as either ``{"routes": [[...]]}`` or a bare array of routes. The stop ids
    are returned as written; :func:`normalize_routes` maps them onto the instance.
    With ``exact_report="text"`` the declared cost retains its numeric token, including trailing zeros and exponents.
    """
    stripped = text.strip()
    if not stripped:
        raise SolutionFormatError("The solution is empty.")

    if stripped[0] in "[{" or filename.lower().endswith(".json"):
        routes, declared, _, _ = _parse_json_solution(stripped, exact_report=exact_report)
        return routes, declared

    routes = []
    declared_cost = None
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
            declared_cost = _reported_number(cost_match.group(1), exact_report)
    if not routes and not _COST_LINE.fullmatch(stripped):
        raise SolutionFormatError(
            "No 'Route #k: ...' lines found. Provide a CVRPLIB .sol file or JSON routes."
        )
    return routes, declared_cost


def _finite_cost(value: int | float | str | Decimal) -> float:
    if isinstance(value, str) and not re.fullmatch(_NUMBER, value):
        raise SolutionFormatError(f"Expected a finite numerical cost, got {value!r}.")
    try:
        cost = float(value)
    except (ValueError, OverflowError) as exc:
        raise SolutionFormatError("Cost exceeds the finite binary64 range.") from exc
    if not math.isfinite(cost):
        raise SolutionFormatError("Cost exceeds the finite binary64 range.")
    return cost


def _parse_json_solution(
    text: str, *, exact_report: _ReportMode = False,
) -> tuple[list[list[int]], int | float | Decimal | str | None, dict | None, BenchmarkBKS | None]:
    try:
        if exact_report == "text":
            payload = strict_json_loads(text, parse_float=_JsonNumberToken, parse_int=_JsonNumberToken)
        elif exact_report:
            payload = strict_json_loads(text, parse_float=Decimal, parse_int=_report_json_integer)
        else:
            payload = strict_json_loads(text)
    except (ValueError, InvalidOperation) as exc:
        raise SolutionFormatError(f"Not valid JSON: {exc}") from exc
    declared = metadata = None
    if isinstance(payload, list):
        routes = payload
    elif isinstance(payload, dict):
        routes, declared = payload.get("routes"), payload.get("cost")
        metadata = payload.get("metadata")
    else:
        raise SolutionFormatError(
            f"JSON solution must be an object with a 'routes' array or an array of routes, not {text[:40]!r}."
        )
    if not isinstance(routes, list):
        raise SolutionFormatError("JSON solution has no 'routes' array.")
    for route in routes:
        if not isinstance(route, list):
            raise SolutionFormatError("Every entry of 'routes' must be a list of stops.")
        if not route:
            raise SolutionFormatError("JSON routes cannot contain an empty route.")
        if exact_report == "text":
            for index, stop in enumerate(route):
                if isinstance(stop, _JsonNumberToken) and re.fullmatch(r"-?[0-9]+", stop):
                    route[index] = _report_json_integer(stop)
        if any(type(stop) is not int for stop in route):
            raise SolutionFormatError("Route stops must be integers.")
    if declared is not None and type(declared) not in (int, float, Decimal, _JsonNumberToken):
        raise SolutionFormatError(f"JSON 'cost' must be a number, got {declared!r}.")
    cost = None if declared is None else _reported_number(declared, exact_report)
    metadata = metadata if isinstance(metadata, dict) else None
    marked = None
    if metadata is not None and ("exact_evaluation" in metadata or "standard" in metadata):
        if exact_report:
            # Marked BKS fields and metadata retain their published JSON types.
            _, _, metadata, marked = _parse_json_solution(text)
            return routes, cost, metadata, marked
        if "cost" not in payload:
            raise SolutionFormatError("An exact BKS must explicitly store its cost projection (number or null).")
        reject_pre_013_optimality_claim(payload)
        try:
            marked = BenchmarkBKS.model_validate(payload)
        except ValueError as error:
            raise SolutionFormatError(f"Invalid marked BKS: {error}") from error
    return routes, cost, metadata, marked


def parse_lkh_mtsp_solution(
    text: str, *, exact_report: _ReportMode = False,
) -> tuple[list[list[int]], int | float | Decimal | None, int | float | Decimal | str | None]:
    """Routes (1-based node ids, depot node 1 removed), penalty and cost of an LKH-3 MTSP solution file.

    LKH-3 writes each vehicle as ``1 a b c 1 (#3)  Cost: X`` and empty vehicles as
    ``1 1 (#0)  Cost: 0`` (with ``MTSP_MIN_SIZE = 0``); empty routes are dropped. A
    non-zero penalty means LKH-3 found no feasible solution. With ``exact_report="text"`` the cost is the original token and the penalty is numeric.
    """
    lines = text.strip().splitlines()
    if not lines:
        raise SolutionFormatError("The solution is empty.")
    penalty = cost = None
    header = _LKH_HEADER.search(lines[0])
    if header:
        penalty = _reported_number(header.group(1), bool(exact_report))
        cost = _reported_number(header.group(2), exact_report)
    elif re.search(r"Cost:", lines[0], re.IGNORECASE):
        raise SolutionFormatError("Malformed LKH-3 penalty and cost header.")
    depot_id: int | None = None
    routes: list[list[int]] = []
    for line in lines[1:]:
        match = _LKH_ROUTE.match(line)
        if not match:
            continue
        _reported_number(match.group(3), exact_report)
        try:
            nodes = [int(part) for part in match.group(1).split()]
        except ValueError as exc:
            raise SolutionFormatError("Unreadable LKH-3 stop id.") from exc
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


def resolve_convention(routes: list[list[int]], num_customers: int, convention: IdConvention = "auto") -> str:
    """``zero-based`` or ``one-based``: what ``convention="auto"`` reads these stop ids as."""
    if convention in ("zero-based", "one-based"):
        return convention
    if convention != "auto":
        raise SolutionFormatError(f"Invalid id_convention: {convention!r}")
    raw_stops = [stop for route in routes for stop in route]
    covers_zero_to_n_minus_1 = (
        0 in raw_stops and len(raw_stops) == len(set(raw_stops)) and set(raw_stops) == set(range(num_customers))
    )
    return "zero-based" if covers_zero_to_n_minus_1 else "one-based"


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
    everything else is read as 1..n with 0 meaning the depot. Under 1..n a 0 inside
    a route is a return to the depot: the route is split there into two trips (the
    cost of the stops as written), never merged across it.
    """
    if depot != 0:
        raise SolutionFormatError(
            f"Solution files number customers around a depot at index 0; this instance has its depot at "
            f"{depot}, so external stop ids cannot be mapped safely."
        )
    raw_stops = [stop for route in routes for stop in route]
    if not routes and num_customers == 0:
        return []
    if not raw_stops:
        raise SolutionFormatError("The solution contains no customer stops.")

    zero_based = resolve_convention(routes, num_customers, convention) == "zero-based"
    if zero_based:
        normalized = [[stop + 1 for stop in route] for route in routes]
    else:
        # 0 is the depot: dropped at the route ends, a split into two trips inside a route.
        normalized = []
        for route in routes:
            trip: list[int] = []
            for stop in route:
                if stop == 0:
                    normalized.append(trip)
                    trip = []
                else:
                    trip.append(stop)
            normalized.append(trip)

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
    preserve_solver_report: bool = False,
) -> ParsedSolution:
    """Parse any supported format and normalise its ids onto an instance with ``num_customers``."""
    stripped = text.strip()
    token = None
    report_mode = "text" if preserve_solver_report else False
    if filename.lower().endswith(".json") or stripped.startswith(("[", "{")):
        routes, declared, metadata, marked = _parse_json_solution(text)
        if marked is not None:
            if convention == "zero-based":
                raise SolutionFormatError("Marked BKS routes use one-based customer identifiers.")
            return ParsedSolution(routes, declared, "json", metadata=metadata, marked_input=marked)
        fmt = "json"
    elif _looks_like_lkh(text):
        routes, penalty, cost = parse_lkh_mtsp_solution(text, exact_report=report_mode)
        if penalty:
            raise SolutionFormatError(f"LKH-3 reports a penalty of {penalty}: the solution is infeasible.")
        # LKH node k is library node k-1 (the depot is node 1).
        library = normalize_routes([[node - 1 for node in route] for route in routes], num_customers, depot=depot, convention="one-based")
        token = cost if isinstance(cost, str) else None
        display = float(cost) if cost is not None else None
        return ParsedSolution(library, display if display is not None and math.isfinite(display) else None,
                              "lkh-mtsp", penalty, reported_cost_token=token)
    else:
        routes, declared = ([], None) if not stripped and num_customers == 0 else parse_solution_text(
            text, filename=filename, exact_report=report_mode,
        )
        if isinstance(declared, str):
            token = declared
            number = float(declared)
            declared = number if math.isfinite(number) else None
        fmt, metadata = "cvrplib-sol", None
    ids = resolve_convention(routes, num_customers, convention)
    library = normalize_routes(routes, num_customers, depot=depot, convention=ids)  # type: ignore[arg-type]
    return ParsedSolution(library, declared, fmt, ids=ids, metadata=metadata, reported_cost_token=token)


def read_solution_file(
    path: str | Path, instance: Any, *, convention: IdConvention = "auto", preserve_solver_report: bool = False,
) -> ParsedSolution:
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
        preserve_solver_report=preserve_solver_report,
    )


def solver_submission_metadata(parsed: ParsedSolution, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Carry a solver report into the submission and check declarations against its source."""
    from mamut_routing_lib.solvers.report import checked_solver_metadata, make_solver_report
    from mamut_routing_lib.solvers.run import BKS_PROVENANCE_KEYS

    extra = extra or {}
    metadata = {}
    source = parsed.metadata or {}
    entries = [source]
    if isinstance(source.get("bks_metadata"), dict):
        entries.append(source["bks_metadata"])
    protected = {"solver_report", "solver_cost"}
    if any("solver_report" in entry for entry in entries):
        protected.update(("solver", "solver_version", "method"))
    for entry in entries:
        for key in BKS_PROVENANCE_KEYS:
            if key not in entry:
                continue
            if key in protected and key in metadata and entry[key] != metadata[key]:
                raise ValueError(f"conflicting {key} in the solution file's run metadata")
            metadata[key] = entry[key]
    for key in protected:
        if key in metadata and key in extra and metadata[key] != extra[key]:
            raise ValueError(f"conflicting {key} in supplied metadata and the solution file")
    metadata.update(extra)
    token = parsed.reported_cost_token
    if token is not None:
        known_objective = "total-arc-cost" if parsed.format == "lkh-mtsp" else "unspecified"
        if "solver_report" not in metadata:
            encoding = "integer" if token.lstrip("+-").isdigit() else "decimal-text"
            metadata["solver_report"] = make_solver_report(known_objective, token, encoding=encoding)
        else:
            declared = metadata["solver_report"]
            if not isinstance(declared, dict) or declared.get("value") != token:
                raise ValueError("solver_report must preserve the solution file's original Cost token")
            if known_objective != "unspecified" and declared.get("objective") != known_objective:
                raise ValueError("solver_report objective conflicts with the LKH solution file")
    if "solver_report" not in metadata:
        metadata.pop("solver_cost", None)
    return checked_solver_metadata(metadata)


def write_cvrplib_solution(routes: list[list[int]], cost: int | float | None = None) -> str:
    """CVRPLIB ``.sol`` text of library routes (customers 1..n), ``Cost`` line when given."""
    lines = [f"Route #{index}: {' '.join(str(stop) for stop in route)}" for index, route in enumerate(routes, start=1)]
    if cost is not None:
        if isinstance(cost, bool):
            raise SolutionFormatError("Cost must be a number.")
        value = _finite_cost(cost)
        lines.append(f"Cost {int(value)}" if value.is_integer() else f"Cost {value!r}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def is_time_dependent(instance: Any) -> bool:
    from mamut_routing_lib.artifacts import instance_problem_type

    return instance_problem_type(instance) in (ProblemType.TDVRP, ProblemType.TDVRPTW)


# A verified preload remains reusable while the caller holds it. Keep only its
# fingerprints here, so the index never owns a large matrix.
_verified_preloads: WeakValueDictionary = WeakValueDictionary()
_matrix_row = TypeAdapter(list)


def _distance_source_revision(instance, instance_path):
    from mamut_routing_lib.models import ArcCostsDistancesRef
    from mamut_routing_lib.sidecars import require_collection_root

    source = getattr(instance, "arc_costs_source", None)
    if not isinstance(source, ArcCostsDistancesRef) or instance_path is None:
        return None
    path = require_collection_root(instance_path) / source.distances.path
    stat = path.stat()
    descriptor_digest = sha256(instance.model_dump_json().encode()).digest()
    return (path.resolve(), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, descriptor_digest)


def _preload_key(revision, checkable):
    # The public model and every matrix row are mutable. One row at a time
    # bounds the temporary storage and preserves the original numeric values.
    digest = sha256(checkable.model_dump_json(exclude={"arc_costs"}).encode())
    for row in checkable.arc_costs:
        digest.update(_matrix_row.dump_json(row))
    return id(checkable), revision, digest.digest()


def load_checkable(instance: Any, instance_path: str | Path | None) -> Any:
    """The instance in the form the reference checker reads.

    A static instance with its collection matrix hydrated from the sidecar
    (``artifacts.hydrate_collection_instance``; other static instances pass through), or the
    ``LoadedTDInstance`` of a time-dependent one. Raises what those loaders raise.
    """
    if is_time_dependent(instance):
        from mamut_routing_lib.td._preload import load_verified_td

        if instance_path is None:
            raise ValueError("Time-dependent instances are loaded from their file; pass instance_path.")
        return load_verified_td(instance, Path(instance_path))
    from mamut_routing_lib.artifacts import hydrate_collection_instance

    revision = _distance_source_revision(instance, instance_path)
    checked = hydrate_collection_instance(instance, instance_path)
    if revision is not None and revision == _distance_source_revision(instance, instance_path):
        _verified_preloads[_preload_key(revision, checked)] = checked
    return checked


def _require_descriptor_at_path(instance, path):
    from mamut_routing_lib.artifacts import hydrate_collection_instance, load_benchmark_instance
    from mamut_routing_lib.models import LoadedStaticInstance

    descriptor = load_benchmark_instance(path)
    if isinstance(instance, LoadedStaticInstance):
        descriptor = hydrate_collection_instance(descriptor, path)
    if instance != descriptor:
        raise ValueError(f"Supplied instance differs from the descriptor or matrix at {path}")


def _require_marked_target(marked: BenchmarkBKS, instance: Any, routes, objective: ObjectiveFunction) -> None:
    from mamut_routing_lib.artifacts import get_instance_identifier

    if "standard" not in marked.metadata and "exact_evaluation" not in marked.metadata:
        raise ValueError("Expected a marked evaluation or certified record.")
    if marked.instance_name != get_instance_identifier(instance) or marked.objective_function != objective:
        raise ValueError("Marked BKS identifies a different instance or objective.")
    if marked.routes != routes:
        raise ValueError("Marked BKS routes must match their original customer identifiers and order.")


def _check_solution_model(instance, instance_path, checkable, solution, objective_function, backend):
    """Apply the path-aware checker to an already loaded solution model.

    A mismatched instance name or declared objective raises ``ValueError``. No
    ``SolutionCheckStatus`` describes it, because the routes were never checked.
    """
    from mamut_routing_lib.objectives import base_objective

    objective = objective_function or base_objective(instance)
    if solution.instance_name != instance.instance_name:
        raise ValueError(f"Solution identifies {solution.instance_name!r}, expected {instance.instance_name!r}.")
    if getattr(solution, "objective_function", objective) != objective:
        raise ValueError(f"Solution declares objective {solution.objective_function!r}, expected {objective!r}.")
    marked = isinstance(solution, BenchmarkBKS) and ("standard" in solution.metadata or "exact_evaluation" in solution.metadata)
    return check_routes(instance, instance_path, solution.routes, objective, declared_cost=solution.cost,
                        backend=backend, checkable=checkable, marked_input=solution if marked else None)


def check_routes(
    instance: Any,
    instance_path: str | Path | None,
    routes: list[list[int]],
    objective_function: ObjectiveFunction | None = None,
    *,
    declared_cost: int | float | None = None,
    backend: ExactBackend = "native",
    checkable: Any = None,
    marked_input: BenchmarkBKS | None = None,
) -> Any:
    """Check and price a submission under the exact profile for its instance type."""
    from mamut_routing_lib.checker import SolutionCheckResult, SolutionCheckStatus
    from mamut_routing_lib.exact import prepare_exact
    from mamut_routing_lib.td.checker import TDRouteEvaluation, TDSolutionCheckResult

    from mamut_routing_lib.manifests import adjacent_manifest, manifest_path_for
    manifest = None
    if instance_path is not None:
        if manifest_path_for(instance_path).exists():
            manifest = adjacent_manifest(instance_path, objective_function, routes=routes,
                                         supplied_instance=instance, supplied_checkable=checkable)
        else:
            _require_descriptor_at_path(instance, instance_path)
    if manifest is not None:
        checkable = manifest.checkable
    elif checkable is None:
        checkable = load_checkable(instance, instance_path)
    elif instance_path is not None:
        _require_same_instance(instance, checkable, Path(instance_path))
    elif getattr(checkable, "instance", checkable) is not instance:
        raise ValueError("A preloaded checkable requires its instance path for identity verification.")
    prepared = prepare_exact(checkable, backend=backend, objective_function=objective_function)
    is_td = is_time_dependent(instance)
    result_type = TDSolutionCheckResult if is_td else SolutionCheckResult
    try:
        exact = prepared.evaluate(routes)
    except ExactRouteError as error:
        return result_type.make_invalid(error.status, str(error))
    if marked_input is not None:
        from mamut_routing_lib.exact_store import verify_stored
        try:
            _require_marked_target(marked_input, instance, routes, exact.objective)
            verify_stored(marked_input, exact, manifest)
        except ValueError as error:
            return result_type.make_invalid(SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH, str(error))
    if declared_cost is not None and declared_cost != exact.display_cost:
        return result_type.make_invalid(SolutionCheckStatus.OBJECTIVE_VALUE_MISMATCH,
                                        f"Declared display cost differs from the exact {exact.objective.value} projection.")
    fields = dict(status=SolutionCheckStatus.VALID, routing_cost=exact.display_cost,
                  num_routes=len(exact.routes), exact_evaluation=exact.as_dict())
    if is_td:
        fields["route_evaluations"] = [
            TDRouteEvaluation(route=list(route), feasible=True,
                              duration=witness.duration, departure_time=witness.departure_time)
            for route, witness in zip(exact.routes, exact.route_evaluations)
        ]
    return result_type(**fields)


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


def submit_bks(
    instance_path: str | Path,
    routes: list[list[int]],
    *,
    authors: str,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | None = None,
    metadata: Mapping[str, Any] | None = None,
    instance: Any = None,
    checkable: Any = None,
    marked_input: BenchmarkBKS | None = None,
) -> Any:
    """Store ``routes`` as the BKS of the instance if they strictly improve it.

    Static instances go through ``bks.save_solution_as_bks_if_improved`` (exact binary64
    comparison), TD instances through ``td.save_td_solution_as_bks_if_improved``. The
    objective defaults to the family's published one (``objectives.base_objective``).
    Returns the store's ``BKSUpdateResult``; raises ``ValueError`` for an invalid solution,
    an objective that does not apply to the instance, empty ``authors``, a ``metadata`` key
    of :data:`STORE_METADATA_KEYS`, or a ``checkable`` that is not this instance. ``checkable``
    (:func:`load_checkable`, or ``CheckedSolve.checkable`` after a solve) spares the store
    its own load of the matrix or the travel-time functions.
    """
    from mamut_routing_lib.artifacts import get_instance_identifier, load_benchmark_instance
    from mamut_routing_lib.manifests import manifest_path_for
    from mamut_routing_lib.models import BenchmarkSolution
    from mamut_routing_lib.objectives import base_objective, require_supported_objective

    if not isinstance(authors, str) or not authors.strip():
        raise ValueError("authors must be a non-empty string (who found the solution)")
    reserved = sorted(STORE_METADATA_KEYS.intersection(metadata or {}))
    if reserved:
        raise ValueError(f"metadata cannot set what the BKS store and the checker tools write: {', '.join(reserved)}")
    path = Path(instance_path)
    if instance is not None:
        _require_descriptor_at_path(instance, path)
    loaded = instance if instance is not None else load_benchmark_instance(path)
    if checkable is not None and is_time_dependent(loaded):
        from mamut_routing_lib.td.artifacts import LoadedTDInstance
        if not isinstance(checkable, LoadedTDInstance) or Path(checkable.instance_path).resolve() != path.resolve():
            raise ValueError(f"The supplied TD checkable does not identify {path}")
        # The TD store binds the operands once it has both candidate and incumbent routes.
    elif checkable is not None and not manifest_path_for(path).exists():
        _require_same_instance(loaded, checkable, path)
    objective = require_supported_objective(loaded, objective_function or base_objective(loaded))
    solution = BenchmarkSolution(
        instance_name=get_instance_identifier(loaded),
        routes=routes,
        cost=None,
        metadata={},
    )
    if marked_input is not None:
        _require_marked_target(marked_input, loaded, routes, objective)
        solution = marked_input
    if is_time_dependent(loaded):
        from mamut_routing_lib.td import save_td_solution_as_bks_if_improved

        return save_td_solution_as_bks_if_improved(
            checkable if checkable is not None else path,
            solution,
            authors=authors,
            metadata=dict(metadata or {}),
            objective_function=objective,
            backend=backend,
        )
    from mamut_routing_lib.bks import _save_bks_if_improved, create_bks_from_solution, save_solution_as_bks_if_improved

    if checkable is None or manifest_path_for(path).exists():
        return save_solution_as_bks_if_improved(path, objective, solution, authors=authors, metadata=dict(metadata or {}),
                                               backend=backend, supplied_checkable=checkable)
    # The same two steps as save_solution_as_bks_if_improved, on the instance already in hand.
    bks = create_bks_from_solution(checkable, solution, objective, authors=authors, metadata=dict(metadata or {}), backend=backend)
    return _save_bks_if_improved(checkable, bks, path, backend=backend, check_descriptor=False)


def _require_same_instance(loaded: Any, checkable: Any, path: Path) -> None:
    """Refuse a ``checkable`` that was built from another instance than the one being stored."""
    from mamut_routing_lib.artifacts import get_instance_identifier, hydrate_collection_instance

    from mamut_routing_lib.models import LoadedStaticInstance
    from mamut_routing_lib.td.artifacts import LoadedTDInstance

    expected = get_instance_identifier(loaded)
    held_model = checkable.instance if isinstance(checkable, LoadedTDInstance) else checkable
    held = get_instance_identifier(held_model)
    if held != expected or held_model.benchmark_name != loaded.benchmark_name:
        raise ValueError(f"checkable holds instance {held!r} of {held_model.benchmark_name}, expected {expected!r} of {loaded.benchmark_name}")
    if isinstance(checkable, LoadedTDInstance):
        from mamut_routing_lib.td._preload import require_td_preload
        require_td_preload(checkable, loaded, path)
    else:
        if not isinstance(checkable, LoadedStaticInstance):
            raise ValueError("Expected a materialized static instance as the checkable.")
        revision = _distance_source_revision(loaded, path)
        if revision is not None and _verified_preloads.get(_preload_key(revision, checkable)) is checkable:
            return
        if checkable != hydrate_collection_instance(loaded, path):
            raise ValueError(f"Supplied materialized input differs from the descriptor or matrix at {path}")



def _preview(summary: dict[str, Any], action: str, **fields: Any) -> dict[str, Any]:
    from typing import get_args

    from mamut_routing_lib.bks import BKSPreviewAction

    if action not in get_args(BKSPreviewAction):
        raise ValueError(f"Unknown BKS preview action {action!r}")
    return {**summary, "action": action, **fields}


def preview_bks_update(
    instance_path: str | Path,
    routes: list[list[int]],
    *,
    objective_function: ObjectiveFunction | None = None,
    backend: ExactBackend = "native",
    instance: Any = None,
    checkable: Any = None,
    marked_input: BenchmarkBKS | None = None,
) -> dict[str, Any]:
    """What ``submit_bks`` would do, without writing: ``action`` (a ``BKSPreviewAction``) is ``would_create``,
    ``would_replace``, ``would_keep`` (``tie`` when equal), ``invalid`` (the candidate) or
    ``stored_invalid`` (the stored BKS fails the checker; ``submit_bks`` would raise).

    Re-evaluates both sets of routes under the same exact profile and objective.
    """
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, get_instance_identifier, load_benchmark_instance, load_bks
    from mamut_routing_lib.exact import prepare_exact
    from mamut_routing_lib.exact_store import verify_stored
    from mamut_routing_lib.objectives import base_objective, require_supported_objective

    path = Path(instance_path)
    from mamut_routing_lib.manifests import adjacent_manifest, manifest_path_for
    if instance is not None and not manifest_path_for(path).exists():
        _require_descriptor_at_path(instance, path)
    loaded = instance if instance is not None else load_benchmark_instance(path)
    objective = require_supported_objective(loaded, objective_function or base_objective(loaded))
    bks_path = get_bks_path_for_instance(path, objective)
    summary = dict(objective_function=objective.value, path=str(bks_path), candidate_cost=None,
                   candidate_num_routes=None, existing_cost=None, existing_num_routes=None, tie=False)
    try:
        existing = load_bks(bks_path) if bks_path.exists() else None
    except ValueError as error:
        return _preview(summary, "stored_invalid", error=str(error))
    all_routes = [*routes, *existing.routes] if existing is not None else routes
    manifest = adjacent_manifest(path, objective, routes=all_routes, supplied_instance=loaded, supplied_checkable=checkable)
    if manifest is not None:
        checkable = manifest.checkable
    elif checkable is None:
        checkable = load_checkable(loaded, path)
    else:
        _require_same_instance(loaded, checkable, path)
    prepared = prepare_exact(checkable, backend=backend, objective_function=objective)
    try:
        candidate = prepared.evaluate(routes)
    except ExactRouteError as error:
        return _preview(summary, "invalid", error=str(error))
    if marked_input is not None:
        try:
            _require_marked_target(marked_input, loaded, routes, objective)
            verify_stored(marked_input, candidate, manifest)
        except ValueError as error:
            return _preview(summary, "invalid", error=str(error))
    summary.update(candidate_cost=candidate.display_cost, candidate_num_routes=len(candidate.routes))
    if existing is None:
        return _preview(summary, "would_create")
    try:
        if existing.instance_name != get_instance_identifier(loaded):
            raise ValueError("Stored BKS belongs to a different instance.")
        incumbent = prepared.evaluate(existing.routes)
        verify_stored(existing, incumbent, manifest)
        summary.update(existing_cost=incumbent.display_cost, existing_num_routes=len(incumbent.routes))
    except ValueError as error:
        return _preview(summary, "stored_invalid", error=str(error))
    if candidate.objective_key < incumbent.objective_key:
        return _preview(summary, "would_replace")
    return _preview(summary, "would_keep", tie=candidate.objective_key == incumbent.objective_key)

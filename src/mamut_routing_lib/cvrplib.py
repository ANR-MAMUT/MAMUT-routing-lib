"""CVRPLIB / TSPLIB ``.vrp`` (and Solomon ``.txt``) export of benchmark instances.

Classic solvers read the TSPLIB-derived CVRPLIB text format, not the lib's
``.vrp.json`` contract. This module renders any *static* instance (CVRP or
VRPTW, embedded matrix or slim collection instance) into that format with one
contract, shared by the ``mamut-routing export vrp`` CLI, the MAMUT-routing-tools
workbench and the client-side writer of the benchmark website:

.. code-block:: text

    NAME : <instance_name>
    COMMENT : <comment>
    TYPE : CVRP | CVRPTW
    DIMENSION : <num_customers + 1>
    [VEHICLES : <num_vehicles>]            (only when the fleet is fixed)
    EDGE_WEIGHT_TYPE : EXPLICIT | EUC_2D
    [EDGE_WEIGHT_FORMAT : FULL_MATRIX]     (EXPLICIT only)
    CAPACITY : <vehicle_capacity>
    [EDGE_WEIGHT_SECTION + rows]           (EXPLICIT only)
    NODE_COORD_SECTION      "<i+1> <x> <y>"
    DEMAND_SECTION          "<i+1> <demand>"
    [TIME_WINDOW_SECTION    "<i+1> <ready> <due>"]   (CVRPTW)
    [SERVICE_TIME_SECTION   "<i+1> <service>"]       (CVRPTW)
    DEPOT_SECTION / <depot+1> / -1 / EOF

Node ids are 1-based (the JSON ``depot`` index plus one). Number formatting is
value-driven so that a JavaScript mirror working from parsed JSON (which has no
int/float distinction) produces the same bytes:

- arc costs of a collection source (``distances-sidecar`` / ``euclidean``)
  print with a fixed number of decimals (the source's ``decimals``, 3);
- any other vector prints as integers when *every* entry is integral, else
  as shortest round-trip floats (``.0`` kept for integral entries) -- except
  coordinates, which print with 6 decimals when not all integral;
- the ``EXPLICIT`` default is byte-identical to the ``.vrp`` files committed
  next to the Poryos2026/Mamut2026 CVRP instances.

``EUC_2D`` is an opt-in for euclidean-metric instances only: it drops the
matrix, and TSPLIB readers then use ``nint(hypot)`` distances, which differ
from the published 3-decimal costs. Time-dependent instances have no static
matrix and are refused (:class:`UnsupportedInstanceError`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from mamut_routing_lib.artifacts import (
    AnyBenchmarkInstance,
    instance_problem_type,
    load_benchmark_instance,
    resolve_arc_costs,
)
from mamut_routing_lib.enums import BenchmarkName, MetricVariant, ProblemType
from mamut_routing_lib.models import (
    ArcCostsEuclidean,
    BenchmarkInstance,
    BenchmarkInstanceCVRP,
    BenchmarkInstanceCVRPCollection,
    BenchmarkInstanceVRPTWCollection,
)

EdgeWeightType = Literal["EXPLICIT", "EUC_2D"]
ExportFormat = Literal["vrp", "solomon"]
ExportStatus = Literal["written", "exists", "unsupported"]

EDGE_WEIGHT_TYPES: tuple[EdgeWeightType, ...] = ("EXPLICIT", "EUC_2D")
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("vrp", "solomon")

#: Decimals of collection arc costs (Poryos2026 / Mamut2026 sidecars).
DEFAULT_COST_DECIMALS = 3
#: Decimals of non-integral coordinates (ENU metres).
COORDINATE_DECIMALS = 6
#: Families whose names carry the fleet lower bound; their COMMENT records it
#: the way CVRPLIB does ("No of trucks"), worded as a lower bound.
FAMILIES_WITH_FLEET_IN_COMMENT = frozenset({BenchmarkName.MAMUT_2026.value})
#: Suffix of the export file by format.
EXPORT_SUFFIXES: dict[str, str] = {"vrp": ".vrp", "solomon": ".txt"}


class UnsupportedInstanceError(ValueError):
    """The instance cannot be expressed in the requested classic format."""


@dataclass(frozen=True)
class VrpExportOptions:
    edge_weight_type: EdgeWeightType = "EXPLICIT"
    format: ExportFormat = "vrp"
    comment: str | None = None

    def __post_init__(self) -> None:
        if self.edge_weight_type not in EDGE_WEIGHT_TYPES:
            raise ValueError(f"edge_weight_type must be one of {EDGE_WEIGHT_TYPES}, got {self.edge_weight_type!r}")
        if self.format not in EXPORT_FORMATS:
            raise ValueError(f"format must be one of {EXPORT_FORMATS}, got {self.format!r}")


@dataclass(frozen=True)
class ExportResult:
    instance_path: Path
    output_path: Path
    status: ExportStatus
    problem_type: ProblemType
    num_customers: int
    message: str = ""


# --------------------------------------------------------------------------- #
# Number formatting
# --------------------------------------------------------------------------- #


def _is_integral(value: Any) -> bool:
    number = float(value)
    return math.isfinite(number) and number.is_integer()


def _format_float(value: Any) -> str:
    number = float(value)
    return f"{number:.1f}" if number.is_integer() else repr(number)


def vector_formatter(values: Sequence[Any], decimals: int | None = None) -> Callable[[Any], str]:
    """The formatter of one vector (or flattened matrix): fixed decimals when
    given, else integers when every entry is integral, else round-trip floats."""
    if decimals is not None:
        return lambda value: f"{float(value):.{decimals}f}"
    if all(_is_integral(value) for value in values):
        return lambda value: str(int(float(value)))
    return _format_float


def coordinate_formatter(coordinates: Sequence[Sequence[Any]]) -> Callable[[Any], str]:
    flat = [component for point in coordinates for component in point]
    if all(_is_integral(value) for value in flat):
        return lambda value: str(int(float(value)))
    return lambda value: f"{float(value):.{COORDINATE_DECIMALS}f}"


def euclidean_arc_costs(coordinates: Sequence[Sequence[Any]], decimals: int = DEFAULT_COST_DECIMALS) -> list[list[float]]:
    """The canonical euclidean matrix: ``round(hypot(dx, dy), decimals)``, zero diagonal."""
    points = [(float(x), float(y)) for x, y in coordinates]
    return [
        [0.0 if i == j else round(math.hypot(bx - ax, by - ay), decimals) for j, (bx, by) in enumerate(points)]
        for i, (ax, ay) in enumerate(points)
    ]


# --------------------------------------------------------------------------- #
# Renderers over plain lists
# --------------------------------------------------------------------------- #


def _check_lengths(dimension: int, **vectors: Sequence[Any] | None) -> None:
    for field_name, vector in vectors.items():
        if vector is not None and len(vector) != dimension:
            raise ValueError(f"{field_name} has {len(vector)} entries, expected DIMENSION {dimension}")


def render_cvrplib(
    *,
    name: str,
    comment: str,
    coordinates: Sequence[Sequence[Any]],
    demands: Sequence[Any],
    capacity: int,
    depot: int = 0,
    num_vehicles: int | None = None,
    arc_costs: Sequence[Sequence[Any]] | None = None,
    decimals: int | None = None,
    time_windows: Sequence[Sequence[Any]] | None = None,
    service_times: Sequence[Any] | None = None,
    edge_weight_type: EdgeWeightType = "EXPLICIT",
) -> str:
    """Render the CVRPLIB text of a CVRP (``time_windows is None``) or CVRPTW instance.

    ``arc_costs`` is required for ``EXPLICIT`` and ignored for ``EUC_2D``;
    ``decimals`` fixes the matrix precision (collection sources), ``None``
    selects the value-driven integer/round-trip rule.
    """
    if edge_weight_type not in EDGE_WEIGHT_TYPES:
        raise ValueError(f"unsupported edge_weight_type {edge_weight_type!r}")
    dimension = len(coordinates)
    if dimension < 2:
        raise ValueError("an instance needs a depot and at least one customer")
    if not 0 <= depot < dimension:
        raise ValueError(f"depot index {depot} out of range for DIMENSION {dimension}")
    _check_lengths(dimension, demands=demands, time_windows=time_windows, service_times=service_times)
    if (time_windows is None) != (service_times is None):
        raise ValueError("time_windows and service_times must be given together")
    explicit = edge_weight_type == "EXPLICIT"
    if explicit:
        if arc_costs is None:
            raise ValueError("EXPLICIT edge weights need arc_costs")
        _check_lengths(dimension, arc_costs=arc_costs)
        for index, row in enumerate(arc_costs):
            if len(row) != dimension:
                raise ValueError(f"arc_costs row {index} has {len(row)} entries, expected {dimension}")

    lines = [f"NAME : {name}"]
    if comment:
        lines.append(f"COMMENT : {comment}")
    lines.extend(
        [
            f"TYPE : {'CVRPTW' if time_windows is not None else 'CVRP'}",
            f"DIMENSION : {dimension}",
        ]
    )
    if num_vehicles is not None:
        lines.append(f"VEHICLES : {int(num_vehicles)}")
    lines.append(f"EDGE_WEIGHT_TYPE : {edge_weight_type}")
    if explicit:
        lines.append("EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append(f"CAPACITY : {int(capacity)}")
    if explicit:
        assert arc_costs is not None
        fmt_cost = vector_formatter([value for row in arc_costs for value in row], decimals)
        lines.append("EDGE_WEIGHT_SECTION")
        lines.extend(" ".join(fmt_cost(value) for value in row) for row in arc_costs)

    fmt_coord = coordinate_formatter(coordinates)
    lines.append("NODE_COORD_SECTION")
    lines.extend(f"{index + 1} {fmt_coord(x)} {fmt_coord(y)}" for index, (x, y) in enumerate(coordinates))

    fmt_demand = vector_formatter(demands)
    lines.append("DEMAND_SECTION")
    lines.extend(f"{index + 1} {fmt_demand(demand)}" for index, demand in enumerate(demands))

    if time_windows is not None and service_times is not None:
        fmt_tw = vector_formatter([bound for window in time_windows for bound in window])
        lines.append("TIME_WINDOW_SECTION")
        lines.extend(f"{index + 1} {fmt_tw(ready)} {fmt_tw(due)}" for index, (ready, due) in enumerate(time_windows))
        fmt_service = vector_formatter(service_times)
        lines.append("SERVICE_TIME_SECTION")
        lines.extend(f"{index + 1} {fmt_service(service)}" for index, service in enumerate(service_times))

    lines.extend(["DEPOT_SECTION", str(depot + 1), "-1", "EOF", ""])
    return "\n".join(lines)


def render_solomon(
    *,
    name: str,
    capacity: int,
    num_vehicles: int | None,
    coordinates: Sequence[Sequence[Any]],
    demands: Sequence[Any],
    time_windows: Sequence[Sequence[Any]],
    service_times: Sequence[Any],
    depot: int = 0,
) -> str:
    """Render the Solomon / Gehring-Homberger VRPTW text (coordinates only).

    Customer ids are 0-based with the depot first, as in the original files;
    ``NUMBER`` is the fixed fleet or the customer count when the fleet is free.
    """
    dimension = len(coordinates)
    if dimension < 2:
        raise ValueError("an instance needs a depot and at least one customer")
    _check_lengths(dimension, demands=demands, time_windows=time_windows, service_times=service_times)
    if depot != 0:
        raise ValueError("the Solomon format expects the depot at index 0")
    fleet = int(num_vehicles) if num_vehicles is not None else dimension - 1
    fmt_coord = coordinate_formatter(coordinates)
    fmt_demand = vector_formatter(demands)
    fmt_tw = vector_formatter([bound for window in time_windows for bound in window])
    fmt_service = vector_formatter(service_times)
    lines = [
        name,
        "",
        "VEHICLE",
        "NUMBER     CAPACITY",
        f"{fleet:>6}       {int(capacity)}",
        "",
        "CUSTOMER",
        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME",
        "",
    ]
    for index in range(dimension):
        x, y = coordinates[index]
        ready, due = time_windows[index]
        lines.append(
            f"{index:>5} {fmt_coord(x):>10} {fmt_coord(y):>10} {fmt_demand(demands[index]):>10} "
            f"{fmt_tw(ready):>10} {fmt_tw(due):>10} {fmt_service(service_times[index]):>10}"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Instance-level conversion
# --------------------------------------------------------------------------- #


def _metadata_value(instance: Any, key: str) -> Any:
    metadata = getattr(instance, "metadata", None)
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _is_collection(instance: Any) -> bool:
    return isinstance(instance, (BenchmarkInstanceCVRPCollection, BenchmarkInstanceVRPTWCollection))


def _is_vrptw(instance: Any) -> bool:
    return isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTWCollection))


def _reject_time_dependent(instance: Any) -> None:
    if instance_problem_type(instance) in (ProblemType.TDVRP, ProblemType.TDVRPTW):
        raise UnsupportedInstanceError(
            f"{getattr(instance, 'instance_name', '?')} is time-dependent (travel is an arrival-time "
            "function, not a static matrix) and cannot be written as a classic .vrp"
        )


def instance_metric_variant(instance: AnyBenchmarkInstance) -> MetricVariant | None:
    """The metric of an instance: ``metric_variant`` (collection) or the
    ``metadata.metric_variant`` that enriched historical instances carry."""
    value = getattr(instance, "metric_variant", None)
    if value is None:
        value = _metadata_value(instance, "metric_variant")
    if value is None:
        return None
    try:
        return MetricVariant(str(_enum_value(value)))
    except ValueError:
        return None


def _require_euclidean(instance: AnyBenchmarkInstance, what: str) -> None:
    metric = instance_metric_variant(instance)
    if metric != MetricVariant.EUCLIDEAN:
        raise UnsupportedInstanceError(
            f"{what} is only meaningful for euclidean-metric instances; "
            f"{instance.instance_name} has metric {metric.value if metric else 'unknown'}"
        )


def collection_cost_decimals(instance: AnyBenchmarkInstance) -> int | None:
    """Fixed decimals of a collection matrix (``None`` for embedded matrices)."""
    if not _is_collection(instance):
        return None
    source = instance.arc_costs_source
    if isinstance(source, ArcCostsEuclidean):
        return int(source.decimals)
    return DEFAULT_COST_DECIMALS


def format_vrp_comment(instance: AnyBenchmarkInstance, *, edge_weight_type: EdgeWeightType = "EXPLICIT") -> str:
    """The default COMMENT, reconstructed from the instance JSON alone.

    Collection instances reproduce the comment of the committed family
    ``.vrp`` files byte for byte; historical instances name their family and
    authors. ``EUC_2D`` appends the rounding caveat.
    """
    benchmark = str(_enum_value(instance.benchmark_name))
    if _is_collection(instance):
        city = _metadata_value(instance, "city") or _metadata_value(instance, "place_slug") or "unknown"
        head = f"{benchmark} {instance.metric_variant.value} metric; city {city}; "
        fleet = ""
        fleet_lb = _metadata_value(instance, "num_vehicles_lb")
        if benchmark in FAMILIES_WITH_FLEET_IN_COMMENT and fleet_lb is not None:
            fleet = f"No of trucks: {int(fleet_lb)} (lower bound, fleet not fixed); "
        comment = f"{head}{fleet}3-decimal seconds/meters; ENU ref in {instance.instance_name}.vrp.json"
        tw_set = _metadata_value(instance, "tw_set")
        tw_name = tw_set.get("name") if isinstance(tw_set, dict) else None
        if tw_name:
            comment = f"{comment}; time windows set {tw_name}"
    else:
        authors = _metadata_value(instance, "authors")
        parts = [f"{benchmark} {instance.instance_name}"]
        if authors:
            parts.append(f"authors: {authors}")
        parts.append("converted from MAMUT-routing .vrp.json")
        comment = "; ".join(parts)
    if edge_weight_type == "EUC_2D":
        comment = f"{comment}; EUC_2D: costs are TSPLIB nint distances, not the published 3-decimal costs"
    return comment


def _explicit_matrix(
    instance: AnyBenchmarkInstance,
    arc_costs: Sequence[Sequence[Any]] | None,
    instance_path: str | Path | None,
    collection_root: str | Path | None,
) -> tuple[Sequence[Sequence[Any]], int | None]:
    if arc_costs is not None:
        return arc_costs, collection_cost_decimals(instance)
    if _is_collection(instance):
        decimals = collection_cost_decimals(instance)
        if isinstance(instance.arc_costs_source, ArcCostsEuclidean):
            # Same formula as resolve_arc_costs, without needing the file on disk.
            return euclidean_arc_costs(instance.coordinates, decimals or 0), decimals
        if instance_path is None:
            raise ValueError("collection instances resolve arc costs from their sidecars; pass instance_path")
        return resolve_arc_costs(instance, instance_path, collection_root), decimals
    return instance.arc_costs, None


def instance_to_vrp_text(
    instance: AnyBenchmarkInstance,
    arc_costs: Sequence[Sequence[Any]] | None = None,
    *,
    instance_path: str | Path | None = None,
    collection_root: str | Path | None = None,
    options: VrpExportOptions = VrpExportOptions(),
) -> str:
    """The classic text of a static instance under ``options``.

    ``arc_costs`` short-circuits matrix resolution (callers that already hold
    the hydrated matrix); slim collection instances otherwise resolve it from
    their sidecar, which needs ``instance_path`` (and ``collection_root`` when
    the file was copied out of its collection tree).
    """
    _reject_time_dependent(instance)
    vrptw = _is_vrptw(instance)
    time_windows = list(instance.time_windows) if vrptw else None
    service_times = list(instance.service_times) if vrptw else None

    if options.format == "solomon":
        if not vrptw:
            raise UnsupportedInstanceError(
                f"the Solomon format is VRPTW-only; {instance.instance_name} is a CVRP instance"
            )
        _require_euclidean(instance, "the Solomon format (coordinates only)")
        assert time_windows is not None and service_times is not None
        return render_solomon(
            name=instance.instance_name,
            capacity=instance.vehicle_capacity,
            num_vehicles=instance.num_vehicles,
            coordinates=instance.coordinates,
            demands=instance.demands,
            time_windows=time_windows,
            service_times=service_times,
            depot=instance.depot,
        )

    matrix: Sequence[Sequence[Any]] | None = None
    decimals: int | None = None
    if options.edge_weight_type == "EUC_2D":
        _require_euclidean(instance, "EUC_2D (coordinates only)")
    else:
        matrix, decimals = _explicit_matrix(instance, arc_costs, instance_path, collection_root)

    comment = options.comment if options.comment is not None else format_vrp_comment(
        instance, edge_weight_type=options.edge_weight_type
    )
    return render_cvrplib(
        name=instance.instance_name,
        comment=comment,
        coordinates=instance.coordinates,
        demands=instance.demands,
        capacity=instance.vehicle_capacity,
        depot=instance.depot,
        num_vehicles=instance.num_vehicles,
        arc_costs=matrix,
        decimals=decimals,
        time_windows=time_windows,
        service_times=service_times,
        edge_weight_type=options.edge_weight_type,
    )


def export_filename(source_name: str, options: VrpExportOptions = VrpExportOptions()) -> str:
    """``<stem>.vrp`` (or ``.txt`` for Solomon) for an instance file or name."""
    stem = source_name.removesuffix(".vrp.json").removesuffix(".json")
    return f"{stem}{EXPORT_SUFFIXES[options.format]}"


def export_instance_file(
    instance_path: str | Path,
    output_path: str | Path | None = None,
    *,
    collection_root: str | Path | None = None,
    options: VrpExportOptions = VrpExportOptions(),
    overwrite: bool = False,
) -> ExportResult:
    """Convert one ``.vrp.json`` file; the default output sits next to it.

    Existing outputs are left alone unless ``overwrite``; unsupported instances
    (time-dependent, or EUC_2D/Solomon on a non-euclidean metric) are reported
    with status ``unsupported`` rather than raised, so batches keep going.
    """
    source = Path(instance_path)
    instance = load_benchmark_instance(source)
    problem_type = instance_problem_type(instance)
    num_customers = int(instance.num_customers)
    target = Path(output_path) if output_path is not None else source.with_name(export_filename(source.name, options))
    if target.exists() and not overwrite:
        return ExportResult(source, target, "exists", problem_type, num_customers, "output exists (use overwrite)")
    try:
        text = instance_to_vrp_text(instance, instance_path=source, collection_root=collection_root, options=options)
    except UnsupportedInstanceError as exc:
        return ExportResult(source, target, "unsupported", problem_type, num_customers, str(exc))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return ExportResult(source, target, "written", problem_type, num_customers)


__all__ = [
    "COORDINATE_DECIMALS",
    "DEFAULT_COST_DECIMALS",
    "EDGE_WEIGHT_TYPES",
    "EXPORT_FORMATS",
    "EXPORT_SUFFIXES",
    "ExportResult",
    "UnsupportedInstanceError",
    "VrpExportOptions",
    "collection_cost_decimals",
    "coordinate_formatter",
    "euclidean_arc_costs",
    "export_filename",
    "export_instance_file",
    "format_vrp_comment",
    "instance_metric_variant",
    "instance_to_vrp_text",
    "render_cvrplib",
    "render_solomon",
    "vector_formatter",
]

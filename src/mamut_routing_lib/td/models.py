"""Pydantic models for time-dependent (TDVRPTW / TDVRP) benchmark instances.

TD instances have no static ``arc_costs`` matrix: travel is described by
per-arc arrival-time functions (ATFs) shipped in a sidecar file referenced by
the ``td`` block. The TDVRPTW variant carries service times and time windows;
the TDVRP variant keeps service times but has no time windows. Both share the
same ATF sidecar content.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mamut_routing_lib.enums import BenchmarkName, InstanceOrigin
from mamut_routing_lib.models import Coordinate, ReferenceLLA
from mamut_routing_lib.sidecars import SidecarRef

TD_ATF_MODEL = "atf-ndcpwlf"
TD_IGP_MODEL = "igp-profile"
TD_ROAD_MODEL = "road-graph"
ATF_FORMAT = "mamut-td-atf"
ATF_FORMAT_VERSION = 1
IGP_CATEGORIES_FORMAT = "mamut-td-igp-categories"
IGP_CATEGORIES_FORMAT_VERSION = 1
ROAD_GRAPH_FORMAT = "mamut-road-graph"
ROAD_GRAPH_FORMAT_VERSION = 2
TRAFFIC_OVERLAY_FORMAT = "mamut-traffic-overlay"
TRAFFIC_OVERLAY_FORMAT_VERSION = 1


def _validate_sidecar_path(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value.startswith("/") or ".." in value.split("/"):
        raise ValueError(f"{field_name} must be a plain relative path next to the instance file")
    return value


class TDArrivalFunctionsRef(BaseModel):
    """Reference to the ATF sidecar file of a TD instance.

    ``atf_path`` is relative to the directory containing the instance file.
    ``atf_sha256`` is computed over the uncompressed canonical JSON bytes of
    the sidecar, so it is stable across ``.json`` and ``.json.gz`` storage.
    """

    model_config = ConfigDict(extra="forbid")

    model: Literal["atf-ndcpwlf"] = TD_ATF_MODEL
    atf_path: str
    atf_sha256: str | None = None

    @field_validator("atf_path")
    @classmethod
    def validate_atf_path(cls, value: str) -> str:
        return _validate_sidecar_path(value, "atf_path")


class TDIGPProfileRef(BaseModel):
    """Compact IGP (Ichoua-Gendreau-Potvin 2003) time-dependent travel model.

    Instead of shipping a consolidated ATF sidecar, the instance stores the
    IGP data that fully determines it: contiguous ``time_periods`` (stored
    absolute floats are canonical — never recomputed), a strictly positive
    ``speeds`` matrix (``num_categories x num_periods``; FIFO by
    construction), and a reference to the arc-category sidecar. The canonical
    ATFs are materialized deterministically on load
    (``mamut_routing_lib.td.igp``); ``atf_sha256`` pins the materialized
    canonical ATF bytes exactly as it would for a committed sidecar. See the
    TD benchmark standard and the Lera2026 family design note.
    """

    model_config = ConfigDict(extra="forbid")

    model: Literal["igp-profile"] = TD_IGP_MODEL
    time_periods: list[tuple[float, float]]
    speeds: list[list[float]]
    categories_path: str
    categories_sha256: str | None = None
    atf_sha256: str | None = None

    @field_validator("categories_path")
    @classmethod
    def validate_categories_path(cls, value: str) -> str:
        return _validate_sidecar_path(value, "categories_path")

    @field_validator("time_periods")
    @classmethod
    def validate_time_periods(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not value:
            raise ValueError("time_periods must be non-empty")
        for index, (start, end) in enumerate(value):
            if start >= end:
                raise ValueError(f"time period {index} is empty or reversed: [{start}, {end}]")
            if index > 0 and value[index - 1][1] != start:
                raise ValueError(
                    f"time periods must be contiguous: period {index} starts at {start}, "
                    f"previous ends at {value[index - 1][1]}"
                )
        return value

    @field_validator("speeds")
    @classmethod
    def validate_speeds(cls, value: list[list[float]], info: Any) -> list[list[float]]:
        if not value:
            raise ValueError("speeds must have at least one category row")
        num_periods = len(info.data.get("time_periods", []))
        for c, row in enumerate(value):
            if num_periods and len(row) != num_periods:
                raise ValueError(
                    f"speeds row {c} has {len(row)} entries, expected one per time period ({num_periods})"
                )
            for p, speed in enumerate(row):
                if speed <= 0:
                    raise ValueError(f"speeds[{c}][{p}] = {speed} must be strictly positive (FIFO)")
        return value

    def num_categories(self) -> int:
        return len(self.speeds)


class TDRoadGraphRef(BaseModel):
    """Compact road-network time-dependent travel model (format v2).

    The instance references **two** sidecars, both collection-root-relative
    and sha-pinned: the static road graph shared by every traffic subinstance
    of the base (``<base>.road.json.gz``: trimmed subgraph with per-edge
    free-flow ``speed_limit`` and vertex coordinates) and one traffic overlay
    (``<base>.traffic-<model>-<intensity>.json.gz``: per-edge 24-bin speeds
    aligned with the graph's edge order, clamped at free-flow). The canonical
    ATFs are materialized deterministically on load
    (``mamut_routing_lib.td.roadgraph``): pinned Dijkstra over the static
    free-flow times, exact per-edge arrival functions from the overlay
    speeds sampled exactly along the pinned paths on a fixed departure grid,
    then deterministic decimation. ``sample_step`` and ``simplify_tolerance``
    are the authoritative materialization parameters and must equal the
    values recorded in the road sidecar. ``atf_sha256`` pins the materialized
    canonical ATF bytes exactly as it would for a committed sidecar. See the
    TD benchmark standard (road-graph v2).
    """

    model_config = ConfigDict(extra="forbid")

    model: Literal["road-graph"] = TD_ROAD_MODEL
    graph: SidecarRef
    traffic: SidecarRef
    atf_sha256: str | None = None
    sample_step: float
    simplify_tolerance: float

    @field_validator("sample_step")
    @classmethod
    def validate_sample_step(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"sample_step must be strictly positive, got {value}")
        return value

    @field_validator("simplify_tolerance")
    @classmethod
    def validate_simplify_tolerance(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"simplify_tolerance must be >= 0, got {value}")
        return value


AnyTDTravelModelRef = Annotated[
    TDArrivalFunctionsRef | TDIGPProfileRef | TDRoadGraphRef,
    Field(discriminator="model"),
]


class _TDInstanceValidationMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    num_customers: int
    num_vehicles: int | None = None
    vehicle_capacity: int
    coordinates: list[Coordinate]
    demands: list[int]
    service_times: list[int | float]
    depot: int = Field(default=0, ge=0)
    reference_lla: ReferenceLLA | None = None
    horizon: tuple[int | float, int | float]
    fleet_fixed_cost: int | float | None = None
    td: AnyTDTravelModelRef
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("num_customers", "vehicle_capacity")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("num_vehicles")
    @classmethod
    def validate_positive_optional(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("coordinates", "demands", "service_times")
    @classmethod
    def validate_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of {info.field_name} must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        return value

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, value: tuple[int | float, int | float]) -> tuple[int | float, int | float]:
        if value[0] >= value[1]:
            raise ValueError("horizon must be a non-empty interval [start, end]")
        # Anchored at abscissa 0: the vertex ready-time functions (td.pwlf.make_theta) and the
        # return clamp in td.checker start their domain at 0, and the composition hypothesis (the
        # outer domain does not start above the inner function's first value) needs every arc
        # function anchored at a horizon start >= 0.
        if value[0] < 0:
            raise ValueError(f"horizon must start at or after 0, got {value[0]!r}")
        return value

    @field_validator("fleet_fixed_cost")
    @classmethod
    def validate_fleet_fixed_cost(cls, value: int | float | None) -> int | float | None:
        # Normative per-used-vehicle fixed cost of the FleetCostDuration
        # objective, in the instance's time unit; None on families that only
        # score Duration.
        if value is not None and not value >= 0:
            raise ValueError(f"fleet_fixed_cost must be >= 0, got {value!r}")
        return value

    @model_validator(mode="after")
    def validate_igp_periods_span_horizon(self) -> "_TDInstanceValidationMixin":
        if isinstance(self.td, TDIGPProfileRef):
            start, end = float(self.horizon[0]), float(self.horizon[1])
            periods = self.td.time_periods
            if periods[0][0] != start or periods[-1][1] != end:
                raise ValueError(
                    f"igp-profile time_periods [{periods[0][0]}, {periods[-1][1]}] "
                    f"must span exactly the horizon [{start}, {end}]"
                )
        return self


class BenchmarkInstanceTDVRPTW(_TDInstanceValidationMixin):
    time_windows: list[tuple[int | float, int | float]]

    @field_validator("time_windows")
    @classmethod
    def validate_time_window_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of time_windows must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        for index, (earliest, latest) in enumerate(value):
            if earliest > latest:
                raise ValueError(f"time window at index {index} has earliest > latest")
        return value


class BenchmarkInstanceTDVRP(_TDInstanceValidationMixin):
    pass


AnyTDBenchmarkInstance = BenchmarkInstanceTDVRPTW | BenchmarkInstanceTDVRP

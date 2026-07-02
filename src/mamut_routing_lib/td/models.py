"""Pydantic models for time-dependent (TDVRPTW / TDVRP) benchmark instances.

TD instances have no static ``arc_costs`` matrix: travel is described by
per-arc arrival-time functions (ATFs) shipped in a sidecar file referenced by
the ``td`` block. The TDVRPTW variant carries service times and time windows;
the TDVRP variant keeps service times but has no time windows. Both share the
same ATF sidecar content.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mamut_routing_lib.enums import BenchmarkName, InstanceOrigin
from mamut_routing_lib.models import Coordinate

TD_ATF_MODEL = "atf-ndcpwlf"
ATF_FORMAT = "mamut-td-atf"
ATF_FORMAT_VERSION = 1


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
        if not value:
            raise ValueError("atf_path must be non-empty")
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("atf_path must be a plain relative path next to the instance file")
        return value


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
    horizon: tuple[int | float, int | float]
    td: TDArrivalFunctionsRef
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
        return value


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

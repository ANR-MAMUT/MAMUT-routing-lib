from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mamut_routing_lib.enums import (
    BenchmarkName,
    InstanceOrigin,
    MetricVariant,
    ObjectiveFunction,
    ProblemType,
)


Coordinate: TypeAlias = tuple[int | float, int | float]
ArcCost: TypeAlias = int | float

# Instance-model schema version. Bumped from 1.0.0 -> 1.1.0 when
# BenchmarkInstanceTDVRP was added as an additive optional model; existing
# CVRP/VRPTW instances remain valid without modification.
INSTANCE_MODEL_SCHEMA_VERSION = "1.1.0"


def _validate_relative_path(path_value: str) -> str:
    if path_value.startswith("/"):
        raise ValueError("paths must be relative to the benchmark repository root")
    if not path_value:
        raise ValueError("paths must be non-empty")
    return path_value


class ArtifactPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vrp_json: str
    vrp: str
    meta: str
    manifest: str

    @field_validator("vrp_json", "vrp", "meta", "manifest")
    @classmethod
    def validate_relative_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class InstanceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authors: str
    generated_at: str
    problem_type: ProblemType
    metric_variant: MetricVariant
    place_slug: str
    source_base_name: str
    source_city: str
    source_seed: int
    source_folder: str
    num_vehicles_lb: int | None = None
    submodule_git_commit: str | None = None
    generator_version: str | None = None
    artifact_paths: ArtifactPaths
    sibling_variant_paths: dict[str, str] = Field(default_factory=dict)
    derived_problem_paths: dict[str, str] = Field(default_factory=dict)
    source_problem_paths: dict[str, str] = Field(default_factory=dict)
    license: str | None = None
    license_url: str | None = None

    @field_validator("sibling_variant_paths", "derived_problem_paths", "source_problem_paths")
    @classmethod
    def validate_path_map(cls, value: dict[str, str]) -> dict[str, str]:
        for path_value in value.values():
            _validate_relative_path(path_value)
        return value

    @field_validator("num_vehicles_lb")
    @classmethod
    def validate_num_vehicles_lb(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("num_vehicles_lb must be positive")
        return value


class ReferenceLLA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float
    lon: float
    alt: float = 0.0


class _InstanceValidationMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_customers: int
    num_vehicles: int | None = None
    vehicle_capacity: int
    coordinates: list[Coordinate]
    demands: list[int]
    depot: int = Field(default=0, ge=0)
    arc_costs: list[list[ArcCost]]
    reference_lla: ReferenceLLA | None = None

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

    @field_validator("coordinates", "demands")
    @classmethod
    def validate_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of {info.field_name} must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        return value

    @field_validator("arc_costs")
    @classmethod
    def validate_arc_costs(cls, value: list[list[ArcCost]], info: Any) -> list[list[ArcCost]]:
        expected_size = info.data["num_customers"] + 1
        if len(value) != expected_size:
            raise ValueError(
                f"arc_costs must have {expected_size} rows "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        for row in value:
            if len(row) != expected_size:
                raise ValueError(
                    f"Each row in arc_costs must have {expected_size} columns "
                    f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
                )
        return value


class BenchmarkInstance(_InstanceValidationMixin):
    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    service_times: list[int]
    time_windows: list[tuple[int, int]]
    metadata: InstanceMetadata | dict[str, Any] = Field(
        default_factory=dict,
        union_mode="left_to_right",
    )

    @field_validator("service_times", "time_windows")
    @classmethod
    def validate_vrptw_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of {info.field_name} must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        return value

    @classmethod
    def from_legacy_dict(cls, legacy_instance: dict[str, Any]) -> "BenchmarkInstance":
        if "arc_costs" in legacy_instance:
            raise ValueError("Legacy instance already contains 'arc_costs'")
        if "arc_travel_times" not in legacy_instance:
            raise ValueError("Legacy instance is missing required field 'arc_travel_times'")

        migrated = dict(legacy_instance)
        migrated["arc_costs"] = migrated.pop("arc_travel_times")
        return cls(**migrated)


class BenchmarkInstanceCVRP(_InstanceValidationMixin):
    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    metadata: InstanceMetadata


class BenchmarkInstanceTDVRP(_InstanceValidationMixin):
    """Time-Dependent VRP instance.

    arc_costs_time_dependent[h][i][j] is the travel time from node i to node j
    when departing during hour bin h. The horizon is num_time_bins * bin_seconds.
    arc_costs is the static fallback (typically the mean across bins) so naive
    consumers can treat the instance as a VRPTW.
    """

    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    service_times: list[int]
    time_windows: list[tuple[int, int]]
    arc_costs_time_dependent: list[list[list[ArcCost]]]
    num_time_bins: int = 24
    bin_seconds: int = 3600
    problem_type: ProblemType = ProblemType.TDVRP
    schema_version: str = INSTANCE_MODEL_SCHEMA_VERSION
    metadata: InstanceMetadata | dict[str, Any] = Field(
        default_factory=dict,
        union_mode="left_to_right",
    )

    @field_validator("service_times", "time_windows")
    @classmethod
    def validate_tdvrp_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of {info.field_name} must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        return value

    @field_validator("num_time_bins", "bin_seconds")
    @classmethod
    def validate_positive_time_params(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("arc_costs_time_dependent")
    @classmethod
    def validate_tdvrp_tensor(
        cls,
        value: list[list[list[ArcCost]]],
        info: Any,
    ) -> list[list[list[ArcCost]]]:
        expected_n = info.data["num_customers"] + 1
        num_time_bins = info.data.get("num_time_bins", 24)
        if len(value) != num_time_bins:
            raise ValueError(
                f"arc_costs_time_dependent must have num_time_bins={num_time_bins} entries "
                f"(got {len(value)})"
            )
        for h, layer in enumerate(value):
            if len(layer) != expected_n:
                raise ValueError(
                    f"arc_costs_time_dependent[{h}] must have {expected_n} rows "
                    f"(based on num_customers={info.data['num_customers']} + 1)"
                )
            for row in layer:
                if len(row) != expected_n:
                    raise ValueError(
                        f"arc_costs_time_dependent[{h}] rows must have {expected_n} columns"
                    )
        return value


class _SolutionValidationMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_name: str
    routes: list[list[int]]
    cost: int | float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("routes")
    @classmethod
    def validate_routes(cls, value: list[list[int]]) -> list[list[int]]:
        for route in value:
            if not route:
                raise ValueError("routes must not contain empty routes")
            if any(customer <= 0 for customer in route):
                raise ValueError("route customer ids must be positive")
            if len(set(route)) != len(route):
                raise ValueError("routes must be elementary")
        return value

    @property
    def num_routes(self) -> int:
        return len(self.routes)


class BenchmarkSolution(_SolutionValidationMixin):
    pass


class BenchmarkBKS(_SolutionValidationMixin):
    objective_function: ObjectiveFunction

    @classmethod
    def from_legacy_dict(
        cls,
        legacy_bks: dict[str, Any],
        objective_function: ObjectiveFunction,
    ) -> "BenchmarkBKS":
        migrated = dict(legacy_bks)
        migrated["objective_function"] = objective_function
        return cls(**migrated)

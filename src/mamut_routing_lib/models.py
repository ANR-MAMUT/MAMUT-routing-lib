from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mamut_routing_lib.enums import (
    BenchmarkName,
    InstanceOrigin,
    MetricVariant,
    ObjectiveFunction,
    ProblemType,
)
from mamut_routing_lib.sidecars import SidecarRef


Coordinate: TypeAlias = tuple[int | float, int | float]
ArcCost: TypeAlias = int | float


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


ARC_COSTS_DISTANCES_MODEL = "distances-sidecar"
ARC_COSTS_EUCLIDEAN_MODEL = "euclidean"


class ArcCostsDistancesRef(BaseModel):
    """Arc costs by sha-pinned reference to a collection distances sidecar."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["distances-sidecar"] = ARC_COSTS_DISTANCES_MODEL
    distances: SidecarRef


class ArcCostsEuclidean(BaseModel):
    """Arc costs materialized from the instance coordinates on load.

    The canonical definition is ``round(hypot(dx, dy), decimals)`` on the
    stored ENU coordinates (IEEE-754 doubles, Python round-half-to-even), so
    no sidecar is needed for the euclidean metric variant.
    """

    model_config = ConfigDict(extra="forbid")

    model: Literal["euclidean"] = ARC_COSTS_EUCLIDEAN_MODEL
    decimals: int = 3

    @field_validator("decimals")
    @classmethod
    def validate_decimals(cls, value: int) -> int:
        if value < 0:
            raise ValueError("decimals must be >= 0")
        return value


AnyArcCostsSource = Annotated[
    ArcCostsDistancesRef | ArcCostsEuclidean,
    Field(discriminator="model"),
]


class _SlimInstanceValidationMixin(BaseModel):
    """Static (CVRP/VRPTW) collection instance core: matrix by source, not embedded.

    Collection instances (Mamut2026 v2) reference their arc-cost matrix
    through ``arc_costs_source`` instead of embedding it; use
    ``mamut_routing_lib.artifacts.resolve_arc_costs`` to hydrate the matrix.
    ``metric_variant`` names the metric slot the instance is published under.
    """

    model_config = ConfigDict(extra="forbid")

    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    num_customers: int
    num_vehicles: int | None = None
    vehicle_capacity: int
    coordinates: list[Coordinate]
    demands: list[int]
    depot: int = Field(default=0, ge=0)
    reference_lla: ReferenceLLA | None = None
    metric_variant: MetricVariant
    arc_costs_source: AnyArcCostsSource
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


class BenchmarkInstanceCVRPCollection(_SlimInstanceValidationMixin):
    pass


class BenchmarkInstanceVRPTWCollection(_SlimInstanceValidationMixin):
    service_times: list[int | float]
    time_windows: list[tuple[int | float, int | float]]

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

    @field_validator("time_windows")
    @classmethod
    def validate_time_windows(cls, value: list[Any]) -> list[Any]:
        for index, (earliest, latest) in enumerate(value):
            if earliest > latest:
                raise ValueError(f"time window at index {index} has earliest > latest")
        return value


AnyCollectionStaticInstance = BenchmarkInstanceCVRPCollection | BenchmarkInstanceVRPTWCollection


class OptimalityMetadata(BaseModel):
    """Structured record of an optimality proof, stored under ``metadata["optimality"]``.

    The stamp asserts that the stored solution's cost is a proven optimum. The
    required fields identify the claim (who proved it, under which caveats,
    when); the optional fields carry the proof numerics and provenance. All
    free-text fields must be self-contained — meaningful to an external reader
    of the benchmark repository without access to any private context.
    """

    model_config = ConfigDict(extra="forbid")

    proven: Literal[True]
    prover: str
    certificate: str
    date: str
    arithmetic: str | None = None
    proven_optimum: int | float | None = None
    dual_bound: int | float | None = None
    wall_time_s: float | None = None
    time_limit_s: float | None = None
    checker: str | None = None
    campaign: str | None = None
    note: str | None = None


class _SolutionValidationMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_name: str
    routes: list[list[int]]
    cost: int | float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_optimality_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "optimality" in value:
            optimality = OptimalityMetadata.model_validate(value["optimality"])
            value = dict(value)
            value["optimality"] = optimality.model_dump(mode="json", exclude_none=True)
        return value

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

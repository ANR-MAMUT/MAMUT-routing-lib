"""Pydantic models of the static benchmark contract (CVRP, VRPTW).

Every artifact is validated on load with ``extra="forbid"``: an instance file
(``BenchmarkInstanceVRPTW`` for VRPTW, ``BenchmarkInstanceCVRP`` for CVRP, both
with an embedded ``(n+1) x (n+1)`` ``arc_costs`` matrix), a slim *collection*
instance (``BenchmarkInstanceCVRPCollection`` / ``BenchmarkInstanceVRPTWCollection``,
whose matrix comes from an ``arc_costs_source``: a sha-pinned distances
sidecar or the euclidean rule), a solution (``BenchmarkSolution``), a
best-known solution (``BenchmarkBKS``, a solution plus its objective) and
a solver claim of optimality (``SolverReportedOptimality``, validated inside
``metadata["optimality"]``). Node index 0 is the depot, and validation rejects
any other ``depot`` value and a nonzero depot service time. Customers are
``1..num_customers``. Time-dependent instances live
in ``mamut_routing_lib.td.models``.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

from mamut_routing_lib.enums import (
    BenchmarkName,
    InstanceOrigin,
    MetricVariant,
    ObjectiveFunction,
    ProblemType,
)
from mamut_routing_lib.sidecars import SidecarRef


NumericOperand: TypeAlias = StrictInt | StrictFloat
Coordinate: TypeAlias = tuple[NumericOperand, NumericOperand]
ArcCost: TypeAlias = NumericOperand
NonnegativeInteger: TypeAlias = Annotated[StrictInt, Field(ge=0)]


def require_depot_zero(value: int) -> int:
    """Validate the depot index on load. The exact profiles place the depot at vertex 0."""
    if value != 0:
        raise ValueError(f"depot must be vertex 0, got {value!r}")
    return value


def require_zero_depot_service(value: list[Any]) -> list[Any]:
    """Validate a service-time vector on load. The depot has no service duration."""
    if value and value[0] != 0:
        raise ValueError(f"depot service time must be zero, got {value[0]!r}")
    return value


def _validate_relative_path(path_value: str) -> str:
    if path_value.startswith("/"):
        raise ValueError("paths must be relative to the benchmark repository root")
    if not path_value:
        raise ValueError("paths must be non-empty")
    return path_value


class ArtifactPaths(BaseModel):
    """Repository-relative paths of the four artifacts of a generated instance (``.vrp.json``, ``.vrp``, meta, manifest)."""
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
    """Structured metadata of a generated (OSM-derived) instance.

    Historic families carry a free-form ``dict`` instead; ``artifacts.has_structured_metadata``
    tells the two apart. ``num_vehicles_lb`` is the bin-packing lower bound on the
    route count that the ``-k<K>`` name suffix advertises.
    """
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
    """Geodetic origin (latitude, longitude, altitude) of the local ENU frame the coordinates are expressed in."""
    model_config = ConfigDict(extra="forbid")

    lat: float
    lon: float
    alt: float = 0.0


class _InstanceValidationMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_customers: NonnegativeInteger
    num_vehicles: NonnegativeInteger | None = None
    vehicle_capacity: NonnegativeInteger
    coordinates: list[Coordinate]
    demands: list[NonnegativeInteger]
    depot: NonnegativeInteger = 0
    arc_costs: list[list[ArcCost]]
    reference_lla: ReferenceLLA | None = None

    @field_validator("depot")
    @classmethod
    def validate_depot(cls, value: int) -> int:
        return require_depot_zero(value)

    @field_validator("coordinates", "demands")
    @classmethod
    def validate_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        if "num_customers" not in info.data:
            return value
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
        if "num_customers" not in info.data:
            return value
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


class BenchmarkInstanceVRPTW(_InstanceValidationMixin):
    """A VRPTW instance with an embedded arc-cost matrix.

    Adds per-node ``service_times`` and ``time_windows`` (``[ready, due]``,
    depot included) to the CVRP core. Historic families (Sintef2008, Dimacs2021,
    Ortec2022) and hydrated collection instances use this model; service
    starts at or after ``ready`` and at or before ``due``.
    """
    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    service_times: list[NumericOperand]
    time_windows: list[tuple[NumericOperand, NumericOperand]]
    metadata: InstanceMetadata | dict[str, Any] = Field(
        default_factory=dict,
        union_mode="left_to_right",
    )

    @field_validator("service_times")
    @classmethod
    def validate_depot_service(cls, value: list[Any]) -> list[Any]:
        return require_zero_depot_service(value)

    @field_validator("service_times", "time_windows")
    @classmethod
    def validate_vrptw_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        if "num_customers" not in info.data:
            return value
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of {info.field_name} must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        return value

    @classmethod
    def from_legacy_dict(cls, legacy_instance: dict[str, Any]) -> "BenchmarkInstanceVRPTW":
        """Load a legacy VRPTW descriptor, renaming its arc_travel_times matrix to arc_costs."""
        if "arc_costs" in legacy_instance:
            raise ValueError("Legacy instance already contains 'arc_costs'")
        if "arc_travel_times" not in legacy_instance:
            raise ValueError("Legacy instance is missing required field 'arc_travel_times'")

        migrated = dict(legacy_instance)
        migrated["arc_costs"] = migrated.pop("arc_travel_times")
        return cls(**migrated)


class BenchmarkInstanceCVRP(_InstanceValidationMixin):
    """A CVRP instance with an embedded arc-cost matrix (no service times, no time windows)."""
    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    metadata: InstanceMetadata | dict[str, Any] = Field(
        default_factory=dict,
        union_mode="left_to_right",
    )


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
    decimals: StrictInt = 3

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

    Collection instances (Poryos2026 v2) reference their arc-cost matrix
    through ``arc_costs_source`` instead of embedding it; use
    ``mamut_routing_lib.artifacts.resolve_arc_costs`` to hydrate the matrix.
    ``metric_variant`` names the metric slot the instance is published under.
    """

    model_config = ConfigDict(extra="forbid")

    instance_name: str
    instance_origin: InstanceOrigin
    benchmark_name: BenchmarkName
    num_customers: NonnegativeInteger
    num_vehicles: NonnegativeInteger | None = None
    vehicle_capacity: NonnegativeInteger
    coordinates: list[Coordinate]
    demands: list[NonnegativeInteger]
    depot: NonnegativeInteger = 0
    reference_lla: ReferenceLLA | None = None
    metric_variant: MetricVariant
    arc_costs_source: AnyArcCostsSource
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("depot")
    @classmethod
    def validate_depot(cls, value: int) -> int:
        return require_depot_zero(value)

    @field_validator("coordinates", "demands")
    @classmethod
    def validate_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        if "num_customers" not in info.data:
            return value
        expected_length = info.data["num_customers"] + 1
        if len(value) != expected_length:
            raise ValueError(
                f"Length of {info.field_name} must be {expected_length} "
                f"(based on num_customers={info.data['num_customers']} + 1 for depot)"
            )
        return value


class BenchmarkInstanceCVRPCollection(_SlimInstanceValidationMixin):
    """Slim CVRP instance of a family-first collection: arc costs by ``arc_costs_source``, see ``artifacts.hydrate_collection_instance``."""
    pass


class BenchmarkInstanceVRPTWCollection(_SlimInstanceValidationMixin):
    """Slim VRPTW instance of a family-first collection (service times and windows may be fractional)."""
    service_times: list[NumericOperand]
    time_windows: list[tuple[NumericOperand, NumericOperand]]

    @field_validator("service_times")
    @classmethod
    def validate_depot_service(cls, value: list[Any]) -> list[Any]:
        return require_zero_depot_service(value)

    @field_validator("service_times", "time_windows")
    @classmethod
    def validate_vrptw_node_vector_lengths(cls, value: list[Any], info: Any) -> list[Any]:
        if "num_customers" not in info.data:
            return value
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


LoadedStaticInstance: TypeAlias = BenchmarkInstanceCVRP | BenchmarkInstanceVRPTW
AnyCollectionStaticInstance: TypeAlias = BenchmarkInstanceCVRPCollection | BenchmarkInstanceVRPTWCollection
AnyStaticBenchmarkInstance: TypeAlias = LoadedStaticInstance | AnyCollectionStaticInstance


class ClaimEvidence(BaseModel):
    """A file preserved as evidence of a solver claim, pinned by the sha256 of its bytes."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: Literal["file"] = "file"

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class SolverReportedOptimality(BaseModel):
    """An attributable solver self-report of optimality, stored under ``metadata["optimality"]``.

    ``target_profile`` names the evaluation profile under which the solver claims
    optimality. The store requires it to equal the profile of the evaluated family.
    ``reported_primal`` and ``reported_dual`` are the solver's own bounds, in its
    own arithmetic, and ``evidence`` pins a preserved file of the run. The checker
    verifies the routes but not the claim.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["solver-reported-optimal"]
    target_profile: str = Field(min_length=1)
    solver: str = Field(min_length=1)
    campaign: str | None = None
    reported_primal: NumericOperand | None = None
    reported_dual: NumericOperand | None = None
    evidence: ClaimEvidence | None = None

    @field_validator("reported_primal", "reported_dual")
    @classmethod
    def validate_bound(cls, value):
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("a reported bound must be finite")
        return value


LEGACY_OPTIMALITY_CLAIM_MESSAGE = (
    "metadata.optimality holds a claim in the format of mamut-routing-lib versions before 0.13 "
    "(a 'proven: true' stamp). This library does not read that format. "
    "Read such historical files with mamut-routing-lib 0.12.0."
)


class LegacyOptimalityClaimError(ValueError):
    """A BKS or solution file carries an optimality claim in the format of library versions before 0.13."""


def is_pre_013_optimality_claim(claim: Any) -> bool:
    """True for the ``proven: true`` stamp that library versions before 0.13 wrote."""
    return isinstance(claim, dict) and "proven" in claim and "status" not in claim


def reject_pre_013_optimality_claim(payload: Any, source: Any = None) -> None:
    """Raise ``LegacyOptimalityClaimError`` when ``payload["metadata"]["optimality"]`` has the pre-0.13 shape."""
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict) and is_pre_013_optimality_claim(metadata.get("optimality")):
        prefix = f"{source}: " if source is not None else ""
        raise LegacyOptimalityClaimError(prefix + LEGACY_OPTIMALITY_CLAIM_MESSAGE)


class _SolutionValidationMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_name: str
    routes: list[list[StrictInt]]
    cost: NumericOperand | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cost")
    @classmethod
    def validate_cost(cls, value):
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("display cost must be finite")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_optimality_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "solver_report" in value:
            from mamut_routing_lib.solvers.report import checked_solver_metadata

            value = checked_solver_metadata(value)
        if "optimality" in value:
            claim = value["optimality"]
            if is_pre_013_optimality_claim(claim):
                raise ValueError(LEGACY_OPTIMALITY_CLAIM_MESSAGE)
            optimality = SolverReportedOptimality.model_validate(claim)
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
    """A candidate solution: elementary routes of positive customer ids, optional cost.

    ``cost`` may be ``None`` (the checker prices the routes) or the exact expected
    value (the checker then reports ``objective_value_mismatch`` on any deviation).
    """
    pass


class BenchmarkBKS(_SolutionValidationMixin):
    """A best-known solution: a solution plus the ``objective_function`` it is best under.

    Stored as ``<base>.bks.<ObjectiveFunction>.json`` next to the instance and
    written only through ``mamut_routing_lib.bks`` (checker cost, mandatory
    ``metadata.authors``, replace-only-if-strictly-better).
    """
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


if TYPE_CHECKING:
    BenchmarkInstance = BenchmarkInstanceVRPTW


def _deprecated_model_name(module: str, name: str):
    if name == "BenchmarkInstance":
        from warnings import warn

        warn("BenchmarkInstance is deprecated in v0.13. Use BenchmarkInstanceVRPTW.",
             DeprecationWarning, stacklevel=3)
        return BenchmarkInstanceVRPTW
    raise AttributeError(f"module {module!r} has no attribute {name!r}")


def __getattr__(name: str):
    return _deprecated_model_name(__name__, name)

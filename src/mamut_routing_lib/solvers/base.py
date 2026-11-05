"""Solver-agnostic types shared by every solver integration.

Nothing here imports a solver: PyVRP, KAYROS and external executables plug in
through the :class:`Solver` protocol and are looked up by name in
:mod:`mamut_routing_lib.solvers.registry`. The contract, in three parts:

- **Capabilities** (:class:`SolverCapabilities`) say what a solver can take:
  problem types and objectives, symmetric or asymmetric costs, coordinates or
  an explicit matrix, a fixed fleet, warm starts, live progress.
- **A request** (:class:`SolveRequest`) carries one instance, its path, the
  objective, the wall-clock budget, the seed and optional warm-start routes.
- **A result** (:class:`SolveResult`) carries the routes in library ids (depot
  0, customers 1..n), a status, the solver's own cost brought back to instance
  units, and provenance. The solver's cost is informational: callers re-price
  every route set with the reference checker (``solvers.run.run_solver``).

``SolveMonitor`` is the one progress/cancellation hook for all drivers: PyVRP
calls it per iteration as a stopping criterion, the subprocess runner polls it,
and KAYROS's incumbent hook feeds it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mamut_routing_lib.artifacts import AnyBenchmarkInstance

__all__ = [
    "INFEASIBLE_COST",
    "Availability",
    "InstanceNotSupportedError",
    "InstanceProfile",
    "SolveMonitor",
    "SolveRequest",
    "SolveResult",
    "SolveStatus",
    "SolveTick",
    "Solver",
    "SolverCapabilities",
    "SolverUnavailableError",
    "Support",
    "capability_support",
    "profile_instance",
]

#: PyVRP evaluates an infeasible incumbent at INT64_MAX (see ``pyvrp.stop.FirstFeasible``), so any
#: cost at or above this is "nothing feasible found yet" rather than an astronomically bad tour.
INFEASIBLE_COST = float(2**63 - 1)

SolveStatus = Literal["feasible", "infeasible", "timeout", "cancelled", "failed"]


class SolverUnavailableError(RuntimeError):
    """The solver cannot run here: missing Python extra, executable not found, bad configuration."""


class InstanceNotSupportedError(ValueError):
    """The solver cannot take this instance/objective (see ``Support.reasons``).

    Distinct from ``cvrplib.UnsupportedInstanceError``, which is about classic export formats.
    """


@dataclass(frozen=True)
class SolveTick:
    """One observation of a running solve."""

    phase: str
    #: Search iterations so far; ``None`` for solvers that do not expose them (external executables).
    iterations: int | None
    elapsed_s: float
    best_cost: float | None
    budget_s: float | None = None


class SolveMonitor:
    """Progress and cancellation hook shared by every solver driver.

    - ``observe(best_cost, iterations=...)`` is the generic entry point: it checks for a stop request
      on every call, records the trajectory, and relays a throttled :class:`SolveTick` to ``on_tick``.
      It returns True when the solve must stop (``on_tick`` returned True or ``request_stop`` was called).
    - ``__call__(best_cost)`` is the PyVRP stopping-criterion protocol: PyVRP calls it once per
      iteration with the incumbent's cost. It counts the iteration, maps PyVRP's infeasible sentinel to
      ``None`` and delegates to ``observe``. Compose it *before* ``MaxRuntime`` in a
      ``MultipleCriteria``: that class uses ``any()``, which short-circuits, so a monitor placed second
      would go silent the moment the budget expires.

    ``best_cost`` is in solver units; ``rescale`` sets the factor that brings it back to instance units.
    """

    def __init__(
        self,
        on_tick: Callable[[SolveTick], bool] | None = None,
        *,
        scale: float = 1,
        interval_s: float = 1.0,
        budget_s: float | None = None,
        record_trajectory: bool = False,
        trajectory_min_gap_s: float = 0.5,
    ):
        self._on_tick = on_tick
        self._scale = scale if scale and scale > 0 else 1
        self._interval_s = max(0.0, interval_s)
        self._budget_s = budget_s
        self._phase = "search"
        self._iterations = 0
        self._start = time.perf_counter()
        # -inf, not the start time: the first observation should report immediately rather than
        # leave the caller showing nothing for a whole interval.
        self._last_tick = float("-inf")
        self._stop = threading.Event()
        self._record = record_trajectory
        self._trajectory_gap = max(0.0, trajectory_min_gap_s)
        self.trajectory: list[tuple[float, float]] = []
        self.stopped_early = False
        self.best_cost: float | None = None

    # -- configuration -------------------------------------------------------------------------

    def phase(self, name: str) -> None:
        """Name the stage the solve has just entered, and tick on the next observation so the label lands."""
        self._phase = name
        self._last_tick = float("-inf")
        # Phases can optimise different objectives (PyVRP's hierarchical solve adds a fleet cost in
        # its last phase), so the running best restarts with each phase.
        self.best_cost = None

    @property
    def current_phase(self) -> str:
        return self._phase

    def rescale(self, scale: float) -> None:
        """Adopt the solve's cost scale, which is only known once the instance is converted."""
        self._scale = scale if scale and scale > 0 else 1

    def set_budget(self, budget_s: float | None) -> None:
        self._budget_s = budget_s

    def restart_clock(self) -> None:
        """Measure elapsed time from now (the monitor may be built well before the solve starts)."""
        self._start = time.perf_counter()
        self._last_tick = float("-inf")

    # -- state ---------------------------------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        """Ask the solve to stop at its next observation (thread-safe)."""
        self._stop.set()

    # -- observation ---------------------------------------------------------------------------

    def observe(
        self,
        best_cost: float | None = None,
        *,
        iterations: int | None = None,
        force: bool = False,
    ) -> bool:
        """Record one observation; True means the solve must stop.

        ``best_cost`` is the incumbent in solver units (``None`` when nothing feasible is known);
        ``iterations`` overrides the internal counter for drivers that report their own.
        """
        if iterations is not None:
            self._iterations = iterations
        now = time.perf_counter()
        elapsed = now - self._start
        real_cost = None
        if best_cost is not None and best_cost < INFEASIBLE_COST:
            real_cost = best_cost / self._scale
            if self.best_cost is None or real_cost < self.best_cost:
                self.best_cost = real_cost
                if self._record:
                    self._record_point(elapsed, real_cost)
        if self._stop.is_set():
            self.stopped_early = True
            return True
        if not force and now - self._last_tick < self._interval_s:
            return False
        self._last_tick = now
        if self._on_tick is None:
            return False
        tick = SolveTick(
            phase=self._phase,
            iterations=self._iterations if (iterations is not None or self._iterations) else None,
            elapsed_s=elapsed,
            best_cost=real_cost if real_cost is not None else self.best_cost,
            budget_s=self._budget_s,
        )
        if self._on_tick(tick):
            self._stop.set()
            self.stopped_early = True
            return True
        return False

    def __call__(self, best_cost: float) -> bool:
        """PyVRP stopping criterion: one call per search iteration."""
        self._iterations += 1
        return self.observe(best_cost, iterations=self._iterations)

    def _record_point(self, elapsed: float, cost: float) -> None:
        if self.trajectory and elapsed - self.trajectory[-1][0] < self._trajectory_gap:
            self.trajectory[-1] = (round(elapsed, 3), cost)
        else:
            self.trajectory.append((round(elapsed, 3), cost))


@dataclass(frozen=True)
class SolverCapabilities:
    """What a solver can take. ``objectives`` maps each supported problem type to its objectives."""

    objectives: Mapping[ProblemType, frozenset[ObjectiveFunction]]
    #: ``model``: the solver reads the library model in-process; ``coordinates``: it computes Euclidean
    #: distances from node coordinates (only faithful when the coordinates define the arc costs);
    #: ``explicit``: it reads a full cost matrix.
    input: Literal["model", "coordinates", "explicit"] = "model"
    symmetric_only: bool = False
    integer_costs: bool = False
    #: Whether the solver honours a fixed fleet (``num_vehicles``). If not, fixed-fleet instances get a
    #: support warning (the checker still rejects a solution with too many routes).
    fleet_limit: bool = True
    warm_start: bool = False
    live_progress: Literal["none", "elapsed", "incumbent"] = "none"
    cancellable: bool = True
    in_process: bool = False

    @property
    def problem_types(self) -> frozenset[ProblemType]:
        return frozenset(self.objectives)

    @property
    def time_dependent(self) -> bool:
        return bool(self.problem_types & {ProblemType.TDVRP, ProblemType.TDVRPTW})

    def to_json(self) -> dict[str, Any]:
        return {
            "objectives": {problem.value: sorted(obj.value for obj in objs) for problem, objs in self.objectives.items()},
            "input": self.input,
            "symmetric_only": self.symmetric_only,
            "integer_costs": self.integer_costs,
            "fleet_limit": self.fleet_limit,
            "warm_start": self.warm_start,
            "live_progress": self.live_progress,
            "cancellable": self.cancellable,
            "in_process": self.in_process,
            "time_dependent": self.time_dependent,
        }


@dataclass(frozen=True)
class Availability:
    """Whether a solver can run here (``reason`` explains a ``False``).

    ``hint`` says how to make it available (install, configure); ``note`` is a standing
    requirement worth knowing either way (a patched build, for instance).
    """

    available: bool
    reason: str | None = None
    version: str | None = None
    executable: str | None = None
    hint: str | None = None
    note: str | None = None

    def describe(self) -> str:
        """``reason (hint). note`` -- one line for an error message."""
        text = self.reason or ("available" if self.available else "not available")
        if self.hint:
            text = f"{text} ({self.hint})"
        if self.note:
            text = f"{text}. {self.note[0].upper()}{self.note[1:]}"
        return text

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "version": self.version,
            "executable": self.executable,
            "hint": self.hint,
            "note": self.note,
        }


@dataclass(frozen=True)
class Support:
    """Whether a solver can take one instance under one objective; ``warnings`` never block."""

    supported: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def ok(cls, *warnings: str) -> "Support":
        return cls(True, (), tuple(warnings))

    @classmethod
    def refuse(cls, *reasons: str) -> "Support":
        return cls(False, tuple(reasons), ())

    def to_json(self) -> dict[str, Any]:
        return {"supported": self.supported, "reasons": list(self.reasons), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class InstanceProfile:
    """Cheap facts about an instance, enough to decide solver support without solving.

    ``symmetric`` is ``True`` for the euclidean metric (symmetric by construction), ``False`` or
    ``True`` when a matrix was inspected, and ``None`` when unknown; solvers that need symmetry check
    the matrix at solve time.
    """

    problem_type: ProblemType
    benchmark_name: str | None
    metric_variant: str | None
    num_customers: int
    depot: int
    fixed_fleet: int | None
    has_coordinates: bool
    coordinates_define_costs: bool
    symmetric: bool | None


def profile_instance(
    instance: Any,
    instance_path: str | Path | None = None,
    *,
    inspect_matrix: bool = False,
) -> InstanceProfile:
    """Profile an instance (static or time-dependent) for ``Solver.supports``.

    With ``inspect_matrix`` the arc costs are hydrated and checked for symmetry and for being a
    rounding of the coordinates' Euclidean distances (O(n²); done at solve time, not for listings).
    """
    from mamut_routing_lib.artifacts import instance_problem_type, is_collection_instance

    problem_type = instance_problem_type(instance)
    metadata = getattr(instance, "metadata", None)
    metric = getattr(instance, "metric_variant", None)
    if metric is None:
        if isinstance(metadata, Mapping):
            metric = metadata.get("metric_variant")
        else:
            metric = getattr(metadata, "metric_variant", None)
    metric_value = getattr(metric, "value", metric)
    benchmark = getattr(instance, "benchmark_name", None)
    benchmark_value = getattr(benchmark, "value", benchmark)
    coordinates = getattr(instance, "coordinates", None)
    time_dependent = problem_type in (ProblemType.TDVRP, ProblemType.TDVRPTW)

    symmetric: bool | None = True if metric_value == "euclidean" else None
    coords_define = False
    if not time_dependent:
        source = getattr(instance, "arc_costs_source", None)
        if is_collection_instance(instance) and getattr(source, "model", None) == "euclidean":
            coords_define = True
        elif inspect_matrix or not is_collection_instance(instance):
            try:
                from mamut_routing_lib.artifacts import resolve_arc_costs
                from mamut_routing_lib.cvrplib import coordinates_define_arc_costs, is_symmetric_matrix

                if is_collection_instance(instance):
                    if instance_path is None:
                        raise ValueError("collection instance without a path")
                    matrix = resolve_arc_costs(instance, instance_path)
                else:
                    matrix = instance.arc_costs
                coords_define = coordinates_define_arc_costs(instance, matrix)
                if inspect_matrix or not is_collection_instance(instance):
                    symmetric = is_symmetric_matrix(matrix)
            except (ValueError, OSError, FileNotFoundError):
                pass
    return InstanceProfile(
        problem_type=problem_type,
        benchmark_name=benchmark_value,
        metric_variant=metric_value,
        num_customers=int(getattr(instance, "num_customers", 0) or 0),
        depot=int(getattr(instance, "depot", 0) or 0),
        fixed_fleet=getattr(instance, "num_vehicles", None),
        has_coordinates=bool(coordinates),
        coordinates_define_costs=coords_define,
        symmetric=symmetric,
    )


@dataclass(frozen=True)
class SolveRequest:
    """One solve: the instance (already loaded), where it lives, and the run parameters."""

    instance: Any
    instance_path: Path | None
    objective: ObjectiveFunction
    time_limit_s: float
    seed: int = 42
    initial_routes: list[list[int]] | None = None
    #: Directory for solver input/output files; a temporary one is used (and removed) when ``None``.
    workdir: Path | None = None
    keep_artifacts: bool = False
    display: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveResult:
    """What a solver returned, before (and independent of) the reference checker's verdict.

    ``solver_cost`` is the solver's own objective value brought back to instance units (informational:
    the checker's cost is authoritative); ``solver_cost_raw`` is the value as the solver reported it,
    in its own (scaled) units; ``cost_scale`` is the factor between the two; ``cost_exact`` says whether
    the solver worked on exactly the published costs (``False``: rounded or approximated input).
    """

    solver: str
    method: str
    problem_type: str
    objective_function: str
    instance_id: str
    seed: int
    time_limit_s: float
    wall_time: float
    status: SolveStatus
    routes: list[list[int]] = field(default_factory=list)
    solver_cost: float | None = None
    solver_cost_raw: int | float | None = None
    cost_scale: float = 1.0
    cost_exact: bool | None = None
    stopped_early: bool = False
    error: str | None = None
    solver_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def has_routes(self) -> bool:
        return bool(self.routes)

    def to_json(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "method": self.method,
            "problem_type": self.problem_type,
            "objective_function": self.objective_function,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "time_limit_s": self.time_limit_s,
            "wall_time": self.wall_time,
            "status": self.status,
            "routes": self.routes,
            "solver_cost": self.solver_cost,
            "solver_cost_raw": self.solver_cost_raw,
            "cost_scale": self.cost_scale,
            "cost_exact": self.cost_exact,
            "stopped_early": self.stopped_early,
            "error": self.error,
            "solver_version": self.solver_version,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


@runtime_checkable
class Solver(Protocol):
    """A solver integration. ``solve`` never raises for a solver failure (timeout, crash, garbage
    output): it returns a ``SolveResult`` with ``status`` ``failed`` or ``timeout``. It raises
    ``SolverUnavailableError`` / ``InstanceNotSupportedError`` only for usage errors."""

    name: str
    label: str
    capabilities: SolverCapabilities

    def availability(self) -> Availability: ...

    def supports(self, profile: InstanceProfile, objective: ObjectiveFunction) -> Support: ...

    def solve(self, request: SolveRequest, monitor: SolveMonitor | None = None) -> SolveResult: ...


def capability_support(
    capabilities: SolverCapabilities,
    profile: InstanceProfile,
    objective: ObjectiveFunction,
    *,
    solver_label: str,
) -> Support:
    """The checks every solver shares: problem type, objective, symmetry, coordinates, fleet."""
    objectives = capabilities.objectives.get(profile.problem_type)
    if objectives is None:
        supported = ", ".join(sorted(problem.value for problem in capabilities.problem_types)) or "nothing"
        return Support.refuse(f"{solver_label} solves {supported}, not {profile.problem_type.value}")
    if objective not in objectives:
        names = ", ".join(sorted(obj.value for obj in objectives))
        return Support.refuse(
            f"{solver_label} optimises {names} on {profile.problem_type.value}, not {objective.value}"
        )
    reasons: list[str] = []
    warnings: list[str] = []
    if capabilities.symmetric_only and profile.symmetric is False:
        reasons.append(f"{solver_label} needs symmetric arc costs; this {profile.metric_variant or ''} variant is asymmetric".replace("  ", " "))
    if capabilities.input == "coordinates":
        if not profile.has_coordinates:
            reasons.append(f"{solver_label} reads coordinates, and this instance has none")
        elif not profile.coordinates_define_costs:
            reasons.append(
                f"{solver_label} computes Euclidean distances from coordinates, which do not define this "
                f"instance's arc costs (metric {profile.metric_variant or 'unknown'})"
            )
        if profile.depot != 0:
            reasons.append(f"{solver_label} needs the depot at index 0 (it is {profile.depot})")
    if capabilities.input == "explicit" and profile.depot != 0:
        reasons.append(f"{solver_label} needs the depot at index 0 (it is {profile.depot})")
    if profile.fixed_fleet is not None and not capabilities.fleet_limit:
        warnings.append(
            f"{solver_label} ignores the fixed fleet of {profile.fixed_fleet}; the checker rejects a solution with more routes"
        )
    if reasons:
        return Support(False, tuple(reasons), tuple(warnings))
    return Support(True, (), tuple(warnings))

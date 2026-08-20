"""PyVRP-backed solver integration for `mamut-routing-lib`.

Importing this module requires the `[pyvrp]` optional dependency:

    pip install "mamut-routing-lib[pyvrp]"

Public API:
    to_vrplib_dict, to_pyvrp_problem            -- conversion helpers
    MethodResult                                 -- solver result dataclass
    SolveTick, SolveMonitor                      -- live progress / cancellation hook
    solve_cvrp, solve_vrptw, solve_instance      -- single-instance solvers
    solve_and_update_bks                         -- solve + write BKS if improved
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
    from pyvrp import ProblemData, SolveParams
    from pyvrp.minimise_fleet import minimise_fleet
    from pyvrp.read import ROUND_FUNCS, _InstanceParser, _ProblemDataBuilder, _RoundingFunc
    from pyvrp.solve import solve as pyvrp_solve
    from pyvrp.stop import FirstFeasible, MaxRuntime, MultipleCriteria
except ImportError as exc:  # pragma: no cover - exercised via test_cli_solve missing-extra path
    raise ImportError(
        "mamut-routing-lib's PyVRP solver module requires the [pyvrp] optional "
        "dependency. Install with: pip install 'mamut-routing-lib[pyvrp]'"
    ) from exc

from mamut_routing_lib.artifacts import AnyBenchmarkInstance, get_instance_identifier, resolve_arc_costs
from mamut_routing_lib.bks import (
    BKSUpdateResult,
    DEFAULT_BKS_AUTHORS,
    create_bks_from_solution,
    save_bks_if_improved,
)
from mamut_routing_lib.checker import check_solution
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import (
    BenchmarkInstance,
    BenchmarkInstanceCVRP,
    BenchmarkInstanceCVRPCollection,
    BenchmarkInstanceVRPTWCollection,
    BenchmarkSolution,
)


__all__ = [
    "MethodResult",
    "SolveMonitor",
    "SolveTick",
    "hydrate_collection_instance",
    "solve_and_update_bks",
    "solve_cvrp",
    "solve_instance",
    "solve_vrptw",
    "to_pyvrp_problem",
    "to_vrplib_dict",
]


# Tuning constants for the hierarchical-vehicle-cost two-phase VRPTW solve.
_FLEET_MIN_TIME_FRACTION = 0.3
_WARM_START_MAX_TIME = 5.0

# PyVRP evaluates an infeasible incumbent at INT_MAX (see pyvrp.stop.FirstFeasible), so any
# cost at or above this is "nothing feasible found yet" rather than an astronomically bad tour.
_INFEASIBLE_COST = float(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class SolveTick:
    """One observation of a running solve, as seen from inside the search loop."""

    phase: str
    iterations: int
    elapsed_s: float
    best_cost: float | None


class SolveMonitor:
    """A PyVRP stopping criterion that reports the incumbent and honours cancellation.

    PyVRP calls the stopping criterion once per iteration with the cost of the best solution
    so far, which is the only point inside a solve where a caller can learn that anything is
    happening. This criterion never stops on its own -- the `MaxRuntime` it is paired with
    still owns the time budget. It relays throttled ticks to `on_tick`, and stops the search
    only when `on_tick` returns True.

    Compose it *before* `MaxRuntime` in a `MultipleCriteria`: that class uses `any()`, which
    short-circuits, so a monitor placed second would go silent the moment the budget expires.
    """

    def __init__(
        self,
        on_tick: Callable[[SolveTick], bool],
        *,
        scale: int = 1,
        interval_s: float = 1.0,
    ):
        self._on_tick = on_tick
        self._scale = scale if scale > 0 else 1
        self._interval_s = max(0.0, interval_s)
        self._phase = "search"
        self._iterations = 0
        self._start = time.perf_counter()
        # -inf, not the start time: the first iteration should report immediately rather than
        # leave the caller showing nothing for a whole interval.
        self._last_tick = float("-inf")
        self.stopped_early = False

    def phase(self, name: str) -> None:
        """Name the stage the solve has just entered, and tick immediately so the label lands."""
        self._phase = name
        self._last_tick = float("-inf")  # tick on the next iteration so the label lands

    def rescale(self, scale: int) -> None:
        """Adopt the solve's cost scale, which is only known once the instance is converted."""
        self._scale = scale if scale > 0 else 1

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    def __call__(self, best_cost: float) -> bool:
        self._iterations += 1
        now = time.perf_counter()
        if now - self._last_tick < self._interval_s:
            return False
        self._last_tick = now
        tick = SolveTick(
            phase=self._phase,
            iterations=self._iterations,
            elapsed_s=now - self._start,
            best_cost=None if best_cost >= _INFEASIBLE_COST else best_cost / self._scale,
        )
        if self._on_tick(tick):
            self.stopped_early = True
            return True
        return False


def _stop_with(monitor: SolveMonitor | None, *criteria: Any) -> Any:
    """The given criteria, with the monitor in front of them when there is one."""
    if monitor is None:
        return criteria[0] if len(criteria) == 1 else MultipleCriteria(list(criteria))
    return MultipleCriteria([monitor, *criteria])


@dataclass(frozen=True)
class MethodResult:
    method: str
    problem_type: str
    objective_function: str
    instance_id: str
    seed: int
    time_limit_s: int
    wall_time: float
    solver_is_feasible: bool
    routes: list[list[int]] = field(default_factory=list)
    solver_cost: int | float | None = None
    route_count: int = 0
    vehicle_penalty: int | float = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_vrptw_like(instance: AnyBenchmarkInstance) -> bool:
    return isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTWCollection))


def _is_collection(instance: AnyBenchmarkInstance) -> bool:
    return isinstance(instance, (BenchmarkInstanceCVRPCollection, BenchmarkInstanceVRPTWCollection))


def _hydrated_arc_costs(
    instance: AnyBenchmarkInstance,
    instance_path: str | Path | None,
) -> list[list[int]] | list[list[float]]:
    """Inline arc costs, or the sidecar/euclidean-resolved matrix for slim
    collection instances (which need ``instance_path`` for sidecar sources)."""
    if not _is_collection(instance):
        return instance.arc_costs
    if instance_path is None:
        raise ValueError(
            "Collection instances resolve arc costs from their sidecars; pass instance_path to solve them."
        )
    return resolve_arc_costs(instance, instance_path)


def hydrate_collection_instance(
    instance: AnyBenchmarkInstance,
    instance_path: str | Path | None = None,
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> AnyBenchmarkInstance:
    """A v1 embedded-model view of a slim collection instance with resolved
    arc costs, usable with ``mamut_routing_lib.checker.check_solution``.
    Non-collection instances pass through unchanged."""
    if not _is_collection(instance):
        return instance
    if arc_costs is None:
        arc_costs = _hydrated_arc_costs(instance, instance_path)
    common: dict[str, Any] = {
        "instance_name": instance.instance_name,
        "instance_origin": instance.instance_origin,
        "benchmark_name": instance.benchmark_name,
        "num_customers": instance.num_customers,
        "num_vehicles": instance.num_vehicles,
        "vehicle_capacity": instance.vehicle_capacity,
        "coordinates": instance.coordinates,
        "demands": instance.demands,
        "depot": instance.depot,
        "arc_costs": arc_costs,
        "reference_lla": instance.reference_lla,
        "metadata": dict(instance.metadata),
    }
    if isinstance(instance, BenchmarkInstanceVRPTWCollection):
        service_times = [int(value) for value in instance.service_times]
        time_windows = [(int(ready), int(due)) for ready, due in instance.time_windows]
        if service_times != [float(v) for v in instance.service_times] or any(
            (float(int(r)), float(int(d))) != (float(r), float(d)) for r, d in instance.time_windows
        ):
            raise ValueError("Non-integer time windows/service times cannot hydrate into the v1 checker model")
        return BenchmarkInstance(**common, service_times=service_times, time_windows=time_windows)
    return BenchmarkInstanceCVRP(**common)


def to_vrplib_dict(
    instance: AnyBenchmarkInstance,
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> dict[str, Any]:
    """Convert a benchmark instance into a VRPLIB-shaped dict consumable by PyVRP's parser."""
    if arc_costs is None:
        arc_costs = instance.arc_costs
    weights = np.asarray(arc_costs)
    if np.allclose(weights, np.round(weights)):
        weights = weights.astype(int)
    payload: dict[str, Any] = {
        "name": get_instance_identifier(instance),
        "type": "CVRPTW" if _is_vrptw_like(instance) else "CVRP",
        "dimension": instance.num_customers + 1,
        "vehicles": instance.num_vehicles if instance.num_vehicles is not None else instance.num_customers,
        "capacity": instance.vehicle_capacity,
        "edge_weight_type": "EXPLICIT",
        "edge_weight_format": "FULL_MATRIX",
        "node_coord": np.asarray(instance.coordinates, dtype=float),
        "demand": np.asarray(instance.demands, dtype=int),
        "depot": np.array([instance.depot], dtype=int),
        "edge_weight": weights,
    }

    if _is_vrptw_like(instance):
        payload["time_window"] = np.asarray(instance.time_windows)
        payload["service_time"] = np.asarray(instance.service_times)

    return payload


def to_pyvrp_problem(
    instance: AnyBenchmarkInstance,
    round_func: str | _RoundingFunc = "none",
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> ProblemData:
    """Convert a benchmark instance into a PyVRP `ProblemData`."""
    if (key := str(round_func)) in ROUND_FUNCS:
        round_func = ROUND_FUNCS[key]

    if not callable(round_func):
        raise TypeError(
            f"round_func = {round_func} is not understood. Can be a function, "
            f"or one of {list(ROUND_FUNCS.keys())}."
        )

    parser = _InstanceParser(to_vrplib_dict(instance, arc_costs), round_func)
    builder = _ProblemDataBuilder(parser)
    return builder.data()


def _extract_routes(pyvrp_solution: Any) -> list[list[int]]:
    routes: list[list[int]] = []
    for route in pyvrp_solution.routes():
        route_customers = [int(customer) for customer in route]
        if route_customers:
            routes.append(route_customers)
    return routes


def _get_vehicle_fixed_cost(
    num_customers: int,
    arc_costs: list[list[int]] | list[list[float]],
    scale: int = 1,
) -> int:
    max_arc_cost = max(max(row) for row in arc_costs)
    return int(2 * (num_customers + 1) * max_arc_cost * scale) + 1


def _build_method_metadata(
    instance: AnyBenchmarkInstance,
    objective_function: ObjectiveFunction,
    *,
    seed: int,
    time_limit_s: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "seed": seed,
        "time_limit_s": time_limit_s,
        "objective_function": objective_function.value,
        "instance_id": get_instance_identifier(instance),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _conversion_plan(
    instance: AnyBenchmarkInstance,
    instance_path: str | Path | None,
) -> tuple[list[list[int]] | list[list[float]], str, int]:
    """(arc costs, round_func, cost scale) for the solve. Slim collection
    instances carry 3-decimal float costs; PyVRP's 'exact' rounding scales
    them x1000 to integers, so reported solver costs are in milli-units."""
    arc_costs = _hydrated_arc_costs(instance, instance_path)
    weights = np.asarray(arc_costs)
    if np.allclose(weights, np.round(weights)):
        return arc_costs, "none", 1
    return arc_costs, "exact", 1000


def solve_cvrp(
    instance: BenchmarkInstanceCVRP | BenchmarkInstanceCVRPCollection,
    *,
    time_limit_s: int,
    seed: int = 42,
    display: bool = False,
    instance_path: str | Path | None = None,
    monitor: SolveMonitor | None = None,
) -> MethodResult:
    """Solve a CVRP instance with PyVRP's ILS (mono-cost objective only)."""
    start_time = time.perf_counter()
    arc_costs, round_func, scale = _conversion_plan(instance, instance_path)
    if monitor is not None:
        monitor.rescale(scale)
    problem = to_pyvrp_problem(instance, round_func=round_func, arc_costs=arc_costs)
    result = pyvrp_solve(
        problem,
        stop=_stop_with(monitor, MaxRuntime(max(1, time_limit_s))),
        seed=seed,
        collect_stats=False,
        display=display,
    )
    wall_time = time.perf_counter() - start_time
    is_feasible = result.is_feasible()
    routes = _extract_routes(result.best) if is_feasible else []
    return MethodResult(
        method="pyvrp-ils-v1",
        problem_type="CVRP",
        objective_function=ObjectiveFunction.MONO_COST.value,
        instance_id=instance.instance_name,
        seed=seed,
        time_limit_s=time_limit_s,
        wall_time=wall_time,
        solver_is_feasible=is_feasible,
        routes=routes,
        solver_cost=result.cost(),
        route_count=len(routes),
        vehicle_penalty=0,
        metadata=_build_method_metadata(
            instance,
            ObjectiveFunction.MONO_COST,
            seed=seed,
            time_limit_s=time_limit_s,
            extra={"arc_cost_scale": scale} if scale != 1 else None,
        ),
    )


def solve_vrptw(
    instance: BenchmarkInstance | BenchmarkInstanceVRPTWCollection,
    *,
    objective_function: ObjectiveFunction,
    time_limit_s: int,
    seed: int = 42,
    display: bool = False,
    instance_path: str | Path | None = None,
    monitor: SolveMonitor | None = None,
) -> MethodResult:
    """Solve a VRPTW instance with PyVRP's ILS.

    For ObjectiveFunction.MONO_COST, runs a single solve with the full time budget.
    For ObjectiveFunction.HIERARCHICAL_VEHICLE_COST, runs a two-phase solve:
      1. Fleet minimization (PyVRP's `minimise_fleet`) within ~30% of the budget.
      2. Cost optimization on the constrained fleet for the remaining budget,
         with a per-vehicle fixed cost large enough to dominate routing cost.
         Phase 2 is warm-started from a feasible solution re-found on the
         constrained fleet: the fixed cost dwarfs the solver's capped
         infeasibility penalties, so a cold start can otherwise stabilize on
         infeasible solutions that use fewer vehicles.
    """
    start_time = time.perf_counter()
    arc_costs, round_func, scale = _conversion_plan(instance, instance_path)
    if monitor is not None:
        monitor.rescale(scale)
    problem = to_pyvrp_problem(instance, round_func=round_func, arc_costs=arc_costs)
    params = SolveParams()
    vehicle_penalty: int | float = 0

    if objective_function == ObjectiveFunction.MONO_COST:
        result = pyvrp_solve(
            problem,
            stop=_stop_with(monitor, MaxRuntime(max(1, time_limit_s))),
            seed=seed,
            collect_stats=False,
            display=display,
            params=params,
        )
    else:
        fleet_budget = max(1.0, float(time_limit_s) * _FLEET_MIN_TIME_FRACTION)
        if monitor is not None:
            monitor.phase("minimising fleet")
        constrained_vehicle_type = minimise_fleet(
            problem,
            stop=_stop_with(monitor, MaxRuntime(fleet_budget)),
            seed=seed,
            params=params,
        )
        fleet_problem = problem.replace(vehicle_types=[constrained_vehicle_type])
        if monitor is not None:
            monitor.phase("warm start")
        warm_result = pyvrp_solve(
            fleet_problem,
            stop=_stop_with(monitor, FirstFeasible(), MaxRuntime(_WARM_START_MAX_TIME)),
            seed=seed,
            collect_stats=False,
            display=False,
            params=params,
        )
        vehicle_penalty = _get_vehicle_fixed_cost(instance.num_customers, arc_costs, scale)
        constrained_problem = problem.replace(
            vehicle_types=[constrained_vehicle_type.replace(fixed_cost=vehicle_penalty)]
        )
        remaining_time = max(1, time_limit_s - int(time.perf_counter() - start_time))
        if monitor is not None:
            monitor.phase("optimising cost")
        result = pyvrp_solve(
            constrained_problem,
            stop=_stop_with(monitor, MaxRuntime(remaining_time)),
            seed=seed,
            collect_stats=False,
            display=display,
            params=params,
            initial_solution=warm_result.best if warm_result.is_feasible() else None,
        )
        if not result.is_feasible() and warm_result.is_feasible():
            # The warm incumbent is a valid hierarchical solution even when
            # phase 2 degrades; never return worse than what phase 1 proved.
            result = warm_result

    wall_time = time.perf_counter() - start_time
    is_feasible = result.is_feasible()
    routes = _extract_routes(result.best) if is_feasible else []
    return MethodResult(
        method="pyvrp-ils-v3",
        problem_type="VRPTW",
        objective_function=objective_function.value,
        instance_id=get_instance_identifier(instance),
        seed=seed,
        time_limit_s=time_limit_s,
        wall_time=wall_time,
        solver_is_feasible=is_feasible,
        routes=routes,
        solver_cost=result.cost(),
        route_count=len(routes),
        vehicle_penalty=vehicle_penalty,
        metadata=_build_method_metadata(
            instance,
            objective_function,
            seed=seed,
            time_limit_s=time_limit_s,
            extra={"vehicle_penalty": vehicle_penalty, **({"arc_cost_scale": scale} if scale != 1 else {})},
        ),
    )


def solve_instance(
    instance: AnyBenchmarkInstance,
    *,
    time_limit_s: int,
    seed: int = 42,
    objective_function: ObjectiveFunction | None = None,
    display: bool = False,
    instance_path: str | Path | None = None,
    monitor: SolveMonitor | None = None,
) -> MethodResult:
    """Dispatch to `solve_cvrp` or `solve_vrptw` based on instance type.

    For VRPTW instances, `objective_function` defaults to MONO_COST when not provided.
    For CVRP instances, the objective is fixed to MONO_COST regardless of the argument.
    Slim collection instances need `instance_path` when their arc costs live in
    a distances sidecar. Pass a `SolveMonitor` to observe the search while it runs.
    """
    if isinstance(instance, (BenchmarkInstanceCVRP, BenchmarkInstanceCVRPCollection)):
        return solve_cvrp(
            instance,
            time_limit_s=time_limit_s,
            seed=seed,
            display=display,
            instance_path=instance_path,
            monitor=monitor,
        )

    if _is_vrptw_like(instance):
        objective = objective_function or ObjectiveFunction.MONO_COST
        return solve_vrptw(
            instance,
            objective_function=objective,
            time_limit_s=time_limit_s,
            seed=seed,
            display=display,
            instance_path=instance_path,
            monitor=monitor,
        )

    raise TypeError(f"Unsupported instance type: {type(instance).__name__}")


def solve_and_update_bks(
    instance: AnyBenchmarkInstance,
    *,
    instance_path: str | Path,
    time_limit_s: int,
    seed: int = 42,
    objective_function: ObjectiveFunction | None = None,
    authors: str = DEFAULT_BKS_AUTHORS,
    display: bool = False,
) -> tuple[MethodResult, BKSUpdateResult | None]:
    """Solve an instance and, if the solution is feasible, write a BKS file in place.

    Returns the `MethodResult` and a `BKSUpdateResult` (or None if the solver did not
    return a feasible solution). The BKS file path is derived from `instance_path`
    via `mamut_routing_lib.artifacts.get_bks_path_for_instance` (handled inside
    `save_bks_if_improved`).
    """
    method_result = solve_instance(
        instance,
        time_limit_s=time_limit_s,
        seed=seed,
        objective_function=objective_function,
        display=display,
        instance_path=instance_path,
    )
    if not method_result.solver_is_feasible or not method_result.routes:
        return method_result, None

    resolved_objective = ObjectiveFunction(method_result.objective_function)
    checkable = hydrate_collection_instance(instance, instance_path)
    candidate = BenchmarkSolution(
        instance_name=get_instance_identifier(instance),
        routes=method_result.routes,
        cost=None,
        metadata={"seed": seed, "method": method_result.method, "wall_time": method_result.wall_time},
    )
    check_result = check_solution(checkable, candidate)
    if not check_result.is_valid():
        return method_result, None

    bks = create_bks_from_solution(
        checkable,
        candidate,
        resolved_objective,
        authors=authors,
        metadata={
            "method": method_result.method,
            "seed": seed,
            "time_limit_s": time_limit_s,
            "wall_time": method_result.wall_time,
            "solver_cost": method_result.solver_cost,
            "vehicle_penalty": method_result.vehicle_penalty,
        },
    )
    update_result = save_bks_if_improved(checkable, bks, Path(instance_path))
    return method_result, update_result

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
from typing import Any

try:
    import numpy as np
    from pyvrp import ProblemData, Solution, SolveParams
    from pyvrp.minimise_fleet import minimise_fleet
    from pyvrp.read import ROUND_FUNCS, _InstanceParser, _ProblemDataBuilder, _RoundingFunc
    from pyvrp.solve import solve as pyvrp_solve
    from pyvrp.stop import FirstFeasible, MaxRuntime, MultipleCriteria
except ImportError as exc:  # pragma: no cover - exercised via test_cli_solve missing-extra path
    raise ImportError(
        "mamut-routing-lib's PyVRP solver module requires the [pyvrp] optional "
        "dependency. Install with: pip install 'mamut-routing-lib[pyvrp]'"
    ) from exc

from mamut_routing_lib.artifacts import (
    AnyBenchmarkInstance,
    get_instance_identifier,
    hydrate_collection_instance,
    resolve_arc_costs,
)
from mamut_routing_lib.bks import (
    BKSUpdateResult,
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
from mamut_routing_lib.solvers.base import INFEASIBLE_COST, SolveMonitor, SolveTick


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

# Kept for callers of the private name; the constant now lives in solvers.base.
_INFEASIBLE_COST = INFEASIBLE_COST


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


def extract_routes(pyvrp_solution: Any) -> list[list[int]]:
    """Public name of ``_extract_routes``: non-empty routes in library ids."""
    return _extract_routes(pyvrp_solution)


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
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> tuple[list[list[int]] | list[list[float]], str, int]:
    """(arc costs, round_func, cost scale) for the solve. Slim collection
    instances carry 3-decimal float costs; PyVRP's 'exact' rounding scales
    them x1000 to integers, so reported solver costs are in milli-units.
    ``arc_costs`` is the instance's matrix when the caller already resolved it."""
    if arc_costs is None:
        arc_costs = _hydrated_arc_costs(instance, instance_path)
    weights = np.asarray(arc_costs)
    if np.allclose(weights, np.round(weights)):
        return arc_costs, "none", 1
    return arc_costs, "exact", 1000


def conversion_plan(
    instance: AnyBenchmarkInstance,
    instance_path: str | Path | None,
) -> tuple[list[list[int]] | list[list[float]], str, int]:
    """Public name of ``_conversion_plan``: (arc costs, round_func, cost scale)."""
    return _conversion_plan(instance, instance_path)


def scale_is_exact(arc_costs: list[list[int]] | list[list[float]], scale: int) -> bool:
    """Whether ``scale`` turns every arc cost into an integer without rounding.

    False for full-precision float matrices (e.g. Sintef2008), which PyVRP's
    ``exact`` rounding approximates to the nearest milli-unit.
    """
    weights = np.asarray(arc_costs, dtype=np.float64) * scale
    return bool(np.all(np.abs(weights - np.round(weights)) <= 1e-9 * np.maximum(1.0, np.abs(weights))))


def _warm_start(problem: ProblemData, initial_routes: list[list[int]] | None) -> Any:
    """A PyVRP ``Solution`` from library routes (depot 0, customers 1..n), or None."""
    if not initial_routes:
        return None
    routes = [[int(stop) for stop in route] for route in initial_routes if route]
    stops = [stop for route in routes for stop in route]
    expected = set(range(1, int(problem.num_clients) + 1))
    repeated = sorted({stop for stop in stops if stops.count(stop) > 1}) if len(stops) != len(set(stops)) else []
    missing = sorted(expected - set(stops))
    foreign = sorted(set(stops) - expected)
    if repeated or missing or foreign:
        # PyVRP would start from an incomplete solution and could hand it back as the result.
        problems = []
        if missing:
            problems.append(f"{len(missing)} customer(s) missing (first: {', '.join(map(str, missing[:5]))})")
        if repeated:
            problems.append(f"customer(s) {', '.join(map(str, repeated[:5]))} visited more than once")
        if foreign:
            problems.append(f"stop id(s) {', '.join(map(str, foreign[:5]))} outside 1..{len(expected)}")
        raise ValueError(f"initial_routes must visit every customer exactly once: {'; '.join(problems)}")
    return Solution(problem, routes)


def solve_cvrp(
    instance: BenchmarkInstanceCVRP | BenchmarkInstanceCVRPCollection,
    *,
    time_limit_s: int,
    seed: int = 42,
    display: bool = False,
    instance_path: str | Path | None = None,
    monitor: SolveMonitor | None = None,
    initial_routes: list[list[int]] | None = None,
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> MethodResult:
    """Solve a CVRP instance with PyVRP's ILS (mono-cost objective only).

    ``initial_routes`` (library ids) warm-start the search. ``arc_costs`` is the instance's
    matrix when the caller already resolved it (a slim collection instance's sidecar is read
    here otherwise).
    """
    start_time = time.perf_counter()
    arc_costs, round_func, scale = _conversion_plan(instance, instance_path, arc_costs)
    if monitor is not None:
        monitor.rescale(scale)
    problem = to_pyvrp_problem(instance, round_func=round_func, arc_costs=arc_costs)
    result = pyvrp_solve(
        problem,
        stop=_stop_with(monitor, MaxRuntime(max(1, time_limit_s))),
        seed=seed,
        collect_stats=False,
        display=display,
        initial_solution=_warm_start(problem, initial_routes),
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
            extra={
                **({"arc_cost_scale": scale} if scale != 1 else {}),
                "cost_exact": scale_is_exact(arc_costs, scale),
                **({"warm_start": True} if initial_routes else {}),
                **({"solver_distance": int(result.best.distance())} if is_feasible else {}),
            },
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
    initial_routes: list[list[int]] | None = None,
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> MethodResult:
    """Solve a VRPTW instance with PyVRP's ILS.

    ``initial_routes`` (library ids) warm-start a MonoCost solve; the hierarchical
    solve builds its own warm start on the minimised fleet and ignores them.
    ``arc_costs`` is the instance's matrix when the caller already resolved it.

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
    arc_costs, round_func, scale = _conversion_plan(instance, instance_path, arc_costs)
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
            initial_solution=_warm_start(problem, initial_routes),
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
            extra={
                "vehicle_penalty": vehicle_penalty,
                **({"arc_cost_scale": scale} if scale != 1 else {}),
                "cost_exact": scale_is_exact(arc_costs, scale),
                **(
                    {"warm_start": True}
                    if initial_routes and objective_function == ObjectiveFunction.MONO_COST
                    else {}
                ),
                # Routing cost of the returned solution in solver units, without the hierarchical
                # fleet cost (which the phase-1 fallback result does not carry at all).
                **({"solver_distance": int(result.best.distance())} if is_feasible else {}),
            },
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
    initial_routes: list[list[int]] | None = None,
    arc_costs: list[list[int]] | list[list[float]] | None = None,
) -> MethodResult:
    """Dispatch to `solve_cvrp` or `solve_vrptw` based on instance type.

    For VRPTW instances, `objective_function` defaults to MONO_COST when not provided.
    For CVRP instances, the objective is fixed to MONO_COST regardless of the argument.
    Slim collection instances need `instance_path` when their arc costs live in
    a distances sidecar, unless `arc_costs` hands the resolved matrix in. Pass a
    `SolveMonitor` to observe the search while it runs.
    """
    if isinstance(instance, (BenchmarkInstanceCVRP, BenchmarkInstanceCVRPCollection)):
        return solve_cvrp(
            instance,
            time_limit_s=time_limit_s,
            seed=seed,
            display=display,
            instance_path=instance_path,
            monitor=monitor,
            initial_routes=initial_routes,
            arc_costs=arc_costs,
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
            initial_routes=initial_routes,
            arc_costs=arc_costs,
        )

    raise TypeError(f"Unsupported instance type: {type(instance).__name__}")


def solve_and_update_bks(
    instance: AnyBenchmarkInstance,
    *,
    instance_path: str | Path,
    time_limit_s: int,
    seed: int = 42,
    objective_function: ObjectiveFunction | None = None,
    authors: str,
    display: bool = False,
) -> tuple[MethodResult, BKSUpdateResult | None]:
    """Solve an instance and, if the solution is feasible, write a BKS file in place.

    ``authors`` (who to credit in the BKS) is required: a stored BKS is published.

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

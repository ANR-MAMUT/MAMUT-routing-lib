"""PyVRP-backed solver integration for `mamut-routing-lib`.

Importing this module requires the `[pyvrp]` optional dependency:

    pip install "mamut-routing-lib[pyvrp]"

Public API:
    to_vrplib_dict, to_pyvrp_problem            -- conversion helpers
    MethodResult                                 -- solver result dataclass
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
    from pyvrp import ProblemData, SolveParams
    from pyvrp.minimise_fleet import _lower_bound
    from pyvrp.read import ROUND_FUNCS, _InstanceParser, _ProblemDataBuilder, _RoundingFunc
    from pyvrp.solve import solve as pyvrp_solve
    from pyvrp.stop import FirstFeasible, MaxRuntime, MultipleCriteria
except ImportError as exc:  # pragma: no cover - exercised via test_cli_solve missing-extra path
    raise ImportError(
        "mamut-routing-lib's PyVRP solver module requires the [pyvrp] optional "
        "dependency. Install with: pip install 'mamut-routing-lib[pyvrp]'"
    ) from exc

from mamut_routing_lib.artifacts import AnyBenchmarkInstance, get_instance_identifier
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
    BenchmarkInstanceVRPTW,
    BenchmarkSolution,
)


__all__ = [
    "MethodResult",
    "solve_and_update_bks",
    "solve_cvrp",
    "solve_instance",
    "solve_vrptw",
    "to_pyvrp_problem",
    "to_vrplib_dict",
]


# Tuning constants for the hierarchical-vehicle-cost two-phase VRPTW solve.
_FLEET_MIN_TIME_FRACTION = 0.3
_FLEET_ATTEMPT_MAX_TIME_FRACTION = 0.2
_FLEET_ATTEMPT_MIN_TIME = 0.5
_FLEET_ATTEMPT_FEASIBLE_MULTIPLIER = 3.0


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


def to_vrplib_dict(instance: AnyBenchmarkInstance) -> dict[str, Any]:
    """Convert a benchmark instance into a VRPLIB-shaped dict consumable by PyVRP's parser."""
    payload: dict[str, Any] = {
        "name": get_instance_identifier(instance),
        "type": "CVRPTW" if isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTW)) else "CVRP",
        "dimension": instance.num_customers + 1,
        "vehicles": instance.num_vehicles if instance.num_vehicles is not None else instance.num_customers,
        "capacity": instance.vehicle_capacity,
        "edge_weight_type": "EXPLICIT",
        "edge_weight_format": "FULL_MATRIX",
        "node_coord": np.asarray(instance.coordinates, dtype=float),
        "demand": np.asarray(instance.demands, dtype=int),
        "depot": np.array([instance.depot], dtype=int),
        "edge_weight": np.asarray(instance.arc_costs, dtype=int),
    }

    if isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTW)):
        payload["time_window"] = np.asarray(instance.time_windows, dtype=int)
        payload["service_time"] = np.asarray(instance.service_times, dtype=int)

    return payload


def to_pyvrp_problem(
    instance: AnyBenchmarkInstance,
    round_func: str | _RoundingFunc = "none",
) -> ProblemData:
    """Convert a benchmark instance into a PyVRP `ProblemData`."""
    if (key := str(round_func)) in ROUND_FUNCS:
        round_func = ROUND_FUNCS[key]

    if not callable(round_func):
        raise TypeError(
            f"round_func = {round_func} is not understood. Can be a function, "
            f"or one of {list(ROUND_FUNCS.keys())}."
        )

    parser = _InstanceParser(to_vrplib_dict(instance), round_func)
    builder = _ProblemDataBuilder(parser)
    return builder.data()


def _extract_routes(pyvrp_solution: Any) -> list[list[int]]:
    routes: list[list[int]] = []
    for route in pyvrp_solution.routes():
        route_customers = [int(customer) for customer in route]
        if route_customers:
            routes.append(route_customers)
    return routes


def _get_vehicle_fixed_cost(instance: AnyBenchmarkInstance) -> int | float:
    max_arc_cost = max(max(row) for row in instance.arc_costs)
    return 2 * (instance.num_customers + 1) * max_arc_cost + 1


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


def _minimise_fleet_adaptive(
    problem: ProblemData,
    *,
    total_phase1_budget: float,
    start_time: float,
    seed: int,
    params: SolveParams | None = None,
) -> Any:
    """Adaptively reduce vehicle count via repeated FirstFeasible|MaxRuntime probes.

    Returns the most constrained `VehicleType` for which a feasible solution was
    found within the phase-1 budget.
    """
    params = params or SolveParams()
    feasible_vehicle_type = problem.vehicle_type(0)
    lower_bound = _lower_bound(problem)
    max_feasible_attempt_time = 0.0
    fleet_attempt_max_time = max(
        _FLEET_ATTEMPT_MIN_TIME,
        _FLEET_ATTEMPT_MAX_TIME_FRACTION * total_phase1_budget,
    )

    while feasible_vehicle_type.num_available > lower_bound:
        elapsed = time.perf_counter() - start_time
        remaining_phase1 = total_phase1_budget - elapsed
        if remaining_phase1 <= 0:
            break

        if max_feasible_attempt_time > 0:
            adaptive_cap = max(
                _FLEET_ATTEMPT_MIN_TIME,
                max_feasible_attempt_time * _FLEET_ATTEMPT_FEASIBLE_MULTIPLIER,
            )
        else:
            adaptive_cap = fleet_attempt_max_time

        attempt_time = min(fleet_attempt_max_time, adaptive_cap, remaining_phase1)
        candidate_vehicle_type = feasible_vehicle_type.replace(
            num_available=feasible_vehicle_type.num_available - 1
        )
        candidate_problem = problem.replace(vehicle_types=[candidate_vehicle_type])
        attempt_start = time.perf_counter()
        stop_criterion = MultipleCriteria([FirstFeasible(), MaxRuntime(attempt_time)])
        result = pyvrp_solve(
            candidate_problem,
            stop=stop_criterion,
            seed=seed,
            collect_stats=False,
            display=False,
            params=params,
        )
        attempt_elapsed = time.perf_counter() - attempt_start
        if not result.is_feasible():
            break

        max_feasible_attempt_time = max(max_feasible_attempt_time, attempt_elapsed)
        feasible_vehicle_type = candidate_vehicle_type

        if result.best.num_routes() < candidate_problem.num_vehicles:
            feasible_vehicle_type = candidate_vehicle_type.replace(
                num_available=result.best.num_routes()
            )

    return feasible_vehicle_type


def solve_cvrp(
    instance: BenchmarkInstanceCVRP,
    *,
    time_limit_s: int,
    seed: int = 42,
    display: bool = False,
) -> MethodResult:
    """Solve a CVRP instance with PyVRP's HGS (mono-cost objective only)."""
    start_time = time.perf_counter()
    problem = to_pyvrp_problem(instance)
    result = pyvrp_solve(
        problem,
        stop=MaxRuntime(max(1, time_limit_s)),
        seed=seed,
        collect_stats=False,
        display=display,
    )
    wall_time = time.perf_counter() - start_time
    is_feasible = result.is_feasible()
    routes = _extract_routes(result.best) if is_feasible else []
    return MethodResult(
        method="hgs-v1",
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
        ),
    )


def solve_vrptw(
    instance: BenchmarkInstance | BenchmarkInstanceVRPTW,
    *,
    objective_function: ObjectiveFunction,
    time_limit_s: int,
    seed: int = 42,
    display: bool = False,
) -> MethodResult:
    """Solve a VRPTW instance with PyVRP's HGS.

    For ObjectiveFunction.MONO_COST, runs a single solve with the full time budget.
    For ObjectiveFunction.HIERARCHICAL_VEHICLE_COST, runs a two-phase solve:
      1. Adaptive fleet minimization within ~30% of the budget.
      2. Cost optimization on the constrained fleet for the remaining budget,
         with a per-vehicle fixed cost large enough to dominate routing cost.
    """
    start_time = time.perf_counter()
    problem = to_pyvrp_problem(instance)
    params = SolveParams()
    vehicle_penalty: int | float = 0

    if objective_function == ObjectiveFunction.MONO_COST:
        result = pyvrp_solve(
            problem,
            stop=MaxRuntime(max(1, time_limit_s)),
            seed=seed,
            collect_stats=False,
            display=display,
            params=params,
        )
    else:
        fleet_budget = float(time_limit_s) * _FLEET_MIN_TIME_FRACTION
        constrained_vehicle_type = _minimise_fleet_adaptive(
            problem,
            total_phase1_budget=fleet_budget,
            start_time=start_time,
            seed=seed,
            params=params,
        )
        vehicle_penalty = _get_vehicle_fixed_cost(instance)
        constrained_problem = problem.replace(
            vehicle_types=[constrained_vehicle_type.replace(fixed_cost=vehicle_penalty)]
        )
        remaining_time = max(1, time_limit_s - int(time.perf_counter() - start_time))
        result = pyvrp_solve(
            constrained_problem,
            stop=MaxRuntime(remaining_time),
            seed=seed,
            collect_stats=False,
            display=display,
            params=params,
        )

    wall_time = time.perf_counter() - start_time
    is_feasible = result.is_feasible()
    routes = _extract_routes(result.best) if is_feasible else []
    return MethodResult(
        method="hgs-v3",
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
            extra={"vehicle_penalty": vehicle_penalty},
        ),
    )


def solve_instance(
    instance: AnyBenchmarkInstance,
    *,
    time_limit_s: int,
    seed: int = 42,
    objective_function: ObjectiveFunction | None = None,
    display: bool = False,
) -> MethodResult:
    """Dispatch to `solve_cvrp` or `solve_vrptw` based on instance type.

    For VRPTW instances, `objective_function` defaults to MONO_COST when not provided.
    For CVRP instances, the objective is fixed to MONO_COST regardless of the argument.
    """
    if isinstance(instance, BenchmarkInstanceCVRP):
        return solve_cvrp(instance, time_limit_s=time_limit_s, seed=seed, display=display)

    if isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTW)):
        objective = objective_function or ObjectiveFunction.MONO_COST
        return solve_vrptw(
            instance,
            objective_function=objective,
            time_limit_s=time_limit_s,
            seed=seed,
            display=display,
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
    )
    if not method_result.solver_is_feasible or not method_result.routes:
        return method_result, None

    resolved_objective = ObjectiveFunction(method_result.objective_function)
    candidate = BenchmarkSolution(
        instance_name=get_instance_identifier(instance),
        routes=method_result.routes,
        cost=None,
        metadata={"seed": seed, "method": method_result.method, "wall_time": method_result.wall_time},
    )
    check_result = check_solution(instance, candidate)
    if not check_result.is_valid():
        return method_result, None

    bks = create_bks_from_solution(
        instance,
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
    update_result = save_bks_if_improved(instance, bks, Path(instance_path))
    return method_result, update_result

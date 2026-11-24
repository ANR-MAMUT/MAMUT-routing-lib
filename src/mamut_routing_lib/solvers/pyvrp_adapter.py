"""PyVRP as a registry solver (``pyvrp``).

This module never imports PyVRP itself: the registry lists the solver (as
unavailable) when the ``[pyvrp]`` extra is missing, and ``solve`` imports
``mamut_routing_lib.solvers.pyvrp`` lazily. It calls ``solve_instance`` through
that module's attribute so ``unittest.mock.patch`` targets keep working.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers.base import (
    Availability,
    InstanceProfile,
    SolveMonitor,
    SolveRequest,
    SolveResult,
    SolverCapabilities,
    Support,
    capability_support,
)
from mamut_routing_lib.solvers.provenance import environment, package_version

__all__ = ["PyVRPSolver", "method_result_to_solve_result"]

_CAPABILITIES = SolverCapabilities(
    objectives={
        ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST}),
        ProblemType.VRPTW: frozenset({ObjectiveFunction.MONO_COST, ObjectiveFunction.HIERARCHICAL_VEHICLE_COST}),
    },
    input="model",
    symmetric_only=False,
    integer_costs=False,
    fleet_limit=True,
    warm_start=True,
    live_progress="incumbent",
    cancellable=True,
    in_process=True,
)


def method_result_to_solve_result(method_result: Any, *, stopped_early: bool = False) -> SolveResult:
    """Pass the PyVRP objective report and returned routes to the registry."""
    metadata = dict(method_result.metadata or {})
    feasible = bool(method_result.solver_is_feasible)
    return SolveResult(
        solver="pyvrp",
        method=method_result.method,
        problem_type=method_result.problem_type,
        objective_function=method_result.objective_function,
        instance_id=method_result.instance_id,
        seed=method_result.seed,
        time_limit_s=float(method_result.time_limit_s),
        wall_time=float(method_result.wall_time),
        # A stop request (cancellation) ends the search early; the routes are the incumbent at the stop.
        status="cancelled" if stopped_early else ("feasible" if feasible else "infeasible"),
        routes=[list(route) for route in method_result.routes] if feasible else [],
        solver_report=method_result.solver_report if feasible else None,
        cost_scale=metadata.get("arc_cost_scale", 1),
        stopped_early=stopped_early,
        solver_version=package_version("pyvrp"),
        metadata=metadata,
        provenance={**environment(), "pyvrp_version": package_version("pyvrp")},
    )


class PyVRPSolver:
    """PyVRP's iterated local search: CVRP (MonoCost) and VRPTW (MonoCost, hierarchical)."""

    name = "pyvrp"
    label = "PyVRP"
    capabilities = _CAPABILITIES

    def availability(self) -> Availability:
        missing = [name for name in ("pyvrp", "numpy") if importlib.util.find_spec(name) is None]
        if missing:
            return Availability(
                False,
                reason=f"the [pyvrp] extra is not installed (missing {', '.join(missing)})",
                hint="pip install 'mamut-routing-lib[pyvrp]'",
            )
        return Availability(True, version=package_version("pyvrp"))

    def supports(self, profile: InstanceProfile, objective: ObjectiveFunction) -> Support:
        support = capability_support(self.capabilities, profile, objective, solver_label=self.label)
        if profile.depot != 0:
            # PyVRP's reader wants its depots in the first indices and refuses the instance otherwise.
            reason = f"{self.label} needs the depot at index 0 (it is {profile.depot})"
            return Support(False, (*support.reasons, reason), support.warnings)
        return support

    def solve(self, request: SolveRequest, monitor: SolveMonitor | None = None) -> SolveResult:
        from mamut_routing_lib.solvers import pyvrp as pyvrp_module

        started = time.perf_counter()
        try:
            method_result = pyvrp_module.solve_instance(
                request.instance,
                time_limit_s=max(1, int(round(request.time_limit_s))),
                seed=request.seed,
                objective_function=request.objective,
                display=request.display,
                instance_path=request.instance_path,
                monitor=monitor,
                initial_routes=request.initial_routes,
                # The matrix the checker will price the routes on: read from its sidecar once.
                arc_costs=request.checkable().arc_costs,
            )
        except Exception as error:  # noqa: BLE001 - a solver failure is a result, not an exception
            return SolveResult(
                solver=self.name,
                method="pyvrp",
                problem_type=getattr(getattr(request.instance, "__class__", None), "__name__", ""),
                objective_function=request.objective.value,
                instance_id=str(getattr(request.instance, "instance_name", "")),
                seed=request.seed,
                time_limit_s=float(request.time_limit_s),
                wall_time=time.perf_counter() - started,
                status="failed",
                error=f"{type(error).__name__}: {error}",
                solver_version=package_version("pyvrp"),
                provenance=environment(),
            )
        return method_result_to_solve_result(
            method_result,
            stopped_early=bool(monitor is not None and monitor.stopped_early),
        )

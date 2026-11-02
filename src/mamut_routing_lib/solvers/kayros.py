"""KAYROS as a registry solver (``kayros``): time-dependent TDVRP / TDVRPTW, Duration.

KAYROS is a separate package (``pip install kayros``, or the tools'
``[kayros]`` extra) that implements the lib's TD checker contract. This module
imports it lazily. Its ``on_incumbent`` hook fires synchronously on every new
incumbent and aborts the solve when it raises: the adapter feeds the hook to the
monitor (live incumbents) and raises a private exception on a stop request,
keeping the last incumbent's routes. The routes are re-checked by the lib's TD
checker afterwards (``solvers.run.run_solver``), like every other solver's.
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

__all__ = ["KayrosSolver"]

_DURATION = frozenset({ObjectiveFunction.DURATION})
_CAPABILITIES = SolverCapabilities(
    objectives={ProblemType.TDVRP: _DURATION, ProblemType.TDVRPTW: _DURATION},
    input="model",
    symmetric_only=False,
    fleet_limit=True,
    warm_start=False,
    live_progress="incumbent",
    cancellable=True,
    in_process=True,
)


class _StopRequested(Exception):
    """Raised inside KAYROS's incumbent hook to abort the solve."""


class KayrosSolver:
    """KAYROS (ACO + ILS) for time-dependent instances, Duration objective."""

    name = "kayros"
    label = "KAYROS"
    capabilities = _CAPABILITIES

    def availability(self) -> Availability:
        try:
            spec = importlib.util.find_spec("kayros")
        except (ValueError, ImportError):
            spec = None
        if spec is None:
            return Availability(
                False,
                reason="the kayros package is not installed",
                hint="pip install 'kayros>=1.1.2' (or mamut-routing-tools[kayros])",
            )
        return Availability(True, version=package_version("kayros"))

    def supports(self, profile: InstanceProfile, objective: ObjectiveFunction) -> Support:
        return capability_support(self.capabilities, profile, objective, solver_label=self.label)

    def solve(self, request: SolveRequest, monitor: SolveMonitor | None = None) -> SolveResult:
        import kayros

        from mamut_routing_lib.td import load_td_instance

        started = time.perf_counter()
        version = package_version("kayros")
        base = dict(
            solver=self.name,
            method="kayros",
            problem_type=type(request.instance).__name__,
            objective_function=request.objective.value,
            instance_id=str(getattr(request.instance, "instance_name", "")),
            seed=int(request.seed),
            time_limit_s=float(request.time_limit_s),
            solver_version=version,
            provenance={**environment(), "kayros_version": version},
        )
        last: dict[str, Any] = {"routes": None, "value": None, "iteration": None}

        def on_incumbent(incumbent: Any, routes: list[list[int]]) -> None:
            last["routes"] = [list(route) for route in routes]
            last["value"] = float(incumbent.value)
            last["iteration"] = int(getattr(incumbent, "iteration", 0) or 0)
            if monitor is not None and monitor.observe(last["value"], iterations=last["iteration"], force=True):
                raise _StopRequested

        try:
            if monitor is not None:
                monitor.phase("solving")
            loaded = load_td_instance(request.instance_path)
            solution = kayros.solve(
                loaded,
                time_limit=float(request.time_limit_s),
                seed=int(request.seed),
                on_incumbent=on_incumbent,
            )
        except _StopRequested:
            return SolveResult(
                **base,
                wall_time=time.perf_counter() - started,
                status="cancelled",
                routes=last["routes"] or [],
                solver_cost=last["value"],
                solver_cost_raw=last["value"],
                stopped_early=True,
                metadata={"iterations": last["iteration"]},
            )
        except Exception as error:  # noqa: BLE001 - a solver failure is a result
            return SolveResult(
                **base,
                wall_time=time.perf_counter() - started,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        routes = [list(route) for route in solution.routes]
        return SolveResult(
            **base,
            wall_time=time.perf_counter() - started,
            status="feasible" if routes else "infeasible",
            routes=routes,
            solver_cost=float(solution.duration),
            solver_cost_raw=float(solution.duration),
            cost_exact=True,
            metadata={
                "status": getattr(solution, "status", None),
                "iterations": getattr(solution, "iterations", None),
                "route_durations": list(getattr(solution, "route_durations", []) or []),
            },
        )

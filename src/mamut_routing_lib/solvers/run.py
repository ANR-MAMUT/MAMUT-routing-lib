"""Run any registered solver on one instance and check its routes.

``run_solver`` is the one entry point the CLIs, the tools GUI and campaign
runners share:

1. load the instance (static or time-dependent) unless it is given;
2. pick the objective (``objectives.default_objective`` when none is given);
3. refuse an unavailable solver (``SolverUnavailableError``) or an instance the
   solver cannot take (``InstanceNotSupportedError``), before any work;
4. solve, with the monitor's clock restarted and its budget set;
5. price the routes with the reference checker (``solutions.check_routes``).

The checker's verdict and cost are authoritative; the solver's own cost is kept
in the result for information. Storing a BKS is the caller's decision
(``solutions.submit_bks`` / ``bks.save_solution_as_bks_if_improved``), so that a
campaign can keep a single writer per instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.solvers.base import (
    InstanceNotSupportedError,
    Solver,
    SolveMonitor,
    SolveRequest,
    SolveResult,
    SolverUnavailableError,
    profile_instance,
)

__all__ = ["CheckedSolve", "prepare_solve", "run_solver"]


@dataclass(frozen=True)
class CheckedSolve:
    """A solver result plus the reference checker's verdict on its routes."""

    result: SolveResult
    check: Any | None
    objective: ObjectiveFunction
    instance_path: Path
    instance_name: str
    support_warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.check is not None and self.check.is_valid()

    @property
    def cost(self) -> int | float | None:
        """The checker's cost of the routes (``None`` unless valid)."""
        return self.check.routing_cost if self.valid else None

    @property
    def num_routes(self) -> int:
        return len(self.result.routes)

    @property
    def validation_status(self) -> str:
        if self.check is None:
            return "no-solution"
        return getattr(self.check.status, "value", str(self.check.status))

    def to_solution(self, *, metadata: Mapping[str, Any] | None = None) -> Any:
        """The routes as a ``BenchmarkSolution`` (cost left to the checker)."""
        from mamut_routing_lib.models import BenchmarkSolution

        return BenchmarkSolution(
            instance_name=self.instance_name,
            routes=[list(route) for route in self.result.routes],
            cost=None,
            metadata=dict(metadata or {}),
        )

    def bks_metadata(self, **extra: Any) -> dict[str, Any]:
        """Provenance for a BKS built from this solve (no hostname: BKS files are published)."""
        result = self.result
        values: dict[str, Any] = {
            "method": result.method,
            "solver": result.solver,
            "solver_version": result.solver_version,
            "seed": result.seed,
            "time_limit_s": result.time_limit_s,
            "wall_time": round(result.wall_time, 3),
            "solver_cost": result.solver_cost,
            "cost_exact": result.cost_exact,
            "machine": result.provenance.get("machine"),
        }
        if "vehicle_penalty" in result.metadata:
            values["vehicle_penalty"] = result.metadata["vehicle_penalty"]
        for key in ("executable_sha256", "input_sha256"):
            if result.provenance.get(key):
                values[key] = result.provenance[key]
        values.update(extra)
        return {key: value for key, value in values.items() if value is not None}

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready summary: the checker's verdict next to the solver result."""
        check = None
        if self.check is not None:
            check = self.check.model_dump(mode="json")
        return {
            "ok": self.valid and self.result.status in ("feasible", "timeout"),
            "valid": self.valid,
            "cost": self.cost,
            "validation": self.validation_status,
            "objective_function": self.objective.value,
            "instance_name": self.instance_name,
            "instance_path": str(self.instance_path),
            "support_warnings": list(self.support_warnings),
            "check": check,
            **{key: value for key, value in self.result.to_json().items() if key != "objective_function"},
        }


def _resolve_solver(solver: str | Solver, config: Any) -> Solver:
    if isinstance(solver, str):
        from mamut_routing_lib.solvers.registry import get_solver

        return get_solver(solver, config=config)
    return solver


def prepare_solve(
    solver: str | Solver,
    instance_path: str | Path,
    *,
    objective_function: ObjectiveFunction | None = None,
    instance: Any = None,
    config: Any = None,
) -> tuple[Solver, Any, ObjectiveFunction, tuple[str, ...]]:
    """Resolve and vet a solve without running it: (solver, instance, objective, support warnings).

    Raises ``SolverUnavailableError`` or ``InstanceNotSupportedError`` with the reasons.
    """
    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.objectives import default_objective

    solver_obj = _resolve_solver(solver, config)
    path = Path(instance_path)
    loaded = instance if instance is not None else load_benchmark_instance(path)
    objective = objective_function or default_objective(loaded)
    availability = solver_obj.availability()
    if not availability.available:
        hint = f" ({availability.hint})" if availability.hint else ""
        raise SolverUnavailableError(f"{solver_obj.name}: {availability.reason}{hint}")
    support = solver_obj.supports(profile_instance(loaded, path), objective)
    if not support.supported:
        raise InstanceNotSupportedError(f"{solver_obj.name}: " + "; ".join(support.reasons))
    return solver_obj, loaded, objective, support.warnings


def run_solver(
    solver: str | Solver,
    instance_path: str | Path,
    *,
    time_limit_s: float,
    seed: int = 42,
    objective_function: ObjectiveFunction | None = None,
    instance: Any = None,
    monitor: SolveMonitor | None = None,
    initial_routes: list[list[int]] | None = None,
    workdir: str | Path | None = None,
    keep_artifacts: bool = False,
    display: bool = False,
    options: Mapping[str, Any] | None = None,
    config: Any = None,
) -> CheckedSolve:
    """Solve the instance at ``instance_path`` with ``solver`` (a name or a ``Solver``) and check it."""
    from mamut_routing_lib.artifacts import get_instance_identifier
    from mamut_routing_lib.solutions import check_routes

    if not time_limit_s or time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    solver_obj, loaded, objective, warnings = prepare_solve(
        solver,
        instance_path,
        objective_function=objective_function,
        instance=instance,
        config=config,
    )
    path = Path(instance_path)
    if monitor is not None:
        monitor.restart_clock()
        monitor.set_budget(float(time_limit_s))
    request = SolveRequest(
        instance=loaded,
        instance_path=path,
        objective=objective,
        time_limit_s=float(time_limit_s),
        seed=int(seed),
        initial_routes=[list(route) for route in initial_routes] if initial_routes else None,
        workdir=Path(workdir) if workdir is not None else None,
        keep_artifacts=keep_artifacts,
        display=display,
        options=dict(options or {}),
    )
    result = solver_obj.solve(request, monitor)
    check = None
    if result.routes:
        check = check_routes(loaded, path, result.routes, objective)
    return CheckedSolve(
        result=result,
        check=check,
        objective=objective,
        instance_path=path,
        instance_name=get_instance_identifier(loaded),
        support_warnings=tuple(warnings),
    )

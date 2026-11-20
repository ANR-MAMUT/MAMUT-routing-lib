"""KAYROS as a registry solver (``kayros``): time-dependent TDVRP / TDVRPTW, Duration.

KAYROS is a separate package (``pip install kayros``, or the tools'
``[kayros]`` extra) that implements the lib's TD checker contract. Its search is
one native call whose only way back into Python is the ``on_incumbent`` hook: a
stop request raised from the hook is not seen once the search stops improving,
which can be the rest of the time limit.

The adapter therefore runs each solve in a child Python process
(``solvers._kayros_worker``) through the external-solver process runner: its own
process group, a hard timeout, and a stop request (the monitor, Ctrl-C, a GUI
cancel) that kills the process within a poll. The worker saves every incumbent
before reporting it, so a stopped solve still returns its last incumbent's
routes. The routes are re-checked by the lib's TD checker afterwards
(``solvers.run.run_solver``), like every other solver's.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
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

CHECKPOINT_FILE = "kayros-incumbent.json"
RESULT_FILE = "kayros-result.json"
LOG_FILE = "kayros.log"
WORKER_MODULE = "mamut_routing_lib.solvers._kayros_worker"
#: The worker's progress lines (``_kayros_worker.save_incumbent``).
PROGRESS_PATTERN = r"^incumbent cost=(?P<cost>\S+) iterations=(?P<iterations>\d+)$"
#: KAYROS stops at its time limit by itself; the hard timeout is only there for a process that
#: never returns. It leaves room for what the limit does not cover: loading a large instance's
#: travel-time functions before the search, and the final check after it.
HARD_TIMEOUT_GRACE_S = 600.0

_DURATION = frozenset({ObjectiveFunction.DURATION})
_CAPABILITIES = SolverCapabilities(
    objectives={ProblemType.TDVRP: _DURATION, ProblemType.TDVRPTW: _DURATION},
    input="model",
    symmetric_only=False,
    fleet_limit=True,
    warm_start=False,
    live_progress="incumbent",
    cancellable=True,
    in_process=False,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _routes(payload: dict[str, Any] | None) -> list[list[int]]:
    return [[int(stop) for stop in route] for route in (payload or {}).get("routes") or []]


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
        from mamut_routing_lib.solvers.external.process import run_process

        started = time.perf_counter()
        version = package_version("kayros")
        provenance: dict[str, Any] = {**environment(), "kayros_version": version}
        base = dict(
            solver=self.name,
            method="kayros",
            problem_type=type(request.instance).__name__,
            objective_function=request.objective.value,
            instance_id=str(getattr(request.instance, "instance_name", "")),
            seed=int(request.seed),
            time_limit_s=float(request.time_limit_s),
            solver_version=version,
            provenance=provenance,
        )
        if request.instance_path is None:
            return SolveResult(**base, wall_time=0.0, status="failed", error="KAYROS reads the instance from its file: no instance path")
        if monitor is not None:
            if monitor.stop_requested:  # cancelled before anything started
                return SolveResult(**base, wall_time=time.perf_counter() - started, status="cancelled", stopped_early=True)
            monitor.phase("solving")
        own_workdir = request.workdir is None
        workdir = Path(request.workdir or tempfile.mkdtemp(prefix="mamut-kayros-")).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        checkpoint, result_file = workdir / CHECKPOINT_FILE, workdir / RESULT_FILE
        try:
            # Files left by an earlier run in a reused workdir must never be read as this run's answer.
            checkpoint.unlink(missing_ok=True)
            result_file.unlink(missing_ok=True)
            grace = max(HARD_TIMEOUT_GRACE_S, float(request.time_limit_s))
            outcome = run_process(
                [
                    sys.executable,
                    "-m",
                    WORKER_MODULE,
                    str(Path(request.instance_path).resolve()),
                    "--time-limit",
                    repr(float(request.time_limit_s)),
                    "--seed",
                    str(int(request.seed)),
                    "--checkpoint",
                    str(checkpoint),
                    "--result",
                    str(result_file),
                ],
                cwd=workdir,
                log_path=workdir / LOG_FILE,
                time_limit_s=float(request.time_limit_s),
                grace_s=grace,
                monitor=monitor,
                progress_pattern=PROGRESS_PATTERN,
            )
            provenance.update(
                returncode=outcome.returncode,
                termination=outcome.termination,
                hard_timeout_s=float(request.time_limit_s) + grace,
            )
            return self._result(base, outcome, _read_json(result_file), _read_json(checkpoint), time.perf_counter() - started)
        finally:
            if own_workdir and not request.keep_artifacts:
                shutil.rmtree(workdir, ignore_errors=True)
            else:
                provenance["artifacts_dir"] = str(workdir)

    @staticmethod
    def _result(
        base: dict[str, Any],
        outcome: Any,
        result: dict[str, Any] | None,
        incumbent: dict[str, Any] | None,
        wall_time: float,
    ) -> SolveResult:
        if outcome.termination == "error":
            return SolveResult(**base, wall_time=wall_time, status="failed", error=outcome.error)
        if result is not None and result.get("ok"):
            # A stop request that lands as the worker finishes still gets the finished solution.
            routes = _routes(result)
            stopped = outcome.termination == "cancelled"
            return SolveResult(
                **base,
                wall_time=wall_time,
                status="cancelled" if stopped else ("feasible" if routes else "infeasible"),
                routes=routes,
                solver_cost=float(result["duration"]),
                solver_cost_raw=float(result["duration"]),
                cost_exact=True,
                stopped_early=stopped,
                metadata={
                    "status": result.get("status"),
                    "iterations": result.get("iterations"),
                    "route_durations": list(result.get("route_durations") or []),
                },
            )
        if outcome.termination == "exited":
            error = (result or {}).get("error")
            if not error:
                lines = [line.strip() for line in outcome.log_tail.splitlines() if line.strip()][-3:]
                error = f"KAYROS exited with code {outcome.returncode} and no result" + (f": {' | '.join(lines)}" if lines else "")
            return SolveResult(**base, wall_time=wall_time, status="failed", error=str(error))
        # Stopped on request, or killed at the hard timeout: the last incumbent the worker saved.
        value = (incumbent or {}).get("value")
        common = dict(
            **base,
            wall_time=wall_time,
            routes=_routes(incumbent),
            solver_cost=None if value is None else float(value),
            solver_cost_raw=None if value is None else float(value),
            metadata={"iterations": (incumbent or {}).get("iteration")},
        )
        if outcome.termination == "cancelled":
            return SolveResult(**common, status="cancelled", stopped_early=True)
        return SolveResult(**common, status="timeout", error="KAYROS did not return by its time limit and was stopped at the hard timeout")

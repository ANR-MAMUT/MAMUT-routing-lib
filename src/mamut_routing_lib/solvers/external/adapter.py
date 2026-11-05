"""``ExternalSolver``: run a described solver executable on a library instance.

One solve, in order:

1. hydrate the instance (collection matrices from their sidecars) and refuse
   what the solver cannot take: depot not at index 0, asymmetric costs for a
   symmetric solver, demands or distances beyond its limits;
2. pick the integer scale (``scaling``) and write the input files with fixed
   names into the work directory (``problem.vrp``, ``problem.par``);
3. run the executable (``process``): own process group, budget + grace,
   cooperative stop through the monitor, progress lines as live incumbents;
4. read the solution file (``solutions`` parsers), map its ids onto the
   library's, and bring the reported cost back to instance units -- exactly
   when the scale was exact, in which case it is cross-checked against the
   checker's exact decimal cost.

The routes still go through the reference checker afterwards
(``solvers.run.run_solver``): a solver's own cost is never trusted.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.solvers.base import (
    Availability,
    InstanceNotSupportedError,
    InstanceProfile,
    SolveMonitor,
    SolveRequest,
    SolveResult,
    Support,
    capability_support,
)
from mamut_routing_lib.solvers.external.descriptor import ExternalSolverDescriptor
from mamut_routing_lib.solvers.external.process import run_process
from mamut_routing_lib.solvers.external.scaling import (
    ScaleError,
    ScalePlan,
    format_float_matrix,
    max_decimals,
    plan_coordinate_scale,
    plan_explicit_scale,
    plan_float_matrix_scale,
    scale_coordinates,
    scale_explicit_data,
    unscale_cost,
)
from mamut_routing_lib.solvers.provenance import environment, file_sha256

__all__ = ["ExternalSolver", "PreparedInput"]

PROBLEM_FILE = "problem.vrp"
PARAMS_FILE = "problem.par"
LOG_FILE = "solver.log"


@dataclass(frozen=True)
class PreparedInput:
    """Input files written for one run, and how the data was scaled."""

    workdir: Path
    problem: Path
    params: Path | None
    solution: Path
    plan: ScalePlan
    vehicles: int | None
    fixed_fleet: int | None
    problem_type_line: str
    checkable: Any


def _fmt_seconds(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


class ExternalSolver:
    """A ``Solver`` backed by an executable described by an ``ExternalSolverDescriptor``."""

    def __init__(self, descriptor: ExternalSolverDescriptor, *, config_path: str | None = None):
        self.descriptor = descriptor
        self.name = descriptor.name
        self.label = descriptor.label
        self.capabilities = descriptor.capabilities
        self._config_path = config_path

    def __repr__(self) -> str:
        return f"ExternalSolver({self.name!r}, source={self.descriptor.source!r})"

    # -- availability ---------------------------------------------------------------------------

    def _config_hint(self) -> str:
        from mamut_routing_lib.solvers.external.config import default_config_path

        where = self._config_path or str(default_config_path())
        if self.descriptor.runtime == "java":
            return f"set solvers.{self.name}.jar in {where}"
        if self.descriptor.executable:
            return f"fix solvers.{self.name}.executable in {where}"
        names = self.descriptor.executable_names
        return f"set solvers.{self.name}.executable in {where}, or put {names[0] if names else self.name} on PATH"

    def resolve_executable(self) -> tuple[str | None, str | None]:
        """(absolute executable or jar, reason when missing)."""
        descriptor = self.descriptor
        if descriptor.runtime == "python":
            return sys.executable, None
        if descriptor.runtime == "java":
            if not descriptor.jar:
                return None, "no jar configured"
            jar = Path(descriptor.jar).expanduser()
            if not jar.is_absolute() or not jar.is_file():
                return None, f"jar not found: {jar}" + ("" if jar.is_absolute() else " (give an absolute path)")
            return str(jar), None
        if descriptor.executable:
            # A configured executable is the only candidate: falling back to another build found on
            # PATH (an unpatched HGS, an older FILO2) would run, and credit, the wrong solver.
            configured = descriptor.executable
            path = Path(configured).expanduser()
            if os.sep in configured or path.is_absolute():
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path.resolve()), None
                return None, f"executable not found or not executable: {path}"
            found = shutil.which(configured)
            if found:
                return str(Path(found).resolve()), None
            return None, f"configured executable {configured!r} is not on PATH"
        for candidate in descriptor.executable_names:
            found = shutil.which(candidate)
            if found:
                return str(Path(found).resolve()), None
        return None, "executable not found"

    def _java(self) -> tuple[str | None, str | None]:
        java = self.descriptor.java
        path = Path(java).expanduser()
        if path.is_absolute():
            return (str(path), None) if path.is_file() else (None, f"java not found: {path}")
        found = shutil.which(java)
        return (found, None) if found else (None, f"{java} not found on PATH")

    def availability(self) -> Availability:
        executable, reason = self.resolve_executable()
        note = self.descriptor.requires_note
        if executable is None:
            return Availability(False, reason=reason, hint=self._config_hint(), note=note)
        if self.descriptor.runtime == "java":
            java, java_reason = self._java()
            if java is None:
                return Availability(
                    False, reason=java_reason, executable=executable, hint=f"install a JDK/JRE or set solvers.{self.name}.java", note=note
                )
        return Availability(True, version=self.descriptor.version, executable=executable, note=note)

    def supports(self, profile: InstanceProfile, objective: ObjectiveFunction) -> Support:
        return capability_support(self.capabilities, profile, objective, solver_label=self.label)

    # -- input ----------------------------------------------------------------------------------

    def _fleet_bound(self, checkable: Any, *, vrptw: bool, options: dict[str, Any]) -> int:
        n = int(checkable.num_customers)
        if options.get("vehicles"):
            return int(options["vehicles"])
        lower = math.ceil(sum(int(d) for d in checkable.demands) / max(1, int(checkable.vehicle_capacity)))
        metadata = getattr(checkable, "metadata", None)
        recorded = metadata.get("num_vehicles_lb") if isinstance(metadata, dict) else getattr(metadata, "num_vehicles_lb", None)
        if recorded:
            lower = max(lower, int(recorded))
        factor = self.descriptor.vrptw_vehicles_factor if vrptw else self.descriptor.vehicles_factor
        return max(1, min(n, math.ceil(factor * max(1, lower)) + 1))

    def prepare(self, request: SolveRequest, workdir: Path) -> PreparedInput:
        """Write the solver's input files into ``workdir``; raises ``InstanceNotSupportedError``."""
        from mamut_routing_lib.artifacts import hydrate_collection_instance, is_collection_instance
        from mamut_routing_lib.cvrplib import collection_cost_decimals, is_symmetric_matrix
        from mamut_routing_lib.solvers.external.problem import (
            render_explicit_cvrplib,
            render_lkh_params,
            render_lkh_problem,
            render_matrix_format,
            render_x_format,
            write_text,
        )

        descriptor = self.descriptor
        instance = request.instance
        checkable = hydrate_collection_instance(instance, request.instance_path)
        if int(checkable.depot) != 0:
            raise InstanceNotSupportedError(f"{self.label} needs the depot at index 0 (it is {checkable.depot})")
        matrix = checkable.arc_costs
        demands = [int(value) for value in checkable.demands]
        limits = descriptor.limits
        if limits.max_demand is not None and max(demands) > limits.max_demand:
            raise InstanceNotSupportedError(f"{self.label} refuses demands above {limits.max_demand:g} (largest {max(demands)})")
        if self.capabilities.symmetric_only and not is_symmetric_matrix(matrix):
            raise InstanceNotSupportedError(f"{self.label} needs symmetric arc costs; this instance's are asymmetric")
        vrptw = hasattr(checkable, "time_windows")
        fixed_fleet = checkable.num_vehicles
        decimals_hint = collection_cost_decimals(instance) if is_collection_instance(instance) else None
        problem = workdir / PROBLEM_FILE
        params: Path | None = None
        solution = workdir / descriptor.solution_file.format(seed=request.seed)
        vehicles: int | None = None
        type_line = "CVRPTW" if vrptw else "CVRP"
        try:
            if descriptor.input_format == "cvrplib-x":
                if vrptw:
                    raise InstanceNotSupportedError(f"{self.label} reads CVRP files; time windows would be dropped")
                published = decimals_hint if decimals_hint is not None else max_decimals(matrix)
                plan = plan_coordinate_scale(
                    checkable.coordinates,
                    candidates=descriptor.coordinate_scales,
                    limits=limits,
                    distance_model="nint" if descriptor.distance_model == "nint" else "float",
                    published_decimals=published,
                )
                write_text(
                    problem,
                    render_x_format(
                        coordinates=scale_coordinates(checkable.coordinates, plan),
                        demands=demands,
                        capacity=int(checkable.vehicle_capacity),
                        coordinate_decimals=int(plan.coordinate_decimals or 0),
                        comment=f"MAMUT-routing-{plan.unit or plan.factor}",
                    ),
                )
                vehicles = fixed_fleet
            elif descriptor.input_format == "cvrplib-matrix":
                if vrptw:
                    raise InstanceNotSupportedError(f"{self.label} reads CVRP files; time windows would be dropped")
                plan = plan_float_matrix_scale(
                    matrix,
                    candidates=descriptor.matrix_scales,
                    limits=limits,
                    decimals_hint=decimals_hint,
                )
                write_text(
                    problem,
                    render_matrix_format(
                        matrix_rows=format_float_matrix(matrix, plan),
                        dimension=len(matrix),
                        coordinates=checkable.coordinates,
                        demands=demands,
                        capacity=int(checkable.vehicle_capacity),
                        comment=f"MAMUT-routing-x{plan.factor:g}",
                    ),
                )
                vehicles = fixed_fleet
            else:
                windows = checkable.time_windows if vrptw else None
                services = checkable.service_times if vrptw else None
                plan = plan_explicit_scale(
                    matrix,
                    time_windows=windows,
                    service_times=services,
                    limits=limits,
                    require_exact=descriptor.require_exact_scale,
                    decimals_hint=decimals_hint,
                )
                scaled, scaled_windows, scaled_services = scale_explicit_data(
                    plan, matrix, time_windows=windows, service_times=services
                )
                if descriptor.input_format == "lkh":
                    if not vrptw:
                        type_line = "CVRP" if is_symmetric_matrix(matrix) else "ACVRP"
                    vehicles = int(fixed_fleet) if fixed_fleet is not None else self._fleet_bound(
                        checkable, vrptw=vrptw, options=dict(request.options)
                    )
                    write_text(
                        problem,
                        render_lkh_problem(
                            problem_type=type_line,
                            matrix=scaled,
                            demands=demands,
                            capacity=int(checkable.vehicle_capacity),
                            vehicles=vehicles,
                            time_windows=scaled_windows,
                            service_times=scaled_services,
                        ),
                    )
                    params = workdir / PARAMS_FILE
                    write_text(
                        params,
                        render_lkh_params(
                            problem=problem,
                            solution=solution,
                            seed=request.seed,
                            time_limit_s=request.time_limit_s,
                            extra=descriptor.extra_args if descriptor.argv == ("{executable}", "{params}") else (),
                        ),
                    )
                else:
                    vehicles = fixed_fleet
                    problem.write_text(
                        render_explicit_cvrplib(
                            matrix=scaled,
                            coordinates=checkable.coordinates,
                            demands=demands,
                            capacity=int(checkable.vehicle_capacity),
                            num_vehicles=fixed_fleet,
                            time_windows=scaled_windows,
                            service_times=scaled_services,
                        ),
                        encoding="ascii",
                    )
        except ScaleError as error:
            raise InstanceNotSupportedError(f"{self.label}: {error}") from error
        return PreparedInput(
            workdir=workdir,
            problem=problem,
            params=params,
            solution=solution,
            plan=plan,
            vehicles=vehicles,
            fixed_fleet=fixed_fleet,
            problem_type_line=type_line,
            checkable=checkable,
        )

    def build_argv(self, prepared: PreparedInput, request: SolveRequest) -> list[str]:
        descriptor = self.descriptor
        executable, reason = self.resolve_executable()
        if executable is None:
            from mamut_routing_lib.solvers.base import SolverUnavailableError

            raise SolverUnavailableError(f"{self.name}: {reason} ({self._config_hint()})")
        values = {
            "executable": executable,
            "jar": executable if descriptor.runtime == "java" else "",
            "java": self._java()[0] or descriptor.java,
            "python": sys.executable,
            "problem": str(prepared.problem),
            "params": str(prepared.params or ""),
            "solution": str(prepared.solution),
            "workdir": str(prepared.workdir),
            "seed": str(int(request.seed)),
            "time_limit_s": _fmt_seconds(request.time_limit_s),
            "time_limit_int": str(max(1, int(round(request.time_limit_s)))),
            "vehicles": str(prepared.vehicles or ""),
        }
        argv: list[str] = []
        for token in descriptor.argv:
            if token == "{java_options}":
                argv.extend(descriptor.java_options)
            else:
                argv.append(token.format_map(values))
        if prepared.fixed_fleet is not None:
            argv.extend(token.format_map(values) for token in descriptor.fleet_args)
        if descriptor.input_format != "lkh" or descriptor.argv != ("{executable}", "{params}"):
            argv.extend(token.format_map(values) for token in descriptor.extra_args)
        return argv

    # -- solve ----------------------------------------------------------------------------------

    def solve(self, request: SolveRequest, monitor: SolveMonitor | None = None) -> SolveResult:
        descriptor = self.descriptor
        started = time.perf_counter()
        own_workdir = request.workdir is None
        workdir = Path(request.workdir or tempfile.mkdtemp(prefix=f"mamut-{self.name}-")).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        base = dict(
            solver=self.name,
            method=descriptor.method,
            problem_type=type(request.instance).__name__,
            objective_function=request.objective.value,
            instance_id=str(getattr(request.instance, "instance_name", "")),
            seed=int(request.seed),
            time_limit_s=float(request.time_limit_s),
            solver_version=descriptor.version,
        )
        provenance: dict[str, Any] = {**environment(), "descriptor_sha256": descriptor.sha256(), "descriptor_source": descriptor.source}
        try:
            if monitor is not None:
                monitor.phase("preparing input")
            prepared = self.prepare(request, workdir)
            argv = self.build_argv(prepared, request)
            provenance.update(
                argv=argv,
                executable=argv[0] if descriptor.runtime != "java" else self.resolve_executable()[0],
                executable_sha256=file_sha256(self.resolve_executable()[0] or argv[0]),
                input_sha256=file_sha256(prepared.problem),
                params_sha256=file_sha256(prepared.params) if prepared.params else None,
                scale=prepared.plan.to_json(),
                problem_type_line=prepared.problem_type_line,
                vehicles=prepared.vehicles,
            )
            if monitor is not None:
                if monitor.stop_requested:
                    return SolveResult(**base, wall_time=time.perf_counter() - started, status="cancelled",
                                       stopped_early=True, provenance=provenance)
                monitor.rescale(prepared.plan.factor)
                monitor.phase("solving")
            # A solution left by an earlier run in a reused workdir (--keep-artifacts, a resumed campaign)
            # must never be read back as this run's answer when the solver writes nothing.
            prepared.solution.unlink(missing_ok=True)
            grace = max(descriptor.timeout_grace_s, descriptor.timeout_grace_fraction * float(request.time_limit_s))
            env = dict(os.environ)
            env.update(descriptor.env)
            outcome = run_process(
                argv,
                cwd=workdir,
                log_path=workdir / LOG_FILE,
                time_limit_s=float(request.time_limit_s),
                grace_s=grace,
                term_grace_s=descriptor.term_grace_s,
                env=env,
                monitor=monitor,
                progress_pattern=descriptor.progress_pattern,
            )
            provenance.update(
                returncode=outcome.returncode,
                termination=outcome.termination,
                hard_timeout_s=float(request.time_limit_s) + grace,
                log_tail=outcome.log_tail[-2000:],
            )
            if monitor is not None:
                monitor.phase("reading solution")
            return self._result(base, prepared, outcome, provenance, started, monitor)
        finally:
            if own_workdir and not request.keep_artifacts:
                shutil.rmtree(workdir, ignore_errors=True)
            elif request.keep_artifacts or not own_workdir:
                provenance["artifacts_dir"] = str(workdir)

    def _result(
        self,
        base: dict[str, Any],
        prepared: PreparedInput,
        outcome: Any,
        provenance: dict[str, Any],
        started: float,
        monitor: SolveMonitor | None,
    ) -> SolveResult:
        from mamut_routing_lib.solutions import (
            SolutionFormatError,
            normalize_routes,
            parse_lkh_mtsp_solution,
            parse_solution_text,
        )

        wall = time.perf_counter() - started
        plan = prepared.plan
        common = dict(
            **base,
            wall_time=wall,
            cost_scale=plan.factor,
            cost_exact=plan.exact,
            stopped_early=outcome.termination == "cancelled",
            provenance=provenance,
            metadata={"progress_best_cost": None if outcome.best_cost is None else outcome.best_cost / plan.factor},
        )
        if outcome.termination == "error":
            return SolveResult(**common, status="failed", error=outcome.error)
        if outcome.termination == "cancelled":
            return SolveResult(**common, status="cancelled")
        solution = prepared.solution
        if not solution.is_file():
            status = "timeout" if outcome.termination == "timeout" else "failed"
            reason = "the solver hit the hard timeout" if status == "timeout" else f"no solution file (returncode {outcome.returncode})"
            return SolveResult(**common, status=status, error=reason)
        provenance["solution_sha256"] = file_sha256(solution)
        text = solution.read_text(encoding="utf-8", errors="replace")
        num_customers = int(prepared.checkable.num_customers)
        try:
            if self.descriptor.output_format == "lkh-mtsp":
                raw_routes, penalty, raw_cost = parse_lkh_mtsp_solution(text)
                if penalty:
                    return SolveResult(**common, status="infeasible", solver_cost_raw=raw_cost,
                                       error=f"{self.label} reports a penalty of {penalty:g} (no feasible solution)")
                routes = normalize_routes([[node - 1 for node in route] for route in raw_routes], num_customers, convention="one-based")
            else:
                raw_routes, raw_cost = parse_solution_text(text, filename=solution.name)
                routes = normalize_routes(raw_routes, num_customers)
        except SolutionFormatError as error:
            return SolveResult(**common, status="failed", error=f"unreadable solution file: {error}")
        solver_cost, exact_cost = unscale_cost(raw_cost, plan)
        # Only integer-cost solvers report a cost that can match the checker's exact decimal cost;
        # HGS-CVRP prints six significant digits, so its integral-looking costs are rounded.
        if exact_cost is not None and self.capabilities.integer_costs:
            try:
                from mamut_routing_lib.checker import compute_exact_solution_cost

                checker_exact = compute_exact_solution_cost(prepared.checkable, routes)
                provenance["reported_cost_matches_checker"] = checker_exact == exact_cost
            except Exception:  # noqa: BLE001 - the checker will reject the routes anyway
                provenance["reported_cost_matches_checker"] = False
        status = "timeout" if outcome.termination == "timeout" else "feasible"
        error = None
        if outcome.returncode not in (0, None) and status == "feasible":
            error = f"the solver exited with code {outcome.returncode} after writing a solution"
        return SolveResult(**common, status=status, routes=routes, solver_cost=solver_cost, solver_cost_raw=raw_cost, error=error)

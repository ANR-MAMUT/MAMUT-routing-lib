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
   library's, and check an integer cost report against the weights in the exported matrix.

The routes still go through the reference checker afterwards
(``solvers.run.run_solver``): a solver's own cost is never trusted.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
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
from mamut_routing_lib.solvers.provenance import environment, file_sha256, java_release
from mamut_routing_lib.solvers.report import make_solver_report

__all__ = ["ExternalSolver", "PreparedInput", "find_on_path"]

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


def find_on_path(name: str) -> tuple[str | None, str | None]:
    """(the command ``name`` on PATH, or a hit that only a relative PATH entry gives).

    Only the absolute entries of PATH are searched. A relative entry (``.``, an empty one, ``bin``)
    is the working directory, which the solvers config never reads on purpose: a file called
    ``filo2`` in a benchmark tree would run, and be credited, as the solver. Such a hit is
    returned apart, for the message that says why the solver is not available.
    """
    entries = os.environ.get("PATH", os.defpath).split(os.pathsep)
    absolute = [entry for entry in entries if entry and os.path.isabs(entry)]
    found = shutil.which(name, path=os.pathsep.join(absolute)) if absolute else None
    if found and os.path.isabs(found):
        return found, None
    return None, (shutil.which(name) if len(absolute) < len(entries) else None)


def _relative_hit_note(relative_hit: str | None) -> str:
    if not relative_hit:
        return ""
    return f" ({relative_hit} is only found through a relative PATH entry, which is not searched: give an absolute path)"


def _fmt_seconds(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _exported_route_cost(problem: Path, routes: list[list[int]], vertices: int) -> int:
    """Sum just the visited edges from the emitted FULL_MATRIX, in solver units."""
    successors = [[] for _ in range(vertices)]
    for route in routes:
        for source, target in pairwise([0, *route, 0]):
            successors[source].append(target)
    with problem.open(encoding="ascii") as stream:
        for line in stream:
            if line.strip() == "EDGE_WEIGHT_SECTION":
                break
        else:
            raise ValueError("missing EDGE_WEIGHT_SECTION")
        total = 0
        for neighbors in successors:
            weights = stream.readline().split()
            if len(weights) != vertices:
                raise ValueError("incomplete exported matrix row")
            total += sum(int(weights[target]) for target in neighbors)
    return total


def _solver_report(token: str | None, plan: ScalePlan, integer_model: bool) -> dict[str, Any] | None:
    if token is None:
        return None
    encoding = "integer" if integer_model and token.lstrip("+-").isdigit() else "decimal-text"
    return make_solver_report(
        "total-arc-cost", token, encoding=encoding, unit=plan.family_units_per_solver_unit,
    )


def _last_log_lines(log_tail: str, *, lines: int = 3, width: int = 200) -> str:
    """The last non-empty lines of a solver log on one line, for an error message."""
    kept = [line.strip() for line in log_tail.splitlines() if line.strip()][-lines:]
    return " | ".join(line if len(line) <= width else line[: width - 1] + "…" for line in kept)


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
        if self.descriptor.runtime == "python":
            return f"fix solvers.{self.name}.executable (the solver's script) in {where}"
        if self.descriptor.executable:
            return f"fix solvers.{self.name}.executable in {where}"
        names = self.descriptor.executable_names
        return f"set solvers.{self.name}.executable in {where}, or put {names[0] if names else self.name} on PATH"

    def resolve_executable(self) -> tuple[str | None, str | None]:
        """(absolute executable, jar or Python script, reason when missing).

        A Python solver names its script in ``executable``. Without one, its argv names a module or a
        script by itself, and this is the interpreter that runs it.
        """
        descriptor = self.descriptor
        if descriptor.runtime == "python":
            if not descriptor.executable:
                return sys.executable, None
            script = Path(descriptor.executable).expanduser()
            if not script.is_absolute() or not script.is_file():
                return None, f"script not found: {script}" + ("" if script.is_absolute() else " (give an absolute path)")
            return str(script.resolve()), None
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
            found, relative_hit = find_on_path(configured)
            if found:
                return str(Path(found).resolve()), None
            return None, f"configured executable {configured!r} is not on PATH" + _relative_hit_note(relative_hit)
        relative_hit = None
        for candidate in descriptor.executable_names:
            found, elsewhere = find_on_path(candidate)
            if found:
                return str(Path(found).resolve()), None
            relative_hit = relative_hit or elsewhere
        return None, "executable not found" + _relative_hit_note(relative_hit)

    def _java(self) -> tuple[str | None, str | None]:
        java = self.descriptor.java
        path = Path(java).expanduser()
        if path.is_absolute():
            return (str(path), None) if path.is_file() else (None, f"java not found: {path}")
        found, relative_hit = find_on_path(java)
        return (found, None) if found else (None, f"{java} not found on PATH" + _relative_hit_note(relative_hit))

    def solver_file(self) -> str | None:
        """The file that is the solver: the native executable, the jar, or the script of a Python solver.

        ``None`` for a Python solver that names no script: the interpreter is not the solver, and its
        hash would say nothing about the code that ran.
        """
        if self.descriptor.runtime == "python" and not self.descriptor.executable:
            return None
        return self.resolve_executable()[0]

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
        from mamut_routing_lib.artifacts import is_collection_instance
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
        checkable = request.checkable()  # hydrated once; run_solver's check reuses it
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
                            extra=descriptor.extra_args,  # literal parameter-file lines for this format
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
        return self._invocation(prepared, request)[0]

    def _invocation(self, prepared: PreparedInput, request: SolveRequest) -> tuple[list[str], str | None, dict[str, Any]]:
        """Resolve the command and identify the artifacts used in that invocation."""
        from mamut_routing_lib.solvers.base import SolverUnavailableError

        descriptor = self.descriptor
        executable, reason = self.resolve_executable()
        if executable is None:
            raise SolverUnavailableError(f"{self.name}: {reason} ({self._config_hint()})")
        java = None
        if descriptor.runtime == "java":
            java, reason = self._java()
            if java is None:
                raise SolverUnavailableError(f"{self.name}: {reason}")
        values = {
            "executable": executable,
            "jar": executable if descriptor.runtime == "java" else "",
            "java": java or descriptor.java,
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
        if descriptor.input_format != "lkh":  # LKH reads its extra_args from the parameter file
            argv.extend(token.format_map(values) for token in descriptor.extra_args)
        solver_file = None
        identity: dict[str, Any] = {}
        if descriptor.runtime == "python" and sys.executable in argv:
            identity.update(runtime_executable=sys.executable, runtime_version=platform.python_version())
            # A Python module or command may receive a script path as an ordinary argument.
            # The first positional interpreter argument identifies the executed script.
            options = iter(argv[argv.index(sys.executable) + 1:])
            for arg in options:
                if arg.startswith(("-m", "-c")):
                    break
                if arg in ("-W", "-X", "--check-hash-based-pycs"):
                    next(options, None)
                    continue
                if arg == "--":
                    continue
                if not arg.startswith("-"):
                    if descriptor.executable and arg == executable:
                        solver_file = executable
                    break
        elif descriptor.runtime == "java" and java in argv:
            identity.update(runtime_executable=str(Path(java).resolve()), runtime_version=java_release(java))
            java_args = argv[argv.index(java) + 1:]
            if ("-jar", executable) in pairwise(java_args):
                solver_file = executable
        elif descriptor.runtime == "native" and executable in argv:
            solver_file = executable
        return argv, solver_file, identity

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
                if monitor.stop_requested:  # cancelled before anything was written
                    return SolveResult(**base, wall_time=time.perf_counter() - started, status="cancelled", stopped_early=True,
                                       provenance=provenance)
                monitor.phase("preparing input")
            prepared = self.prepare(request, workdir)
            argv, solver_file, identity = self._invocation(prepared, request)
            provenance.update(
                argv=argv,
                executable=solver_file or argv[0],
                executable_sha256=file_sha256(solver_file) if solver_file else None,
                **identity,
                input_sha256=file_sha256(prepared.problem, cache=False),
                params_sha256=file_sha256(prepared.params, cache=False) if prepared.params else None,
                scale=prepared.plan.to_json(),
                problem_type_line=prepared.problem_type_line,
                vehicles=prepared.vehicles,
            )
            if monitor is not None:
                if monitor.stop_requested:
                    return SolveResult(**base, wall_time=time.perf_counter() - started, status="cancelled",
                                       stopped_early=True, provenance=provenance)
                scale = 1 / prepared.plan.cost_unit if prepared.plan.cost_unit is not None else prepared.plan.factor
                assert scale is not None
                monitor.rescale(scale)
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
            resolve_convention,
        )

        wall = time.perf_counter() - started
        plan = prepared.plan
        common = dict(
            **base,
            wall_time=wall,
            cost_scale=plan.factor,
            stopped_early=outcome.termination == "cancelled",
            provenance=provenance,
            metadata={"progress_best_cost": unscale_cost(outcome.best_cost, plan)[0]},
        )
        if outcome.termination == "error":
            return SolveResult(**common, status="failed", error=outcome.error)
        if outcome.termination == "cancelled":
            return SolveResult(**common, status="cancelled")
        solution = prepared.solution
        if not solution.is_file():
            status = "timeout" if outcome.termination == "timeout" else "failed"
            reason = "the solver hit the hard timeout" if status == "timeout" else f"no solution file (returncode {outcome.returncode})"
            # The solver's own words are the diagnosis (a rejected input format, a missing library);
            # the work directory, log included, is usually gone by the time the error is read.
            said = _last_log_lines(outcome.log_tail)
            if said:
                reason += f"; the solver's output ends with: {said}"
            return SolveResult(**common, status=status, error=reason)
        provenance["solution_sha256"] = file_sha256(solution, cache=False)
        text = solution.read_text(encoding="utf-8", errors="replace")
        num_customers = int(prepared.checkable.num_customers)
        integer_report = self.capabilities.integer_costs
        lkh = self.descriptor.output_format == "lkh-mtsp"
        try:
            if lkh:
                raw_routes, penalty, raw_cost = parse_lkh_mtsp_solution(text, exact_report="text")
            else:
                raw_routes, raw_cost = parse_solution_text(text, filename=solution.name, exact_report="text")
                penalty = None
            common["solver_report"] = _solver_report(raw_cost, plan, integer_report)
            if penalty:
                return SolveResult(**common, status="infeasible",
                                   error=f"{self.label} reports a penalty of {penalty} (no feasible solution)")
            if lkh:
                routes = normalize_routes([[node - 1 for node in route] for route in raw_routes], num_customers, convention="one-based")
            else:
                # The descriptor states the numbering: guessing it can complete an incomplete answer.
                convention = self.descriptor.id_convention
                provenance["solution_ids"] = resolve_convention(raw_routes, num_customers, convention)
                try:
                    routes = normalize_routes(raw_routes, num_customers, convention=convention)
                except SolutionFormatError as error:
                    if convention == "one-based" and resolve_convention(raw_routes, num_customers) == "zero-based":
                        raise SolutionFormatError(
                            f"{error} The stop ids cover 0..{num_customers - 1} exactly once: if {self.label} numbers "
                            f'customers from 0, declare id_convention = "zero-based" (solvers.{self.name} in the solvers config).'
                        ) from error
                    raise
        except SolutionFormatError as error:
            return SolveResult(**common, status="failed", error=f"unreadable solution file: {error}")
        remarks: list[str] = []
        match_key = "reported_cost_matches_exported_model"
        if raw_cost is not None and integer_report:
            reported = Decimal(raw_cost)
            if reported != reported.to_integral_value():
                remarks.append(f"{self.label} reported {raw_cost} solver units (fractional for this integer model)")
            if self.descriptor.input_format in ("lkh", "cvrplib-explicit"):
                try:
                    exported_cost = _exported_route_cost(prepared.problem, routes, num_customers + 1)
                except (OSError, ValueError) as error:
                    remarks.append(f"could not compare the reported cost with the exported matrix: {error}")
                else:
                    provenance[match_key] = reported == exported_cost
                    if not provenance[match_key]:
                        remarks.append(f"{self.label} reported {raw_cost} solver units. Its routes cost "
                                       f"{exported_cost} units in the exported matrix")
        status = "timeout" if outcome.termination == "timeout" else "feasible"
        if outcome.returncode not in (0, None) and status == "feasible":
            remarks.append(f"the solver exited with code {outcome.returncode} after writing a solution")
        return SolveResult(
            **common,
            status=status,
            routes=routes,
            error="; ".join(remarks) or None,
        )

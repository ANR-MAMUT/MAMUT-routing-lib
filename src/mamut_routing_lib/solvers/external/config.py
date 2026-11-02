"""The user solvers config: where external solvers live, and extra solvers.

Looked up, in order: an explicit path (``--solvers-config``), the
``MAMUT_ROUTING_SOLVERS_CONFIG`` environment variable (an empty value disables
the config), then ``$XDG_CONFIG_HOME/mamut-routing/solvers.toml`` (default
``~/.config``; ``%APPDATA%`` on Windows). It is never read from the working
directory or a benchmark tree: the file names executables that will be run.

.. code-block:: toml

    schema_version = 1

    [solvers.filo2]                       # override a built-in
    executable = "/opt/filo2/build/filo2"
    version = "43653e2"

    [solvers.ails2]
    jar = "/opt/ails2/AILSII.jar"         # absolute paths only for jars
    java = "/opt/jdk-21/bin/java"

    [solvers.lkh3]
    executable = "~/bin/LKH"
    extra_args = ["POPULATION_SIZE = 10"] # LKH-3: extra parameter-file lines

    [solvers.hgs-fork]                    # a variant of a built-in
    extends = "hgs-cvrp"
    executable = "/opt/hgs-fork/hgs"
    label = "HGS fork"

    [solvers.my-solver]                   # a new solver
    label = "My solver"
    executable = "/opt/mine/solve"
    argv = ["{executable}", "{problem}", "{solution}", "{time_limit_s}", "{seed}"]
    input_format = "cvrplib-explicit"     # cvrplib-x | cvrplib-matrix | cvrplib-explicit | lkh
    output_format = "cvrplib-sol"         # cvrplib-sol | lkh-mtsp
    problem_types = ["CVRP", "VRPTW"]
    objectives = ["MonoCost"]
    symmetric_only = false                # it reads asymmetric FULL_MATRIX files (default: true)

Unknown keys are errors, so a typo never silently runs a different command. A new
solver is ``symmetric_only = true`` unless it declares otherwise: road-network
variants (``fastest``, ``shortest``) are asymmetric and are refused for it, and
the generic explicit file says ``TYPE : CVRP`` whatever the matrix. A solver
that ``extends`` a built-in inherits the built-in's value (LKH-3 takes
asymmetric matrices; FILO2, HGS-CVRP and AILS-II do not).
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers.base import SolverCapabilities
from mamut_routing_lib.solvers.external.descriptor import ExternalSolverDescriptor
from mamut_routing_lib.solvers.external.scaling import Limits

__all__ = [
    "CONFIG_ENV",
    "SCHEMA_VERSION",
    "SolverConfigError",
    "clear_cache",
    "config_cache_key",
    "configured_descriptors",
    "configured_solvers",
    "default_config_path",
    "resolve_config_path",
]

CONFIG_ENV = "MAMUT_ROUTING_SOLVERS_CONFIG"
SCHEMA_VERSION = 1

_SCALAR_FIELDS = {
    "label": str,
    "method": str,
    "executable": str,
    "jar": str,
    "java": str,
    "runtime": str,
    "solution_file": str,
    "distance_model": str,
    "input_format": str,
    "output_format": str,
    "progress_pattern": str,
    "version": str,
    "requires_note": str,
    "homepage": str,
    "require_exact_scale": bool,
    "vehicles_factor": float,
    "vrptw_vehicles_factor": float,
    "timeout_grace_s": float,
    "timeout_grace_fraction": float,
    "term_grace_s": float,
}
_LIST_FIELDS = {"argv", "executable_names", "java_options", "fleet_args", "extra_args"}
_CAPABILITY_KEYS = {"problem_types", "objectives", "symmetric_only", "fleet_limit", "live_progress", "warm_start"}
_OTHER_KEYS = {"extends", "env", "limits", "coordinate_scales", "matrix_scales"}
_ALLOWED = set(_SCALAR_FIELDS) | _LIST_FIELDS | _CAPABILITY_KEYS | _OTHER_KEYS
_LIMIT_KEYS = {"max_arc_cost", "max_total_cost", "max_time_value", "max_demand", "min_arc_cost"}

_CACHE: dict[tuple[str, int], dict[str, ExternalSolverDescriptor]] = {}


class SolverConfigError(ValueError):
    """The solvers config cannot be read or describes an impossible solver."""


def default_config_path() -> Path:
    if sys.platform == "win32":  # pragma: no cover
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "mamut-routing" / "solvers.toml"


def resolve_config_path(config: str | Path | None = None) -> Path | None:
    """The config file to read, or ``None`` (an explicit path must exist)."""
    if config is not None and str(config) != "":
        path = Path(config).expanduser()
        if not path.is_file():
            raise SolverConfigError(f"solvers config not found: {path}")
        return path.resolve()
    if CONFIG_ENV in os.environ:
        value = os.environ[CONFIG_ENV]
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise SolverConfigError(f"${CONFIG_ENV} points at a missing file: {path}")
        return path.resolve()
    path = default_config_path()
    return path.resolve() if path.is_file() else None


def config_cache_key(config: str | Path | None = None) -> str:
    try:
        path = resolve_config_path(config)
    except SolverConfigError:
        return f"error:{config}"
    if path is None:
        return ""
    try:
        return f"{path}:{path.stat().st_mtime_ns}"
    except OSError:
        return str(path)


def clear_cache() -> None:
    _CACHE.clear()


def _problem_types(values: Any, name: str) -> list[ProblemType]:
    try:
        return [ProblemType(str(value).upper()) for value in values]
    except ValueError as error:
        raise SolverConfigError(f"solvers.{name}.problem_types: {error}") from error


def _objectives(values: Any, name: str) -> frozenset[ObjectiveFunction]:
    try:
        return frozenset(ObjectiveFunction(str(value)) for value in values)
    except ValueError as error:
        raise SolverConfigError(f"solvers.{name}.objectives: {error}") from error


def _descriptor(name: str, table: dict[str, Any], *, source: str, known: dict[str, ExternalSolverDescriptor]) -> ExternalSolverDescriptor:
    from mamut_routing_lib.solvers.external.catalog import BUILTIN_DESCRIPTORS

    unknown = sorted(set(table) - _ALLOWED)
    if unknown:
        raise SolverConfigError(f"solvers.{name}: unknown key(s) {', '.join(unknown)}")
    parent_name = table.get("extends", name if name in BUILTIN_DESCRIPTORS else None)
    parent = None
    if parent_name is not None:
        parent = known.get(parent_name) or BUILTIN_DESCRIPTORS.get(parent_name)
        if parent is None:
            raise SolverConfigError(f"solvers.{name}.extends: unknown solver {parent_name!r}")
    values: dict[str, Any] = {}
    for key, kind in _SCALAR_FIELDS.items():
        if key in table:
            value = table[key]
            if kind is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, kind):
                raise SolverConfigError(f"solvers.{name}.{key} must be a {kind.__name__}")
            values[key] = value
    for key in _LIST_FIELDS:
        if key in table:
            value = table[key]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SolverConfigError(f"solvers.{name}.{key} must be a list of strings")
            values[key] = tuple(value)
    if "env" in table:
        env = table["env"]
        if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
            raise SolverConfigError(f"solvers.{name}.env must be a table of strings")
        values["env"] = {**(dict(parent.env) if parent else {}), **env}
    for key in ("coordinate_scales", "matrix_scales"):
        if key in table:
            values[key] = tuple(float(value) for value in table[key])
    if "limits" in table:
        limits = table["limits"]
        if not isinstance(limits, dict) or set(limits) - _LIMIT_KEYS:
            raise SolverConfigError(f"solvers.{name}.limits accepts {', '.join(sorted(_LIMIT_KEYS))}")
        base_limits = parent.limits if parent else Limits()
        values["limits"] = Limits(**{**{k: getattr(base_limits, k) for k in _LIMIT_KEYS}, **{k: float(v) for k, v in limits.items()}})
    if "jar" in values and not Path(values["jar"]).expanduser().is_absolute():
        raise SolverConfigError(f"solvers.{name}.jar must be an absolute path")
    if "executable" in values:
        values["executable"] = str(Path(values["executable"]).expanduser()) if values["executable"].startswith("~") else values["executable"]

    capability_values = {key: table[key] for key in _CAPABILITY_KEYS if key in table}
    if parent is None:
        missing = [key for key in ("argv", "input_format", "output_format", "problem_types") if key not in table]
        if missing:
            raise SolverConfigError(f"solvers.{name}: a new solver needs {', '.join(missing)}")
    input_format = values.get("input_format", parent.input_format if parent else None)
    if input_format not in ("cvrplib-x", "cvrplib-matrix", "cvrplib-explicit", "lkh"):
        raise SolverConfigError(f"solvers.{name}.input_format must be cvrplib-x, cvrplib-matrix, cvrplib-explicit or lkh")
    if values.get("output_format", parent.output_format if parent else None) not in ("cvrplib-sol", "lkh-mtsp"):
        raise SolverConfigError(f"solvers.{name}.output_format must be cvrplib-sol or lkh-mtsp")
    if capability_values or parent is None or "input_format" in values:
        base_caps = parent.capabilities if parent else None
        objectives = _objectives(capability_values.get("objectives", ["MonoCost"]), name) if (
            "objectives" in capability_values or parent is None
        ) else None
        if "problem_types" in capability_values:
            problems = _problem_types(capability_values["problem_types"], name)
            objective_map = {
                problem: (objectives if objectives is not None else (base_caps.objectives.get(problem) if base_caps else None))
                or frozenset({ObjectiveFunction.MONO_COST})
                for problem in problems
            }
        elif base_caps is not None:
            objective_map = {
                problem: objectives if objectives is not None else objs for problem, objs in base_caps.objectives.items()
            }
        else:  # pragma: no cover - guarded by the "missing" check above
            objective_map = {}
        if any(problem in (ProblemType.TDVRP, ProblemType.TDVRPTW) for problem in objective_map):
            raise SolverConfigError(f"solvers.{name}: external solvers take static CVRP/VRPTW instances only")
        if ProblemType.VRPTW in objective_map and input_format in ("cvrplib-x", "cvrplib-matrix"):
            raise SolverConfigError(f"solvers.{name}: the {input_format} input format has no time windows")
        values["capabilities"] = SolverCapabilities(
            objectives=objective_map,
            input="coordinates" if input_format == "cvrplib-x" else "explicit",
            # A solver declared here is assumed to need symmetric costs until it says otherwise: one that
            # silently symmetrises an asymmetric FULL_MATRIX would optimise a different instance without error.
            symmetric_only=bool(capability_values.get("symmetric_only", base_caps.symmetric_only if base_caps else True)),
            integer_costs=input_format in ("cvrplib-explicit", "lkh") or bool(base_caps and base_caps.integer_costs and input_format == "cvrplib-x"),
            fleet_limit=bool(capability_values.get("fleet_limit", base_caps.fleet_limit if base_caps else False)),
            warm_start=False,
            live_progress=str(capability_values.get("live_progress", base_caps.live_progress if base_caps else "elapsed")),  # type: ignore[arg-type]
            cancellable=True,
            in_process=False,
        )
    values["source"] = source
    if parent is not None:
        renamed = {"name": name} if name != parent.name else {}
        if name != parent.name and "method" not in values:
            renamed["method"] = name
        if name != parent.name and "label" not in values:
            renamed["label"] = name
        return parent.with_overrides(**renamed, **values)
    values.setdefault("label", name)
    values.setdefault("method", name)
    return ExternalSolverDescriptor(
        name=name,
        argv=values.pop("argv"),
        input_format=values.pop("input_format"),
        output_format=values.pop("output_format"),
        capabilities=values.pop("capabilities"),
        **values,
    )


def configured_descriptors(config: str | Path | None = None) -> dict[str, ExternalSolverDescriptor]:
    """Descriptors defined or overridden by the config (empty without a config)."""
    path = resolve_config_path(config)
    if path is None:
        return {}
    key = (str(path), path.stat().st_mtime_ns)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SolverConfigError(f"{path}: {error}") from error
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise SolverConfigError(f"{path}: schema_version {version} is not supported (expected {SCHEMA_VERSION})")
    unknown = sorted(set(data) - {"schema_version", "solvers"})
    if unknown:
        raise SolverConfigError(f"{path}: unknown top-level key(s) {', '.join(unknown)}")
    solvers = data.get("solvers", {})
    if not isinstance(solvers, dict):
        raise SolverConfigError(f"{path}: [solvers] must be a table")
    descriptors: dict[str, ExternalSolverDescriptor] = {}
    for name, table in solvers.items():
        if name in ("pyvrp", "kayros"):
            raise SolverConfigError(f"solvers.{name}: in-process solvers are not configured here")
        if not isinstance(table, dict):
            raise SolverConfigError(f"solvers.{name} must be a table")
        descriptors[name] = _descriptor(name, table, source=str(path), known=descriptors)
    _CACHE[key] = descriptors
    return descriptors


def configured_solvers(config: str | Path | None = None) -> dict[str, Callable[[], Any]]:
    """Registry factories for every configured external solver."""
    from mamut_routing_lib.solvers.external.adapter import ExternalSolver

    path = resolve_config_path(config)
    return {
        name: (lambda descriptor=descriptor: ExternalSolver(descriptor, config_path=str(path)))
        for name, descriptor in configured_descriptors(config).items()
    }

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
    jar = "/opt/ails2/AILSII.jar"
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

``mamut-routing solvers init`` writes a commented starter file (``starter_config_text``).

Paths: ``~`` is expanded, and a relative path is read from the config file's folder,
never the working directory: ``jar = "AILSII.jar"``, ``executable = "./bin/filo2"``.
A bare ``executable`` or ``java`` name (``filo2``, ``java``) is looked up on PATH.

Unknown keys and mistyped values (``fleet_limit = "false"``) are errors, so a typo
never silently runs a different command. An error in one ``[solvers.<name>]`` table
disables that solver only: it is listed as unavailable with the error, and every
other solver keeps working (a file that is not valid TOML disables all of them).
``extends`` may name a solver defined anywhere in the file. A new solver is ``symmetric_only = true`` unless it declares otherwise: road-network
variants (``fastest``, ``shortest``) are asymmetric and are refused for it, and
the generic explicit file says ``TYPE : CVRP`` whatever the matrix. A solver
that ``extends`` a built-in inherits the built-in's value (LKH-3 takes
asymmetric matrices; FILO2, HGS-CVRP and AILS-II do not).
"""

from __future__ import annotations

import difflib
import os
import re
import string
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers.base import Availability, SolverCapabilities, SolverUnavailableError, Support
from mamut_routing_lib.solvers.external.descriptor import ExternalSolverDescriptor
from mamut_routing_lib.solvers.external.scaling import Limits

__all__ = [
    "CONFIG_ENV",
    "SCHEMA_VERSION",
    "BrokenSolver",
    "SolverConfig",
    "SolverConfigError",
    "clear_cache",
    "config_cache_key",
    "configured_descriptors",
    "configured_solvers",
    "default_config_path",
    "load_config",
    "resolve_config_path",
    "starter_config_text",
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
_CHOICES = {
    "runtime": ("native", "java", "python"),
    "distance_model": ("float", "nint", "integer"),
    "input_format": ("cvrplib-x", "cvrplib-matrix", "cvrplib-explicit", "lkh"),
    "output_format": ("cvrplib-sol", "lkh-mtsp"),
    "live_progress": ("none", "elapsed", "incumbent"),
}
#: argv placeholders (see ``descriptor``); ``{java_options}`` is a whole argv token of its own.
_ARGV_FIELDS = frozenset(
    {"executable", "jar", "java", "python", "problem", "params", "solution", "workdir", "seed", "time_limit_s", "time_limit_int", "vehicles"}
)


@dataclass(frozen=True)
class SolverConfig:
    """A parsed solvers config: the solvers it defines, and the tables that could not be read."""

    path: Path | None
    descriptors: dict[str, ExternalSolverDescriptor] = field(default_factory=dict)
    #: ``[solvers.<name>]`` -> why that table was refused (the solver is listed as unavailable).
    errors: dict[str, str] = field(default_factory=dict)


_CACHE: dict[tuple[str, int], SolverConfig] = {}


class SolverConfigError(ValueError):
    """The solvers config cannot be read or describes an impossible solver."""


def _did_you_mean(key: str, choices: set[str] | tuple[str, ...]) -> str:
    close = difflib.get_close_matches(key, sorted(choices), n=1, cutoff=0.7)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _type_name(kind: type) -> str:
    return {str: "a string", bool: "true or false (unquoted)", float: "a number", int: "an integer"}.get(kind, kind.__name__)


def _value_text(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else repr(value)


def _check_template(name: str, key: str, token: str, allowed: frozenset[str]) -> None:
    try:
        fields = [part[1] for part in string.Formatter().parse(token) if part[1] is not None]
    except ValueError as error:
        raise SolverConfigError(f"solvers.{name}.{key}: {token!r}: {error} (write {{{{ and }}}} for literal braces)") from error
    for field_name in fields:
        if field_name not in allowed:
            known = ", ".join("{" + item + "}" for item in sorted(allowed))
            raise SolverConfigError(
                f"solvers.{name}.{key}: unknown placeholder {{{field_name}}}{_did_you_mean(field_name, tuple(allowed))}; known: {known}"
            )


def _config_relative(value: str, base_dir: Path | None, *, on_path: bool = True) -> str:
    """``~`` expanded; a relative path is read from the config file's folder, except a bare
    name (no separator) when ``on_path``, which is looked up on PATH."""
    path = Path(value).expanduser()
    if path.is_absolute() or base_dir is None:
        return str(path)
    if not on_path or os.sep in value or (os.altsep and os.altsep in value):
        return str((base_dir / path).resolve())
    return value


class BrokenSolver:
    """Placeholder for a ``[solvers.<name>]`` table the config could not read: never available."""

    def __init__(self, name: str, error: str, *, config_path: str, base: ExternalSolverDescriptor | None = None):
        self.name = name
        self.label = base.label if base is not None and base.name == name else name
        self.capabilities = base.capabilities if base is not None else SolverCapabilities(objectives={}, in_process=False)
        self.error = error
        self.config_path = config_path
        self.descriptor = base.with_overrides(source=config_path) if base is not None else None

    def __repr__(self) -> str:
        return f"BrokenSolver({self.name!r}, {self.error!r})"

    def availability(self) -> Availability:
        return Availability(False, reason=f"config error: {self.error}", hint=f"fix [solvers.{self.name}] in {self.config_path}")

    def supports(self, profile: Any, objective: Any) -> Support:
        return Support.refuse(f"{self.name}: config error: {self.error}")

    def solve(self, request: Any, monitor: Any = None) -> Any:
        raise SolverUnavailableError(f"{self.name}: config error: {self.error} (fix [solvers.{self.name}] in {self.config_path})")


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


def _string_list(table: dict[str, Any], key: str, name: str) -> list[str]:
    value = table[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SolverConfigError(f"solvers.{name}.{key} must be a list of strings, e.g. {key} = [\"...\"]")
    return value


def _problem_types(values: list[str], name: str) -> list[ProblemType]:
    problems = []
    for value in values:
        try:
            problems.append(ProblemType(value.upper()))
        except ValueError:
            known = ", ".join(problem.value for problem in ProblemType)
            raise SolverConfigError(f"solvers.{name}.problem_types: unknown problem type {value!r} (known: {known})") from None
    return problems


def _objectives(values: list[str], name: str) -> frozenset[ObjectiveFunction]:
    objectives = set()
    for value in values:
        try:
            objectives.add(ObjectiveFunction(value))
        except ValueError:
            known = tuple(objective.value for objective in ObjectiveFunction)
            raise SolverConfigError(
                f"solvers.{name}.objectives: unknown objective {value!r}{_did_you_mean(value, known)} (known: {', '.join(known)})"
            ) from None
    return frozenset(objectives)


def _number(value: Any, name: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SolverConfigError(f"solvers.{name}.{key} must be a number, not {_value_text(value)}")
    return float(value)


def _descriptor(
    name: str,
    table: dict[str, Any],
    *,
    source: str,
    known: dict[str, ExternalSolverDescriptor],
    base_dir: Path | None = None,
) -> ExternalSolverDescriptor:
    from mamut_routing_lib.solvers.external.catalog import BUILTIN_DESCRIPTORS

    unknown = sorted(set(table) - _ALLOWED)
    if unknown:
        details = ", ".join(f"{key!r}{_did_you_mean(key, _ALLOWED)}" for key in unknown)
        raise SolverConfigError(f"solvers.{name}: unknown key(s) {details}")
    parent_name = table.get("extends", name if name in BUILTIN_DESCRIPTORS else None)
    if parent_name is not None and not isinstance(parent_name, str):
        raise SolverConfigError(f"solvers.{name}.extends must be a solver name (a string)")
    parent = None
    if parent_name is not None:
        parent = (known.get(parent_name) if parent_name != name else None) or BUILTIN_DESCRIPTORS.get(parent_name)
        if parent is None:
            builtins = ", ".join(BUILTIN_DESCRIPTORS)
            raise SolverConfigError(
                f"solvers.{name}.extends: unknown solver {parent_name!r}{_did_you_mean(parent_name, tuple(BUILTIN_DESCRIPTORS))}"
                f" (built-in external solvers: {builtins}, or another [solvers.<name>] of this file)"
            )
    values: dict[str, Any] = {}
    for key, kind in _SCALAR_FIELDS.items():
        if key not in table:
            continue
        value = table[key]
        if kind is float:
            value = _number(value, name, key)
        elif not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
            raise SolverConfigError(f"solvers.{name}.{key} must be {_type_name(kind)}, not {_value_text(value)}")
        if key in _CHOICES and value not in _CHOICES[key]:
            raise SolverConfigError(
                f"solvers.{name}.{key} must be one of {', '.join(_CHOICES[key])}, not {value!r}{_did_you_mean(value, _CHOICES[key])}"
            )
        values[key] = value
    for key in _LIST_FIELDS:
        if key in table:
            values[key] = tuple(_string_list(table, key, name))
    if "env" in table:
        env = table["env"]
        if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
            raise SolverConfigError(f"solvers.{name}.env must be a table of strings, e.g. env = {{ OMP_NUM_THREADS = \"1\" }}")
        values["env"] = {**(dict(parent.env) if parent else {}), **env}
    for key in ("coordinate_scales", "matrix_scales"):
        if key in table:
            raw = table[key]
            if not isinstance(raw, list) or not raw:
                raise SolverConfigError(f"solvers.{name}.{key} must be a non-empty list of numbers")
            scales = tuple(_number(value, name, key) for value in raw)
            if any(scale <= 0 for scale in scales):
                raise SolverConfigError(f"solvers.{name}.{key}: scale factors must be positive")
            values[key] = scales
    if "limits" in table:
        limits = table["limits"]
        if not isinstance(limits, dict) or set(limits) - _LIMIT_KEYS:
            extra = sorted(set(limits) - _LIMIT_KEYS) if isinstance(limits, dict) else []
            hint = "".join(_did_you_mean(key, _LIMIT_KEYS) for key in extra[:1])
            raise SolverConfigError(f"solvers.{name}.limits accepts {', '.join(sorted(_LIMIT_KEYS))}{hint}")
        base_limits = parent.limits if parent else Limits()
        values["limits"] = Limits(
            **{**{k: getattr(base_limits, k) for k in _LIMIT_KEYS}, **{k: _number(v, name, f"limits.{k}") for k, v in limits.items()}}
        )
    for key in ("executable", "jar", "java"):
        if key in values:
            values[key] = _config_relative(values[key], base_dir, on_path=key != "jar")
    if "jar" in values and not Path(values["jar"]).is_absolute():
        raise SolverConfigError(f"solvers.{name}.jar must be a path to the jar (absolute, or relative to the config file)")
    if "progress_pattern" in values:
        try:
            re.compile(values["progress_pattern"])
        except re.error as error:
            raise SolverConfigError(f"solvers.{name}.progress_pattern is not a valid regular expression: {error}") from error
    if "solution_file" in values:
        _check_template(name, "solution_file", values["solution_file"], frozenset({"seed"}))

    if parent is None:
        missing = [key for key in ("argv", "input_format", "output_format", "problem_types") if key not in table]
        if missing:
            raise SolverConfigError(f"solvers.{name}: a new solver needs {', '.join(missing)} (or extends = \"<solver>\")")
    argv = values.get("argv", parent.argv if parent else ())
    for token in argv:
        if token != "{java_options}":
            _check_template(name, "argv", token, _ARGV_FIELDS)
    for token in values.get("fleet_args", ()):
        _check_template(name, "fleet_args", token, _ARGV_FIELDS)
    input_format = values.get("input_format", parent.input_format if parent else None)
    if input_format != "lkh":  # with the lkh format, extra_args are literal parameter-file lines
        for token in values.get("extra_args", ()):
            _check_template(name, "extra_args", token, _ARGV_FIELDS)

    capability_values: dict[str, Any] = {}
    for key in ("problem_types", "objectives"):
        if key in table:
            capability_values[key] = _string_list(table, key, name)
    for key in ("symmetric_only", "fleet_limit", "warm_start"):
        if key in table:
            if not isinstance(table[key], bool):
                raise SolverConfigError(f"solvers.{name}.{key} must be true or false (unquoted), not {_value_text(table[key])}")
            capability_values[key] = table[key]
    if capability_values.get("warm_start"):
        raise SolverConfigError(f"solvers.{name}.warm_start: external solvers take no warm start (only PyVRP does)")
    if "live_progress" in table:
        live = table["live_progress"]
        if live not in _CHOICES["live_progress"]:
            raise SolverConfigError(f"solvers.{name}.live_progress must be one of {', '.join(_CHOICES['live_progress'])}, not {_value_text(live)}")
        capability_values["live_progress"] = live

    input_format = values.get("input_format", parent.input_format if parent else None)
    if input_format not in _CHOICES["input_format"]:  # pragma: no cover - checked above
        raise SolverConfigError(f"solvers.{name}.input_format must be one of {', '.join(_CHOICES['input_format'])}")
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
            symmetric_only=capability_values.get("symmetric_only", base_caps.symmetric_only if base_caps else True),
            integer_costs=input_format in ("cvrplib-explicit", "lkh") or bool(base_caps and base_caps.integer_costs and input_format == "cvrplib-x"),
            fleet_limit=capability_values.get("fleet_limit", base_caps.fleet_limit if base_caps else False),
            warm_start=False,
            live_progress=capability_values.get("live_progress", base_caps.live_progress if base_caps else "elapsed"),
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


def load_config(config: str | Path | None = None) -> SolverConfig:
    """Parse the solvers config. File-level problems (not TOML, wrong schema) raise
    ``SolverConfigError``; a bad ``[solvers.<name>]`` table only lands in ``errors``."""
    path = resolve_config_path(config)
    if path is None:
        return SolverConfig(None)
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
        details = ", ".join(f"{key!r}{_did_you_mean(key, ('schema_version', 'solvers'))}" for key in unknown)
        raise SolverConfigError(f"{path}: unknown top-level key(s) {details}; solvers go in [solvers.<name>] tables")
    solvers = data.get("solvers", {})
    if not isinstance(solvers, dict):
        raise SolverConfigError(f"{path}: [solvers] must be a table")
    descriptors: dict[str, ExternalSolverDescriptor] = {}
    errors: dict[str, str] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> None:
        if name in descriptors or name in errors:
            return
        table = solvers[name]
        if name in ("pyvrp", "kayros"):
            errors[name] = f"solvers.{name}: in-process solvers are not configured here"
            return
        if not isinstance(table, dict):
            errors[name] = f"solvers.{name} must be a table ([solvers.{name}])"
            return
        parent_name = table.get("extends")
        if isinstance(parent_name, str) and parent_name != name and parent_name in solvers:
            if parent_name in chain:
                errors[name] = f"solvers.{name}.extends: circular ({' -> '.join((*chain, name, parent_name))})"
                return
            resolve(parent_name, (*chain, name))
            if parent_name in errors:
                parent_error = errors[parent_name]
                errors[name] = parent_error if "circular" in parent_error else (
                    f"solvers.{name}.extends: [solvers.{parent_name}] has a config error"
                )
                return
        try:
            descriptors[name] = _descriptor(name, table, source=str(path), known=descriptors, base_dir=path.parent)
        except SolverConfigError as error:
            errors[name] = str(error)

    for name in solvers:
        resolve(name, ())
    ordered = {name: descriptors[name] for name in solvers if name in descriptors}
    loaded = SolverConfig(path, ordered, {name: errors[name] for name in solvers if name in errors})
    _CACHE[key] = loaded
    return loaded


def configured_descriptors(config: str | Path | None = None) -> dict[str, ExternalSolverDescriptor]:
    """Descriptors defined or overridden by the config (empty without a config).

    Strict: raises ``SolverConfigError`` for the first table with an error. The registry
    uses ``load_config`` instead, so one bad table does not hide the other solvers.
    """
    loaded = load_config(config)
    if loaded.errors:
        raise SolverConfigError(next(iter(loaded.errors.values())))
    return loaded.descriptors


def configured_solvers(config: str | Path | None = None) -> dict[str, Callable[[], Any]]:
    """Registry factories for every configured external solver; a table with an error
    yields a ``BrokenSolver`` that reports it."""
    from mamut_routing_lib.solvers.external.adapter import ExternalSolver
    from mamut_routing_lib.solvers.external.catalog import BUILTIN_DESCRIPTORS

    loaded = load_config(config)
    path = str(loaded.path)
    factories: dict[str, Callable[[], Any]] = {
        name: (lambda descriptor=descriptor: ExternalSolver(descriptor, config_path=path))
        for name, descriptor in loaded.descriptors.items()
    }
    for name, error in loaded.errors.items():
        if name in ("pyvrp", "kayros"):
            continue  # never shadows the in-process solver; ``solvers list`` reports the error
        factories[name] = lambda name=name, error=error: BrokenSolver(
            name, error, config_path=path, base=BUILTIN_DESCRIPTORS.get(name)
        )
    return factories


# --------------------------------------------------------------------------- #
# Starter file (``mamut-routing solvers init``)
# --------------------------------------------------------------------------- #

_STARTER_HEADER = """\
# MAMUT-routing solvers config, written by `mamut-routing solvers init`.
#
# Read by `mamut-routing solve --solver NAME`, `mamut-tools solve`, the workbench GUI
# and `mamut-tools bks-campaign`. Uncomment the table of each solver you have built,
# set its path, then check with `mamut-routing solvers list`.
#
# - Relative paths are read from this file's folder; a bare executable name is
#   looked up on PATH. Booleans are unquoted (true / false).
# - A mistake in one table disables that solver only; `solvers list` says why.
# - `version` is recorded in the provenance of every BKS a run stores.
schema_version = 1
"""

_STARTER_SOLVERS = (
    (
        "filo2",
        (
            "FILO2 (CVRP) -- {homepage}\n"
            "Build with cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TIMELIMIT=ON (without it the time limit is ignored)."
        ),
        ('executable = "/path/to/filo2/build/filo2"', 'version = "17b8f84"'),
    ),
    (
        "hgs-cvrp",
        (
            "HGS-CVRP (CVRP) -- {homepage}\n"
            "Needs the full-matrix reader patch (upstream HGS-CVRP rejects EDGE_WEIGHT_SECTION), then a rebuild:\n"
            "  git -C HGS-CVRP checkout 1a92795 && git -C HGS-CVRP apply {patch}"
        ),
        ('executable = "/path/to/HGS-CVRP/build/hgs"', 'version = "1a92795"'),
    ),
    (
        "ails2",
        (
            "AILS-II (CVRP, Java 21) -- {homepage}\n"
            "Needs a build with the -seed/-out patch (upstream writes no routes and is not seedable):\n"
            "MAMUT-routing paper7 campaign3/patch_ails2.py."
        ),
        ('jar = "/path/to/AILSII.jar"', 'java = "java"                       # or the path of a JDK/JRE 21 java', 'version = "4ff3611"'),
    ),
    (
        "lkh3",
        "LKH-3 (CVRP, asymmetric CVRP, VRPTW) -- {homepage}",
        ('executable = "/path/to/LKH-3.0.x/LKH"', 'version = "3.0.6"'),
    ),
)

_STARTER_FOOTER = """\

# A variant of a built-in, listed under its own name:
# [solvers.lkh3-long]
# extends = "lkh3"
# executable = "/path/to/LKH-3.0.x/LKH"
# label = "LKH-3, 10 runs"
# extra_args = ["RUNS = 10"]

# A solver of your own (argv placeholders: {executable} {problem} {params} {solution}
# {workdir} {seed} {time_limit_s} {time_limit_int} {vehicles} {python} {java} {jar}):
# [solvers.my-solver]
# label = "My solver"
# executable = "/path/to/my-solver"
# argv = ["{executable}", "{problem}", "{solution}", "{time_limit_s}", "{seed}"]
# input_format = "cvrplib-explicit"   # cvrplib-x | cvrplib-matrix | cvrplib-explicit | lkh
# output_format = "cvrplib-sol"       # cvrplib-sol | lkh-mtsp
# problem_types = ["CVRP"]
# objectives = ["MonoCost"]
# symmetric_only = true               # false: it reads asymmetric FULL_MATRIX files
"""


def _comment(text: str) -> str:
    return "\n".join(f"# {line}" if line else "#" for line in text.splitlines())


def starter_config_text(*, detect: bool = True) -> str:
    """A commented solvers config: one table per built-in external solver, all commented out.

    With ``detect``, a solver already found on PATH is marked as such (it needs no table).
    The text parses as a valid config that defines no solver.
    """
    import shutil

    from mamut_routing_lib.solvers.external.catalog import BUILTIN_DESCRIPTORS, patch_path

    parts = [_STARTER_HEADER]
    for name, about, lines in _STARTER_SOLVERS:
        descriptor = BUILTIN_DESCRIPTORS[name]
        text = about.format(homepage=descriptor.homepage or "", patch=patch_path("hgs-cvrp-explicit-matrix.patch"))
        block = [_comment(text)]
        if detect and descriptor.runtime == "native":
            found = next((hit for candidate in descriptor.executable_names if (hit := shutil.which(candidate))), None)
            if found:
                block.append(f"# Found on PATH: {found} (used as is; a table is only needed to pin a version).")
        block.append(f"# [solvers.{name}]")
        block.extend(f"# {line}" for line in lines)
        parts.append("\n" + "\n".join(block) + "\n")
    parts.append(_STARTER_FOOTER)
    return "".join(parts)

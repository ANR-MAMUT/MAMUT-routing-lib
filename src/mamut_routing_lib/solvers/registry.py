"""Solvers by name.

Three sources, in increasing precedence:

1. built-ins: ``pyvrp``, ``kayros`` and the external executables of
   ``solvers.external.catalog`` (``filo2``, ``hgs-cvrp``, ``ails2``, ``lkh3``);
2. third-party packages, through the ``mamut_routing.solvers`` entry-point group
   (each entry point loads a ``Solver`` instance, class or zero-argument factory);
3. ``register_solver`` calls, and the user solver config
   (``solvers.external.config``: overrides of built-in external solvers and new
   external solvers).

Importing this module never imports PyVRP or KAYROS: solvers whose extra is
missing are listed and report ``availability().available == False``. The user
config cannot define in-process solvers, so ``get_solver("pyvrp")`` and
``get_solver("kayros")`` never read it: a broken config blocks only the
external solvers.
"""

from __future__ import annotations

import warnings
from importlib import metadata
from pathlib import Path
from typing import Callable, Union

from mamut_routing_lib.solvers.base import Solver

__all__ = [
    "BUILTIN_SOLVERS",
    "ENTRY_POINT_GROUP",
    "IN_PROCESS_SOLVERS",
    "UnknownSolverError",
    "get_solver",
    "list_solvers",
    "register_solver",
    "reload_solvers",
    "solver_names",
]

ENTRY_POINT_GROUP = "mamut_routing.solvers"
#: Built-in names, in listing order (the CLI help names these; user config can add more).
BUILTIN_SOLVERS = ("pyvrp", "kayros", "filo2", "hgs-cvrp", "ails2", "lkh3")
#: Built-ins that are Python packages, not configured executables (the user config refuses these
#: names, and a broken config never blocks them). PyVRP runs in this process; KAYROS in a child
#: Python process, so that a stop request can end it.
IN_PROCESS_SOLVERS = ("pyvrp", "kayros")

SolverSource = Union[Solver, Callable[[], Solver]]
ConfigPath = Union[str, Path, None]


class UnknownSolverError(ValueError):
    """No solver of that name; the message lists the known ones."""


_REGISTERED: dict[str, SolverSource] = {}
_CACHE: dict[tuple[str, str], Solver] = {}
_ENTRY_POINTS: dict[str, SolverSource] | None = None


def register_solver(solver: SolverSource, *, name: str | None = None, replace: bool = False) -> None:
    """Register a solver instance or zero-argument factory under ``name`` (default: its ``name``)."""
    key = name or getattr(solver, "name", None)
    if not key:
        raise ValueError("register_solver needs a name (the solver has no 'name' attribute)")
    if key in _REGISTERED and not replace:
        raise ValueError(f"a solver named {key!r} is already registered (pass replace=True)")
    _REGISTERED[key] = solver
    _CACHE.clear()


def reload_solvers() -> None:
    """Forget cached solver objects, entry points and the parsed user config."""
    global _ENTRY_POINTS
    _CACHE.clear()
    _ENTRY_POINTS = None
    try:
        from mamut_routing_lib.solvers.external import config as external_config

        external_config.clear_cache()
    except ImportError:  # pragma: no cover - the external package ships with the lib
        pass


def _builtin_factories() -> dict[str, SolverSource]:
    from mamut_routing_lib.solvers.pyvrp_adapter import PyVRPSolver

    factories: dict[str, SolverSource] = {"pyvrp": PyVRPSolver}
    try:
        from mamut_routing_lib.solvers.kayros import KayrosSolver
    except ModuleNotFoundError:  # pragma: no cover - partial installs only
        pass
    else:
        factories["kayros"] = KayrosSolver
    try:
        from mamut_routing_lib.solvers.external.catalog import builtin_external_factories
    except ModuleNotFoundError:  # pragma: no cover - partial installs only
        return factories
    factories.update(builtin_external_factories())
    return factories


def _entry_point_sources() -> dict[str, SolverSource]:
    global _ENTRY_POINTS
    if _ENTRY_POINTS is None:
        found: dict[str, SolverSource] = {}
        try:
            entries = metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception:  # noqa: BLE001 - a broken environment must not break listing  # pragma: no cover
            entries = ()
        for entry in entries:
            try:
                found[entry.name] = entry.load()
            except Exception as error:  # noqa: BLE001 - one broken plugin must not hide the others
                warnings.warn(f"solver entry point {entry.name!r} failed to load: {error}", stacklevel=2)
        _ENTRY_POINTS = found
    return _ENTRY_POINTS


def _config_sources(config: ConfigPath) -> dict[str, SolverSource]:
    try:
        from mamut_routing_lib.solvers.external.config import configured_solvers
    except ModuleNotFoundError:  # pragma: no cover - partial installs only
        return {}
    return configured_solvers(config)


def _sources(config: ConfigPath, *, with_config: bool = True) -> dict[str, SolverSource]:
    sources: dict[str, SolverSource] = {}
    sources.update(_builtin_factories())
    sources.update(_entry_point_sources())
    sources.update(_REGISTERED)
    if with_config:
        sources.update(_config_sources(config))
    return sources


def _instantiate(source: SolverSource) -> Solver:
    if isinstance(source, type):
        return source()
    if callable(source) and not hasattr(source, "solve"):
        return source()
    return source  # an instance


def _config_key(config: ConfigPath) -> str:
    try:
        from mamut_routing_lib.solvers.external.config import config_cache_key
    except ModuleNotFoundError:  # pragma: no cover - partial installs only
        return str(config or "")
    return config_cache_key(config)


def solver_names(*, config: ConfigPath = None) -> list[str]:
    """Every known solver name: built-ins first, then plugins, registrations and configured solvers."""
    sources = _sources(config)
    names = [name for name in BUILTIN_SOLVERS if name in sources]
    for name in sources:
        if name not in names:
            names.append(name)
    return names


def get_solver(name: str, *, config: ConfigPath = None) -> Solver:
    """The solver called ``name``; ``UnknownSolverError`` lists the known names."""
    in_process = name in IN_PROCESS_SOLVERS
    key = ("" if in_process else _config_key(config), name)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    sources = _sources(config, with_config=not in_process)
    if name not in sources:
        raise UnknownSolverError(f"unknown solver {name!r}; known solvers: {', '.join(solver_names(config=config))}")
    solver = _instantiate(sources[name])
    _CACHE[key] = solver
    return solver


def list_solvers(*, config: ConfigPath = None) -> list[Solver]:
    """Every known solver, available or not, in ``solver_names`` order."""
    return [get_solver(name, config=config) for name in solver_names(config=config)]

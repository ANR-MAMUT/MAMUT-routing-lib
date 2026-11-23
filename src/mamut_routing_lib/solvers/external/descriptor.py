"""Declarative description of an external solver executable.

A descriptor says how to call a solver (argv template), what it reads and
writes, its numeric limits and its capabilities; ``ExternalSolver`` does the
rest. Built-ins live in ``catalog``; users add or override descriptors in the
solvers config (``config``).

argv placeholders: ``{executable}`` (absolute path: the native executable, or
the script of a ``runtime = "python"`` solver), ``{java}``, ``{jar}``,
``{python}`` (this interpreter), ``{problem}``, ``{params}``, ``{solution}``,
``{workdir}`` (absolute paths), ``{seed}``, ``{time_limit_s}`` (``30`` or
``12.5``), ``{time_limit_int}`` (whole seconds, at least 1), ``{vehicles}``.
``fleet_args`` are appended (formatted) when the instance has a fixed fleet,
then ``extra_args`` -- except with ``input_format = "lkh"``, where ``extra_args``
are literal lines appended to the parameter file, whatever the argv.

``id_convention`` says how a ``cvrplib-sol`` solution file numbers its stops. It
is declared, not guessed: a file ``Route #1: 0 1 2`` for 3 customers is complete
if customers are numbered from 0, and misses customer 3 if they are numbered
from 1 with the depot written as 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Mapping

from mamut_routing_lib.solutions import IdConvention
from mamut_routing_lib.solvers.base import SolverCapabilities
from mamut_routing_lib.solvers.external.scaling import Limits

__all__ = ["ExternalSolverDescriptor", "InputFormat", "OutputFormat"]

InputFormat = Literal["cvrplib-x", "cvrplib-matrix", "cvrplib-explicit", "lkh"]
OutputFormat = Literal["cvrplib-sol", "lkh-mtsp"]


@dataclass(frozen=True)
class ExternalSolverDescriptor:
    name: str
    label: str
    method: str
    argv: tuple[str, ...]
    input_format: InputFormat
    output_format: OutputFormat
    capabilities: SolverCapabilities
    #: The solver file: a native executable, or the script of a ``runtime = "python"`` solver (run as
    #: ``{python} {executable}``). It is what ``executable_sha256`` hashes; a Java solver's is its ``jar``.
    executable: str | None = None
    executable_names: tuple[str, ...] = ()
    runtime: Literal["native", "java", "python"] = "native"
    jar: str | None = None
    java: str = "java"
    java_options: tuple[str, ...] = ()
    #: Solution path relative to the work directory (``{seed}`` is substituted).
    solution_file: str = "solution.sol"
    #: How a ``cvrplib-sol`` file numbers its stops: ``one-based`` (customers 1..n, a 0 is the depot:
    #: what CVRPLIB writes), ``zero-based`` (customers 0..n-1), or ``auto``, which reads a file that
    #: covers 0..n-1 exactly once as zero-based and anything else as one-based. ``auto`` reads a
    #: one-based file that writes the depot once and misses customer n as a complete zero-based
    #: one, so it is a choice and never the default. ``lkh-mtsp`` files have one numbering.
    id_convention: IdConvention = "one-based"
    #: How the solver measures distances on coordinates: ``nint`` rounds to integers.
    distance_model: Literal["float", "nint", "integer"] = "float"
    coordinate_scales: tuple[float, ...] = (1.0,)
    #: Scale factors tried, in order, for decimal full-matrix input (``cvrplib-matrix``).
    matrix_scales: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001)
    limits: Limits = field(default_factory=Limits)
    require_exact_scale: bool = False
    fleet_args: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    #: LKH-3 style fleet bound when the instance has no fixed fleet: ceil(factor * lower bound) + 1.
    vehicles_factor: float = 1.2
    vrptw_vehicles_factor: float = 2.0
    timeout_grace_s: float = 30.0
    timeout_grace_fraction: float = 0.1
    term_grace_s: float = 2.0
    progress_pattern: str | None = None
    version: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    requires_note: str | None = None
    homepage: str | None = None
    #: Where the descriptor came from: ``builtin`` or the config file that defined/overrode it.
    source: str = "builtin"

    def with_overrides(self, **values: Any) -> "ExternalSolverDescriptor":
        return replace(self, **values)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = self.capabilities.to_json()
        data["limits"] = {key: value for key, value in asdict(self.limits).items() if value is not None}
        data["env"] = dict(self.env)
        return data

    def sha256(self) -> str:
        """Hash of everything that shapes a run (not the source label)."""
        payload = {key: value for key, value in self.to_json().items() if key != "source"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

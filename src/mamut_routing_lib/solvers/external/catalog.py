"""Built-in descriptors for literature solvers with CVRPLIB-style input and output.

None of these executables ship with the library: point the solvers config at
your builds (``config``), or put them on ``PATH`` under the names below.
Each solver's reader/writer constraints come from its source code at the
commit named in ``homepage`` (read 2026-11-02), and from the Mamut2026 BKS
campaign 3 (2026-09), which ran FILO2, AILS-II and HGS-CVRP by hand:

- **FILO2** (Accorsi & Vigo): coordinates only, Euclidean distances rounded to
  integers -- so metre coordinates go in as millimetres, and the integer
  distances equal the published 3-decimal costs times 1000. Its time limit is
  an integer (``std::stoi``); build with ``-DENABLE_TIMELIMIT=ON``. It names its
  output ``<instance file>_seed-<seed>.vrp.sol``.
- **HGS-CVRP** (Vidal): its search handles asymmetric distances (README; the
  2-opt moves keep reversal costs) and takes a full matrix; only its CVRPLIB
  reader is coordinate-only. With the MAMUT-routing reader patch
  (``patches/hgs-cvrp-explicit-matrix.patch``) it reads the published matrix
  (``cvrplib-matrix``: decimal ``FULL_MATRIX`` plus coordinates for SWAP*'s polar
  sectors), so it solves every metric, ``fastest`` and ``shortest`` included.
  Float distances (``-round 0``); it refuses a largest distance above 1e5 or
  below 0.1 and demands above 1e5, so the matrix is divided by 10 when needed
  (exactly, in decimal); ``-veh`` fixes the fleet. Prints its cost with six
  significant digits.
- **AILS-II** (Maximo, Cordeau & Nascimento): a jar; ``-rounded false`` (the
  rounded mode overflows a ``short`` above 32 767), totals below 2^31 (its
  acceptance threshold is an ``int``). Upstream writes no routes and draws its
  randomness from the clock: it needs the ``-seed``/``-out`` patch
  (MAMUT-routing paper7 ``campaign3/patch_ails2.py``).
- **LKH-3** (Helsgaun): explicit integer matrices, including asymmetric
  (``ACVRP``) and time windows (``CVRPTW``); driven by a parameter file;
  writes an ``MTSP_SOLUTION_FILE``. Distributed for academic, non-commercial
  use by its author.
"""

from __future__ import annotations

from typing import Callable

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers.base import SolverCapabilities
from mamut_routing_lib.solvers.external.descriptor import ExternalSolverDescriptor
from mamut_routing_lib.solvers.external.scaling import Limits

__all__ = ["BUILTIN_DESCRIPTORS", "builtin_external_factories", "builtin_descriptor", "patch_path"]

_CVRP_MONO = {ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST})}
_THREADS_ONE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

FILO2 = ExternalSolverDescriptor(
    name="filo2",
    label="FILO2",
    method="filo2",
    argv=("{executable}", "{problem}", "--outpath", "{workdir}/", "--seed", "{seed}", "--optimization-seconds", "{time_limit_int}"),
    input_format="cvrplib-x",
    output_format="cvrplib-sol",
    capabilities=SolverCapabilities(
        objectives=_CVRP_MONO,
        input="coordinates",
        symmetric_only=True,
        integer_costs=True,
        fleet_limit=False,
        live_progress="elapsed",
    ),
    executable_names=("filo2",),
    solution_file="problem.vrp_seed-{seed}.vrp.sol",
    distance_model="nint",
    coordinate_scales=(1000.0,),
    limits=Limits(max_arc_cost=2.0e9, max_total_cost=2.0e15),
    env=_THREADS_ONE,
    homepage="https://github.com/acco93/filo2 (17b8f84; campaign 3 ran 43653e2)",
)

def patch_path(name: str) -> str:
    """Installed path of a solver patch shipped with the lib (``solvers/external/patches``)."""
    from importlib import resources

    return str(resources.files("mamut_routing_lib.solvers.external") / "patches" / name)


HGS_CVRP = ExternalSolverDescriptor(
    name="hgs-cvrp",
    label="HGS-CVRP",
    method="hgs-cvrp",
    argv=("{executable}", "{problem}", "{solution}", "-t", "{time_limit_s}", "-seed", "{seed}", "-round", "0", "-it", "1000000000"),
    input_format="cvrplib-matrix",
    output_format="cvrplib-sol",
    capabilities=SolverCapabilities(
        objectives=_CVRP_MONO,
        input="explicit",
        symmetric_only=False,
        fleet_limit=True,
        live_progress="incumbent",
    ),
    executable_names=("hgs", "hgs-cvrp"),
    distance_model="float",
    matrix_scales=(1.0, 0.1, 0.01, 0.001),
    limits=Limits(max_arc_cost=1.0e5, min_arc_cost=0.1, max_demand=1.0e5),
    fleet_args=("-veh", "{vehicles}"),
    progress_pattern=r"^It\s+(?P<iterations>\d+)\s+\d+\s+\|\s+T\(s\)\s+[\d.]+\s+\|\s+Feas\s+\d+\s+(?P<cost>[\d.]+)",
    env=_THREADS_ONE,
    requires_note=(
        "needs the full-matrix reader patch (upstream HGS-CVRP rejects EDGE_WEIGHT_SECTION): "
        "git apply " + patch_path("hgs-cvrp-explicit-matrix.patch") + " in vidalt/HGS-CVRP 1a92795, then rebuild"
    ),
    homepage="https://github.com/vidalt/HGS-CVRP (1a92795) + patches/hgs-cvrp-explicit-matrix.patch",
)

AILS2 = ExternalSolverDescriptor(
    name="ails2",
    label="AILS-II",
    method="ails2",
    argv=(
        "{java}", "{java_options}", "-jar", "{jar}", "-file", "{problem}", "-rounded", "false", "-best", "0",
        "-limit", "{time_limit_s}", "-stoppingCriterion", "Time", "-seed", "{seed}", "-out", "{solution}",
    ),
    input_format="cvrplib-x",
    output_format="cvrplib-sol",
    capabilities=SolverCapabilities(
        objectives=_CVRP_MONO,
        input="coordinates",
        symmetric_only=True,
        fleet_limit=False,
        live_progress="elapsed",
    ),
    runtime="java",
    java_options=("-Xmx4g", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=1"),
    distance_model="float",
    coordinate_scales=(1.0,),
    limits=Limits(max_total_cost=2.1e9),
    requires_note=(
        "needs a build with the -seed/-out patch (upstream AILS-II writes no routes and is not seedable): "
        "MAMUT-routing paper7 campaign3/patch_ails2.py"
    ),
    homepage="https://github.com/INFORMSJoC/2023.0106 (4ff3611) + patch",
)

LKH3 = ExternalSolverDescriptor(
    name="lkh3",
    label="LKH-3",
    method="lkh3",
    argv=("{executable}", "{params}"),
    input_format="lkh",
    output_format="lkh-mtsp",
    capabilities=SolverCapabilities(
        objectives={
            ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST}),
            ProblemType.VRPTW: frozenset({ObjectiveFunction.MONO_COST}),
        },
        input="explicit",
        symmetric_only=False,
        integer_costs=True,
        fleet_limit=True,
        live_progress="incumbent",
    ),
    executable_names=("LKH", "lkh", "LKH-3", "lkh3"),
    solution_file="solution.sol",
    distance_model="integer",
    # PRECISION = 1 in the parameter file; LKH keeps C(i, j) and penalties in 32-bit ints.
    limits=Limits(max_arc_cost=1.0e8, max_total_cost=2.0e9, max_time_value=2.0e9),
    progress_pattern=r"Cost = (?P<penalty>\d+)_(?P<cost>\d+)",
    env=_THREADS_ONE,
    homepage="http://webhotel4.ruc.dk/~keld/research/LKH-3/ (I/O read from LKH-3.0.6, github.com/c4v4/LKH3 5bd6368)",
)

BUILTIN_DESCRIPTORS: dict[str, ExternalSolverDescriptor] = {
    descriptor.name: descriptor for descriptor in (FILO2, HGS_CVRP, AILS2, LKH3)
}


def builtin_descriptor(name: str) -> ExternalSolverDescriptor:
    return BUILTIN_DESCRIPTORS[name]


def builtin_external_factories() -> dict[str, Callable[[], object]]:
    """Registry factories for the built-in external solvers (configured overrides come later)."""
    from mamut_routing_lib.solvers.external.adapter import ExternalSolver

    return {name: (lambda descriptor=descriptor: ExternalSolver(descriptor)) for name, descriptor in BUILTIN_DESCRIPTORS.items()}

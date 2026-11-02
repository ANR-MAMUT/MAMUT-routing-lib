"""Solver integrations: a registry of solvers behind one protocol.

    from mamut_routing_lib.solvers import run_solver

    solved = run_solver("pyvrp", "path/to/instance.vrp.json", time_limit_s=30, seed=1)
    solved.valid, solved.cost, solved.result.routes

Importing this package never imports PyVRP or KAYROS; see ``solvers.registry``
for the solver sources and ``solvers.base`` for the protocol.
"""

from mamut_routing_lib.solvers.base import (
    Availability,
    InstanceNotSupportedError,
    InstanceProfile,
    SolveMonitor,
    SolveRequest,
    SolveResult,
    SolveTick,
    Solver,
    SolverCapabilities,
    SolverUnavailableError,
    Support,
    profile_instance,
)
from mamut_routing_lib.solvers.registry import (
    BUILTIN_SOLVERS,
    UnknownSolverError,
    get_solver,
    list_solvers,
    register_solver,
    reload_solvers,
    solver_names,
)
from mamut_routing_lib.solvers.run import CheckedSolve, prepare_solve, run_solver

__all__ = [
    "Availability",
    "BUILTIN_SOLVERS",
    "CheckedSolve",
    "InstanceNotSupportedError",
    "InstanceProfile",
    "SolveMonitor",
    "SolveRequest",
    "SolveResult",
    "SolveTick",
    "Solver",
    "SolverCapabilities",
    "SolverUnavailableError",
    "Support",
    "UnknownSolverError",
    "get_solver",
    "list_solvers",
    "prepare_solve",
    "profile_instance",
    "register_solver",
    "reload_solvers",
    "run_solver",
    "solver_names",
]

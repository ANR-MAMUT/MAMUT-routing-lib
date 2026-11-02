"""External solver executables behind the ``Solver`` protocol (standard library only).

- ``scaling``: exact or approximate integer scaling of costs, times and coordinates;
- ``problem``: the problem files each solver family reads;
- ``process``: subprocess execution with process-group kill, timeouts and cancellation;
- ``descriptor`` / ``adapter``: a declarative ``ExternalSolverDescriptor`` and the
  ``ExternalSolver`` that runs one;
- ``catalog``: built-in descriptors (FILO2, HGS-CVRP, AILS-II, LKH-3);
- ``config``: the user TOML config that points at executables and adds solvers.
"""

"""Problem files for external solvers.

These are solver *inputs*, written into a scratch directory with fixed file
names, not the published CVRPLIB export of ``mamut_routing_lib.cvrplib`` (whose
bytes are pinned by the committed ``.vrp`` files and the website's writer).
Each reader is stricter than the format suggests, which is why there are three
writers:

- ``cvrplib-x``, the Uchoa X-set layout, for coordinate solvers. FILO2's
  ``fscanf`` parser needs exactly ``NAME``, ``COMMENT``, ``TYPE``,
  ``DIMENSION``, ``EDGE_WEIGHT_TYPE``, ``CAPACITY`` in that order (no
  ``VEHICLES`` line) and a one-token name; HGS-CVRP skips the first three lines
  and throws on any header key other than ``DIMENSION``, ``EDGE_WEIGHT_TYPE``,
  ``CAPACITY``, ``DISTANCE`` and ``SERVICE_TIME``.
- ``lkh``: LKH-3 refuses ``NODE_COORD_SECTION`` next to explicit weights (it
  needs a coordinate type), reads ``VEHICLES`` and the VRPTW sections, and
  treats ``ACVRP`` and ``CVRPTW`` as asymmetric (``CVRP`` as symmetric).
- ``cvrplib-explicit``: a generic CVRPLIB file with an integer ``FULL_MATRIX``
  (for user-configured solvers), rendered by ``cvrplib.render_cvrplib``.

Node ``i + 1`` of a file is library node ``i``; the depot is node 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "render_explicit_cvrplib",
    "render_lkh_params",
    "render_lkh_problem",
    "render_x_format",
    "write_text",
]


def write_text(path: Path, lines: Iterable[str]) -> None:
    """Write lines with a final newline, streaming (an n = 4000 matrix is ~130 MB of text)."""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _fmt(value: float, decimals: int) -> str:
    text = f"{float(value):.{decimals}f}"
    return text


def render_x_format(
    *,
    coordinates: Sequence[Sequence[float]],
    demands: Sequence[int],
    capacity: int,
    coordinate_decimals: int,
    comment: str = "MAMUT-routing",
    name: str = "problem",
) -> Iterable[str]:
    """Lines of an X-format CVRP (EUC_2D) with the depot as node 1."""
    if " " in name:
        raise ValueError("the X-format name must be a single token")
    yield f"NAME : {name}"
    yield f"COMMENT : {comment.replace(chr(10), ' ')}"
    yield "TYPE : CVRP"
    yield f"DIMENSION : {len(coordinates)}"
    yield "EDGE_WEIGHT_TYPE : EUC_2D"
    yield f"CAPACITY : {int(capacity)}"
    yield "NODE_COORD_SECTION"
    for index, (x, y) in enumerate(coordinates):
        yield f"{index + 1} {_fmt(x, coordinate_decimals)} {_fmt(y, coordinate_decimals)}"
    yield "DEMAND_SECTION"
    for index, demand in enumerate(demands):
        yield f"{index + 1} {int(demand)}"
    yield "DEPOT_SECTION"
    yield "1"
    yield "-1"
    yield "EOF"


def render_lkh_problem(
    *,
    problem_type: str,
    matrix: Sequence[Sequence[int]],
    demands: Sequence[int],
    capacity: int,
    vehicles: int,
    time_windows: Sequence[Sequence[int]] | None = None,
    service_times: Sequence[int] | None = None,
    name: str = "problem",
) -> Iterable[str]:
    """Lines of an LKH-3 problem file: explicit integer FULL_MATRIX, no coordinates."""
    if problem_type not in ("CVRP", "ACVRP", "CVRPTW"):
        raise ValueError(f"unsupported LKH-3 type {problem_type!r}")
    if (time_windows is None) != (problem_type != "CVRPTW"):
        raise ValueError("CVRPTW needs time windows, and only CVRPTW takes them")
    yield f"NAME : {name}"
    yield f"TYPE : {problem_type}"
    yield f"DIMENSION : {len(matrix)}"
    yield f"VEHICLES : {int(vehicles)}"
    yield f"CAPACITY : {int(capacity)}"
    yield "EDGE_WEIGHT_TYPE : EXPLICIT"
    yield "EDGE_WEIGHT_FORMAT : FULL_MATRIX"
    yield "EDGE_WEIGHT_SECTION"
    for row in matrix:
        yield " ".join(str(int(value)) for value in row)
    yield "DEMAND_SECTION"
    for index, demand in enumerate(demands):
        yield f"{index + 1} {int(demand)}"
    if time_windows is not None:
        yield "SERVICE_TIME_SECTION"
        for index, service in enumerate(service_times or [0] * len(matrix)):
            yield f"{index + 1} {int(service)}"
        yield "TIME_WINDOW_SECTION"
        for index, (ready, due) in enumerate(time_windows):
            yield f"{index + 1} {int(ready)} {int(due)}"
    yield "DEPOT_SECTION"
    yield "1"
    yield "-1"
    yield "EOF"


def render_lkh_params(
    *,
    problem: Path,
    solution: Path,
    seed: int,
    time_limit_s: float,
    extra: Sequence[str] = (),
) -> Iterable[str]:
    """Lines of an LKH-3 parameter file for one bounded VRP run.

    ``SPECIAL`` selects LKH-3's VRP move settings; ``RUNS = 1`` with a huge
    ``MAX_TRIALS`` makes ``TIME_LIMIT`` the stopping rule (in LKH-3.0.6 the limit
    counts from after reading the parameters, preprocessing included);
    ``PRECISION = 1`` keeps scaled costs inside LKH's 32-bit ``C * PRECISION``;
    ``MTSP_MIN_SIZE = 0`` lets vehicles stay unused, so ``VEHICLES`` is an upper
    bound rather than an exact fleet.
    """
    yield "SPECIAL"
    yield f"PROBLEM_FILE = {problem}"
    yield f"MTSP_SOLUTION_FILE = {solution}"
    yield "RUNS = 1"
    yield "MAX_TRIALS = 1000000000"
    yield f"TIME_LIMIT = {float(time_limit_s):.1f}"
    yield f"SEED = {int(seed)}"
    yield "PRECISION = 1"
    yield "MTSP_MIN_SIZE = 0"
    yield "TRACE_LEVEL = 1"
    yield from extra


def render_explicit_cvrplib(
    *,
    matrix: Sequence[Sequence[int]],
    coordinates: Sequence[Sequence[float]],
    demands: Sequence[int],
    capacity: int,
    num_vehicles: int | None,
    time_windows: Sequence[Sequence[int]] | None = None,
    service_times: Sequence[int] | None = None,
    name: str = "problem",
) -> str:
    """A generic CVRPLIB file with an integer FULL_MATRIX (``cvrplib.render_cvrplib``)."""
    from mamut_routing_lib.cvrplib import render_cvrplib

    return render_cvrplib(
        name=name,
        comment="MAMUT-routing (scaled integer costs)",
        coordinates=coordinates,
        demands=demands,
        capacity=capacity,
        depot=0,
        num_vehicles=num_vehicles,
        arc_costs=matrix,
        decimals=None,
        time_windows=time_windows,
        service_times=service_times,
        edge_weight_type="EXPLICIT",
    )

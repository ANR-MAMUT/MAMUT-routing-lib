"""A stand-in solver executable for tests and for checking a solvers config.

    python -m mamut_routing_lib.solvers.external._fake_solver --problem P --solution S [--seed N] [--time-limit T]
    python -m mamut_routing_lib.solvers.external._fake_solver --params problem.par       # LKH-3 style

It reads the same inputs as the real solvers (X-format EUC_2D, explicit
FULL_MATRIX, LKH-3 problem + parameter files), builds a capacity-feasible
solution greedily (customers in index order, one route per customer under time
windows), prints progress lines ``progress cost=<c> it=<i>`` and writes the
solution as CVRPLIB ``.sol`` or, for LKH-3 parameter files, as an MTSP solution
file. Its cost is in solver units: integer sums for explicit matrices, ``nint``
or float Euclidean distances (``FAKE_DISTANCE=nint|float``) for coordinates.

``FAKE_MODE`` selects a misbehaviour: ``ok`` (default), ``sleep`` (never
finishes), ``crash`` (exit 3, no output), ``nonzero-with-solution``,
``no-output``, ``garbage``, ``zero-based`` (customer ids 0..n-1),
``wrong-cost``, ``ignore-sigterm``, ``spawn-grandchild`` (writes the child pid
to ``FAKE_PID_FILE`` and sleeps), ``slow`` (sleeps ``FAKE_SLEEP`` seconds,
printing progress, then answers).
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _read_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip().upper()] = value.strip()
    return values


def _parse_problem(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines()]
    header: dict[str, str] = {}
    section = None
    coords: list[tuple[float, float]] = []
    demands: list[int] = []
    matrix_values: list[float] = []
    windows: list[tuple[float, float]] = []
    services: list[float] = []
    for line in lines:
        if not line or line == "EOF":
            continue
        if line.endswith("_SECTION"):
            section = line
            continue
        if section is None and ":" in line:
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
            continue
        parts = line.split()
        if section == "NODE_COORD_SECTION":
            coords.append((float(parts[1]), float(parts[2])))
        elif section == "DEMAND_SECTION":
            demands.append(int(parts[1]))
        elif section == "EDGE_WEIGHT_SECTION":
            matrix_values.extend(float(part) for part in parts)
        elif section == "TIME_WINDOW_SECTION":
            windows.append((float(parts[1]), float(parts[2])))
        elif section == "SERVICE_TIME_SECTION":
            services.append(float(parts[1]))
    dimension = int(header["DIMENSION"])
    matrix = None
    if matrix_values:
        matrix = [matrix_values[i * dimension : (i + 1) * dimension] for i in range(dimension)]
    return {
        "dimension": dimension,
        "capacity": int(header["CAPACITY"]),
        "vehicles": int(header["VEHICLES"]) if "VEHICLES" in header else None,
        "coords": coords,
        "demands": demands,
        "matrix": matrix,
        "windows": windows or None,
    }


def _distance(problem: dict, i: int, j: int) -> float:
    if problem["matrix"] is not None:
        return problem["matrix"][i][j]
    (xi, yi), (xj, yj) = problem["coords"][i], problem["coords"][j]
    value = math.hypot(xi - xj, yi - yj)
    return float(round(value)) if os.environ.get("FAKE_DISTANCE", "nint") == "nint" else value


def _solve(problem: dict) -> tuple[list[list[int]], float]:
    """Routes in file node ids minus the depot (customers 2..n+1 -> 1..n), and their cost."""
    n = problem["dimension"] - 1
    routes: list[list[int]] = []
    if problem["windows"] is not None:
        routes = [[customer] for customer in range(1, n + 1)]
    else:
        load = 0
        current: list[int] = []
        for customer in range(1, n + 1):
            demand = problem["demands"][customer]
            if current and load + demand > problem["capacity"]:
                routes.append(current)
                current, load = [], 0
            current.append(customer)
            load += demand
        if current:
            routes.append(current)
    cost = 0.0
    for route in routes:
        previous = 0
        for node in route:
            cost += _distance(problem, previous, node)
            previous = node
        cost += _distance(problem, previous, 0)
    return routes, cost


def _write_cvrplib(path: Path, routes: list[list[int]], cost: float, *, zero_based: bool) -> None:
    shift = -1 if zero_based else 0
    lines = [f"Route #{k}: " + " ".join(str(node + shift) for node in route) for k, route in enumerate(routes, 1)]
    lines.append(f"Cost {cost:g}" if not float(cost).is_integer() else f"Cost {int(cost)}")
    path.write_text("\n".join(lines) + "\n")


def _write_lkh(path: Path, routes: list[list[int]], cost: float, vehicles: int) -> None:
    lines = [f"problem, Cost: 0_{int(round(cost))}", f"The tours traveled by the {vehicles} salesmen are:"]
    for route in routes:
        lines.append("1 " + " ".join(str(node + 1) for node in route) + f" 1 (#{len(route)})  Cost: 0")
    for _ in range(max(0, vehicles - len(routes))):
        lines.append("1 1 (#0)  Cost: 0")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem")
    parser.add_argument("--solution")
    parser.add_argument("--params")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=1.0)
    args, _unknown = parser.parse_known_args(argv)
    mode = os.environ.get("FAKE_MODE", "ok")
    lkh = args.params is not None
    if lkh:
        params = _read_params(Path(args.params))
        problem_path = Path(params["PROBLEM_FILE"])
        solution_path = Path(params["MTSP_SOLUTION_FILE"])
    else:
        problem_path = Path(args.problem)
        solution_path = Path(args.solution)
    print(f"fake solver: mode={mode} seed={args.seed}", flush=True)

    if mode == "crash":
        return 3
    if mode == "sleep":
        time.sleep(3600)
        return 0
    if mode == "ignore-sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(3600)
        return 0
    if mode == "spawn-grandchild":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        Path(os.environ["FAKE_PID_FILE"]).write_text(str(child.pid))
        time.sleep(3600)
        return 0

    problem = _parse_problem(problem_path)
    routes, cost = _solve(problem)
    if mode == "slow":
        total = float(os.environ.get("FAKE_SLEEP", "2"))
        step = 0
        while total > 0:
            print(f"progress cost={cost * 2:.3f} it={step}", flush=True)
            time.sleep(min(0.1, total))
            total -= 0.1
            step += 1
    print(f"progress cost={cost * 1.5:.3f} it=1", flush=True)
    print(f"progress cost={cost:.3f} it=2", flush=True)
    if mode == "no-output":
        return 0
    if mode == "garbage":
        solution_path.write_text("this is not a solution\n")
        return 0
    if mode == "wrong-cost":
        cost = cost + 1000
    if lkh:
        _write_lkh(solution_path, routes, cost, problem["vehicles"] or len(routes))
    else:
        _write_cvrplib(solution_path, routes, cost, zero_based=mode == "zero-based")
    return 7 if mode == "nonzero-with-solution" else 0


if __name__ == "__main__":
    sys.exit(main())

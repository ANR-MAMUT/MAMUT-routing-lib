"""Opt-in smoke tests against real solver builds.

Skipped unless ``MAMUT_TEST_SOLVERS_CONFIG`` names a solvers config that points
at real executables (FILO2, HGS-CVRP with the full-matrix reader patch,
AILS-II with the -seed/-out patch, LKH-3); solvers it leaves unconfigured are
skipped one by one::

    MAMUT_TEST_SOLVERS_CONFIG=~/solvers.toml uv run pytest tests/test_external_binaries.py

Each run solves a small generated instance for a few seconds and checks the
solver's routes with the reference checker, the reported cost against the
checker's, and that no solver process outlives a cancelled run.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import pytest

from mamut_routing_lib.models import BenchmarkInstanceCVRP
from mamut_routing_lib.solvers import SolveMonitor, get_solver, run_solver
from mamut_routing_lib.solvers.external.process import live_process_count

CONFIG = os.environ.get("MAMUT_TEST_SOLVERS_CONFIG")
pytestmark = [
    pytest.mark.skipif(not CONFIG, reason="set MAMUT_TEST_SOLVERS_CONFIG to a solvers config with real builds"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups"),
]


def _instance(tmp_path: Path, *, metric: str, n: int = 40, seed: int = 3) -> Path:
    rng = random.Random(seed)
    points = [(round(rng.uniform(0, 5000), 3), round(rng.uniform(0, 5000), 3)) for _ in range(n + 1)]
    costs = [[round(math.hypot(a[0] - b[0], a[1] - b[1]), 3) for b in points] for a in points]
    if metric != "euclidean":  # a road-like asymmetric matrix: detours that depend on the direction
        costs = [[0.0 if i == j else round(value * (1.1 + 0.2 * ((i * 7 + j * 3) % 5) / 4), 3) for j, value in enumerate(row)]
                 for i, row in enumerate(costs)]
    instance = BenchmarkInstanceCVRP(
        instance_name=f"smoke-{metric}",
        instance_origin="OsmCvrpGen",
        benchmark_name="Poryos2026",
        num_customers=n,
        vehicle_capacity=50,
        coordinates=points,
        demands=[0] + [rng.randint(1, 10) for _ in range(n)],
        depot=0,
        arc_costs=costs,
        metadata={"metric_variant": metric},
    )
    path = tmp_path / f"{instance.instance_name}.vrp.json"
    path.write_text(json.dumps(instance.model_dump(mode="json")))
    return path


def _require(name: str):
    solver = get_solver(name, config=CONFIG)
    availability = solver.availability()
    if not availability.available:
        pytest.skip(f"{name}: {availability.reason}")
    return solver


@pytest.mark.parametrize(
    ("name", "metric"),
    [
        ("filo2", "euclidean"),
        ("ails2", "euclidean"),
        ("hgs-cvrp", "euclidean"),
        ("hgs-cvrp", "fastest"),
        ("lkh3", "euclidean"),
        ("lkh3", "fastest"),
    ],
)
def test_real_solver_returns_checker_valid_routes(tmp_path: Path, name: str, metric: str) -> None:
    solver = _require(name)
    solved = run_solver(solver, _instance(tmp_path, metric=metric), time_limit_s=3, seed=1, config=CONFIG)
    result = solved.result
    assert result.status == "feasible", (result.error, result.provenance.get("log_tail"))
    assert solved.valid, solved.check.error_message if solved.check else result.error
    assert result.solver_cost == pytest.approx(solved.cost, rel=1e-5)  # HGS prints 6 significant digits
    if result.provenance.get("reported_cost_matches_checker") is not None:
        assert result.provenance["reported_cost_matches_checker"] is True
    if name == "lkh3" and metric == "fastest":
        assert result.provenance["problem_type_line"] == "ACVRP"


@pytest.mark.parametrize("name", ["hgs-cvrp", "lkh3", "filo2", "ails2"])
def test_cancelling_a_real_solver_kills_it(tmp_path: Path, name: str) -> None:
    solver = _require(name)
    monitor = SolveMonitor(None, interval_s=0.5)
    timer = threading.Timer(2.0, monitor.request_stop)
    timer.start()
    started = time.monotonic()
    solved = run_solver(solver, _instance(tmp_path, metric="euclidean", n=80), time_limit_s=120, monitor=monitor, config=CONFIG)
    timer.cancel()
    assert solved.result.status == "cancelled"
    assert time.monotonic() - started < 15
    assert live_process_count() == 0

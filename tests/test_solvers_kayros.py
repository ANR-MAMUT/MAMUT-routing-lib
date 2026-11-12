"""The KAYROS adapter with a stand-in kayros package: checked results, live incumbents, cancellation.

KAYROS runs in a child Python process (``solvers._kayros_worker``), so the stand-in is a real
package on ``PYTHONPATH`` rather than a module patched into this interpreter.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.solvers import InstanceNotSupportedError, get_solver, run_solver
from mamut_routing_lib.solvers.base import SolveMonitor

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

_FAKE_KAYROS = '''
"""Stand-in for the kayros package (FAKE_KAYROS_MODE: ok | explode | stall)."""
import os
import time
from dataclasses import dataclass


@dataclass
class _Incumbent:
    value: float
    seconds: float
    iteration: int
    origin: str


@dataclass
class _Solution:
    routes: list
    duration: float
    route_durations: list
    status: str
    iterations: int


def solve(instance, params=None, *, time_limit=None, seed=0, on_incumbent=None):
    from mamut_routing_lib.td import LoadedTDInstance

    assert isinstance(instance, LoadedTDInstance)
    mode = os.environ.get("FAKE_KAYROS_MODE", "ok")
    if mode == "explode":
        raise RuntimeError("internal/checker disagreement")
    routes = [[1], [2]]
    on_incumbent(_Incumbent(99.0, 0.01, 1, "ils"), routes)
    if mode == "stall":
        # The real search once it stops improving: no incumbent, so no way back into Python.
        time.sleep(20)
    return _Solution(routes=routes, duration=99.0, route_durations=[50.0, 49.0], status="time_limit", iterations=1)
'''


@pytest.fixture
def fake_kayros(tmp_path: Path, monkeypatch):
    """Put the stand-in package first on the path of this process and of its children."""
    package = tmp_path / "fake-site" / "kayros"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_FAKE_KAYROS, encoding="utf-8")
    site = str(package.parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [site, os.environ.get("PYTHONPATH", "")])))
    monkeypatch.syspath_prepend(site)
    monkeypatch.delitem(sys.modules, "kayros", raising=False)

    def set_mode(mode: str) -> None:
        monkeypatch.setenv("FAKE_KAYROS_MODE", mode)

    return set_mode


@pytest.fixture
def td_instance(tmp_path: Path) -> Path:
    from td_utils import write_toy_instance_files

    return write_toy_instance_files(tmp_path, with_time_windows=True)


def test_kayros_is_listed_and_unavailable_without_the_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kayros", None)
    availability = get_solver("kayros").availability()
    assert not availability.available and "kayros" in availability.hint


def test_kayros_results_are_checked_by_the_td_checker(fake_kayros, td_instance: Path) -> None:
    ticks = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    solved = run_solver("kayros", td_instance, time_limit_s=1, seed=5, monitor=monitor)
    assert solved.objective is ObjectiveFunction.DURATION
    assert solved.result.status == "feasible" and solved.valid
    assert solved.cost == solved.check.routing_cost  # the lib's TD checker, not KAYROS's own number
    assert solved.result.metadata["status"] == "time_limit"
    assert solved.result.metadata["route_durations"] == [50.0, 49.0]
    assert ticks and ticks[-1].best_cost == 99.0
    assert "artifacts_dir" not in solved.result.provenance  # the temporary workdir is removed


def test_a_stop_request_ends_a_search_that_no_longer_improves(fake_kayros, td_instance: Path) -> None:
    """KAYROS only returns to Python on a new incumbent: the stop must not wait for one."""
    fake_kayros("stall")
    reported = []

    def on_tick(tick):
        if tick.best_cost is not None:
            reported.append(tick)
        return len(reported) >= 2  # stop after the only incumbent: no hook will fire again

    started = time.perf_counter()
    solved = run_solver("kayros", td_instance, time_limit_s=300, monitor=SolveMonitor(on_tick, interval_s=0.0))
    assert time.perf_counter() - started < 10  # not the stand-in's 20 s without an incumbent
    assert solved.result.status == "cancelled" and solved.result.stopped_early
    assert solved.result.routes == [[1], [2]] and solved.result.solver_cost == 99.0
    assert solved.valid  # the last incumbent is still checked


def test_kayros_keeps_its_files_in_a_given_workdir(fake_kayros, td_instance: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "kayros-result.json").write_text('{"ok": true, "routes": [[2], [1]], "duration": 1.0}')  # an earlier run's
    fake_kayros("explode")
    solved = run_solver("kayros", td_instance, time_limit_s=1, workdir=workdir, keep_artifacts=True)
    assert solved.result.status == "failed"  # never the stale result
    assert solved.result.provenance["artifacts_dir"] == str(workdir)
    assert "disagreement" in (workdir / "kayros.log").read_text()


def test_kayros_failure_is_a_result(fake_kayros, td_instance: Path) -> None:
    fake_kayros("explode")
    solved = run_solver("kayros", td_instance, time_limit_s=1)
    assert solved.result.status == "failed" and "disagreement" in solved.result.error


def test_kayros_refuses_static_instances_and_fleet_cost(fake_kayros, td_instance: Path, toy_cvrp_instance, tmp_path: Path) -> None:
    import json

    static = tmp_path / "static.vrp.json"
    static.write_text(json.dumps(toy_cvrp_instance.model_dump(mode="json")))
    with pytest.raises(InstanceNotSupportedError, match="CVRP"):
        run_solver("kayros", static, time_limit_s=1)
    with pytest.raises(InstanceNotSupportedError, match="FleetCostDuration"):
        run_solver("kayros", td_instance, time_limit_s=1, objective_function=ObjectiveFunction.FLEET_COST_DURATION)

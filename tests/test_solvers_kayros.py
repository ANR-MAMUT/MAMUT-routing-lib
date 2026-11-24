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
"""Stand-in for the kayros package (FAKE_KAYROS_MODE: ok | explode | stall | killed | killed-early | explode-late)."""
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path


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
    if mode == "killed-early":
        os.kill(os.getpid(), signal.SIGKILL)  # before any incumbent: nothing to recover
    routes = [[1], [2]]
    on_incumbent(_Incumbent(99.0, 0.01, 1, "ils"), routes)
    if mode in ("killed", "explode-late"):
        # The worker saves an incumbent a moment after the hook returns: wait for the file.
        checkpoint = Path(sys.argv[sys.argv.index("--checkpoint") + 1])
        deadline = time.monotonic() + 20
        while not checkpoint.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if mode == "explode-late":
            raise RuntimeError("internal/checker disagreement")
        os.kill(os.getpid(), signal.SIGKILL)  # what the kernel does to a process that runs out of memory
    if mode == "stall":
        # The real search once it stops improving: no incumbent, so no way back into Python.
        time.sleep(20)
    duration = {"missing-duration": None, "infinite-duration": float("inf")}.get(mode, 99.0)
    return _Solution(routes=routes, duration=duration, route_durations=[50.0, 49.0], status="time_limit", iterations=1)
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
    report = solved.bks_metadata()["solver_report"]
    assert report == {
        "objective": "total-route-duration", "encoding": "binary64", "value": 99.0,
        "family_units_per_solver_unit": {"numerator": "1", "denominator": "1"},
    }
    assert isinstance(report["value"], float)
    assert solved.bks_metadata()["solver_cost"] == 99.0
    assert "cost_exact" not in solved.bks_metadata()
    assert solved.result.metadata["status"] == "time_limit"
    assert solved.result.metadata["route_durations"] == [50.0, 49.0]
    assert ticks and ticks[-1].best_cost == 99.0
    assert "artifacts_dir" not in solved.result.provenance  # the temporary workdir is removed


@pytest.mark.parametrize("mode", ["missing-duration", "infinite-duration"])
def test_kayros_with_routes_but_no_finite_duration_keeps_only_the_checker_cost(fake_kayros, td_instance, mode):
    fake_kayros(mode)
    solved = run_solver("kayros", td_instance, time_limit_s=1)
    assert solved.valid and solved.result.routes
    assert solved.result.solver_report is None and solved.result.solver_cost is None
    assert "solver_report" not in solved.bks_metadata() and "solver_cost" not in solved.bks_metadata()


def test_the_worker_runs_the_library_of_the_parent(fake_kayros, td_instance: Path, monkeypatch) -> None:
    """A relative PYTHONPATH must not make the worker, which runs in the work directory, import another copy."""
    import mamut_routing_lib

    package_dir = Path(mamut_routing_lib.__file__).resolve().parent
    monkeypatch.chdir(package_dir.parent)
    site = os.environ["PYTHONPATH"].split(os.pathsep)[0]  # the stand-in kayros package
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([site, "."]))
    solved = run_solver("kayros", td_instance, time_limit_s=1, seed=5)
    assert solved.result.status == "feasible", solved.result.error
    assert solved.result.provenance["worker_lib_is_parent_lib"] is True
    assert solved.result.provenance["worker_lib_version"] == mamut_routing_lib.__version__


@pytest.mark.parametrize(
    ("version", "clauses", "expected"),
    [
        ("0.12.0", ">=0.6.0,<0.12", False),
        ("0.11.3", ">=0.6.0,<0.12", True),
        ("0.13.0.dev0", ">=0.6.0,<0.13", False),
        ("0.13.0.dev0", ">=0.6.0", True),
        ("0.12.0", "~=0.12.0", True),
        ("0.13.0", "~=0.12.0", False),
        ("0.12.0", "==0.12.*", None),
    ],
)
def test_version_clauses(version: str, clauses: str, expected) -> None:
    from mamut_routing_lib.solvers.provenance import satisfies

    assert satisfies(version, clauses) is expected


def test_a_kayros_that_excludes_this_library_is_flagged(fake_kayros, td_instance: Path, monkeypatch) -> None:
    """KAYROS releases cap the library version. Forced together, the mismatch is said and recorded."""
    from mamut_routing_lib.solvers import kayros as kayros_module

    monkeypatch.setattr(kayros_module, "declared_requirement", lambda package, on: ">=0.6.0,<0.1")
    note = get_solver("kayros").availability().note
    assert note is not None and "mamut-routing-lib>=0.6.0,<0.1" in note
    solved = run_solver("kayros", td_instance, time_limit_s=1, seed=5)
    assert solved.valid  # the routes are still priced by this library's checker
    assert solved.result.provenance["kayros_requires_lib"] == ">=0.6.0,<0.1"
    assert solved.result.provenance["kayros_requirement_met"] is False
    monkeypatch.setattr(kayros_module, "declared_requirement", lambda package, on: ">=0.6.0")
    assert get_solver("kayros").availability().note is None


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


def test_a_worker_killed_by_the_system_keeps_its_last_incumbent(fake_kayros, td_instance: Path) -> None:
    """Out of memory, the kernel kills the worker: no result file, and the answer of a long run in the checkpoint."""
    fake_kayros("killed")
    solved = run_solver("kayros", td_instance, time_limit_s=60)
    result = solved.result
    assert result.status == "feasible" and result.stopped_early
    assert result.routes == [[1], [2]] and result.solver_cost == 99.0
    assert solved.valid and solved.to_json()["ok"]  # the last incumbent is checked like any result
    assert "the KAYROS worker was killed by SIGKILL" in result.error and "last incumbent it saved" in result.error
    assert result.provenance["returncode"] == -9 and result.provenance["recovered_from_checkpoint"] is True
    assert result.metadata == {"iterations": 1}
    assert solved.bks_metadata()["solver_report"] == {
        "objective": "total-route-duration", "encoding": "binary64", "value": 99.0,
        "family_units_per_solver_unit": {"numerator": "1", "denominator": "1"},
    }


def test_the_command_line_keeps_the_incumbent_of_a_killed_worker(fake_kayros, td_instance: Path) -> None:
    """A `failed` result is an error row without routes: the recovered incumbent has to be a solved row."""
    from typer.testing import CliRunner

    from mamut_routing_lib.cli import app

    fake_kayros("killed")
    result = CliRunner().invoke(app, ["solve", str(td_instance), "--solver", "kayros", "--time-limit-s", "60"])
    assert result.exit_code == 0, result.output
    assert "feasible 1" in result.stdout and "errors 0" in result.stdout
    assert "the KAYROS worker was killed by SIGKILL" in result.stderr


def test_a_worker_killed_before_any_incumbent_is_a_failure(fake_kayros, td_instance: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "run"
    workdir.mkdir()
    # The checkpoint of an earlier run in a reused work directory is never this run's incumbent.
    (workdir / "kayros-incumbent.json").write_text('{"value": 1.0, "iteration": 7, "routes": [[2], [1]]}')
    fake_kayros("killed-early")
    solved = run_solver("kayros", td_instance, time_limit_s=60, workdir=workdir, keep_artifacts=True)
    assert solved.result.status == "failed" and not solved.result.routes
    assert "KAYROS was killed by SIGKILL" in solved.result.error and "left no result" in solved.result.error
    assert "recovered_from_checkpoint" not in solved.result.provenance


def test_an_error_that_kayros_reports_is_a_failure_whatever_it_saved(fake_kayros, td_instance: Path) -> None:
    """A worker that says its search went wrong is believed: its incumbent is not returned as an answer."""
    fake_kayros("explode-late")
    solved = run_solver("kayros", td_instance, time_limit_s=60)
    assert solved.result.status == "failed" and not solved.result.routes
    assert "disagreement" in solved.result.error
    assert "recovered_from_checkpoint" not in solved.result.provenance


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

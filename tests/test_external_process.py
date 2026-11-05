"""The subprocess runner: the whole process group goes away on timeout, stop request or interrupt."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from mamut_routing_lib.solvers.base import SolveMonitor
from mamut_routing_lib.solvers.external.process import live_process_count, run_process, terminate_all

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX here")

FAKE = [sys.executable, "-m", "mamut_routing_lib.solvers.external._fake_solver"]


def _problem(tmp_path: Path) -> Path:
    path = tmp_path / "problem.vrp"
    path.write_text(
        "NAME : problem\nCOMMENT : t\nTYPE : CVRP\nDIMENSION : 3\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : 10\n"
        "NODE_COORD_SECTION\n1 0 0\n2 3 4\n3 6 8\nDEMAND_SECTION\n1 0\n2 3\n3 4\nDEPOT_SECTION\n1\n-1\nEOF\n"
    )
    return path


def _env(**values: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(values)
    return env


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); check its state.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return False
    return state != "Z"


def test_a_normal_run_exits_and_reports_the_best_progress_line(tmp_path: Path) -> None:
    solution = tmp_path / "solution.sol"
    outcome = run_process(
        [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(solution)],
        cwd=tmp_path,
        log_path=tmp_path / "solver.log",
        time_limit_s=10,
        progress_pattern=r"progress cost=(?P<cost>[0-9.]+) it=(?P<iterations>\d+)",
        poll_s=0.05,
    )
    assert outcome.termination == "exited" and outcome.returncode == 0
    assert outcome.best_cost == pytest.approx(20.0)  # 5 + 5 + 10 with nint distances
    assert solution.read_text().startswith("Route #1: 1 2")
    assert "fake solver" in outcome.log_tail
    assert live_process_count() == 0


def test_timeout_kills_the_whole_group_including_grandchildren(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    started = time.monotonic()
    outcome = run_process(
        [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(tmp_path / "s.sol")],
        cwd=tmp_path,
        log_path=tmp_path / "solver.log",
        time_limit_s=0.5,
        grace_s=0.3,
        term_grace_s=0.3,
        poll_s=0.05,
        env=_env(FAKE_MODE="spawn-grandchild", FAKE_PID_FILE=str(pid_file)),
    )
    assert outcome.termination == "timeout"
    assert time.monotonic() - started < 10
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_a_stop_request_kills_promptly(tmp_path: Path) -> None:
    monitor = SolveMonitor(None, interval_s=0.0)
    timer = threading.Timer(0.3, monitor.request_stop)
    timer.start()
    started = time.monotonic()
    outcome = run_process(
        [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(tmp_path / "s.sol")],
        cwd=tmp_path,
        log_path=tmp_path / "solver.log",
        time_limit_s=60,
        poll_s=0.05,
        term_grace_s=0.3,
        monitor=monitor,
        env=_env(FAKE_MODE="sleep"),
    )
    timer.cancel()
    assert outcome.termination == "cancelled"
    assert time.monotonic() - started < 3
    assert monitor.stopped_early


def test_sigterm_is_escalated_to_sigkill(tmp_path: Path) -> None:
    started = time.monotonic()
    outcome = run_process(
        [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(tmp_path / "s.sol")],
        cwd=tmp_path,
        log_path=tmp_path / "solver.log",
        time_limit_s=0.3,
        grace_s=0.1,
        term_grace_s=0.3,
        poll_s=0.05,
        env=_env(FAKE_MODE="ignore-sigterm"),
    )
    assert outcome.termination == "timeout"
    assert outcome.returncode == -9
    assert time.monotonic() - started < 5


def test_live_progress_reaches_the_monitor(tmp_path: Path) -> None:
    ticks = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    monitor.rescale(1)
    run_process(
        [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(tmp_path / "s.sol")],
        cwd=tmp_path,
        log_path=tmp_path / "solver.log",
        time_limit_s=10,
        poll_s=0.05,
        monitor=monitor,
        progress_pattern=r"progress cost=(?P<cost>[0-9.]+)",
        env=_env(FAKE_MODE="slow", FAKE_SLEEP="0.6"),
    )
    assert any(tick.best_cost == pytest.approx(40.0) for tick in ticks)
    assert all(tick.iterations is None for tick in ticks)


def test_a_missing_executable_is_an_error_outcome(tmp_path: Path) -> None:
    outcome = run_process(
        [str(tmp_path / "no-such-solver")], cwd=tmp_path, log_path=tmp_path / "solver.log", time_limit_s=1
    )
    assert outcome.termination == "error" and "cannot start" in outcome.error


def test_terminate_all_kills_running_solvers(tmp_path: Path) -> None:
    result: dict = {}

    def run() -> None:
        result["outcome"] = run_process(
            [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(tmp_path / "s.sol")],
            cwd=tmp_path,
            log_path=tmp_path / "solver.log",
            time_limit_s=60,
            poll_s=0.05,
            env=_env(FAKE_MODE="sleep"),
        )

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 5
    while live_process_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert terminate_all(term_grace_s=0.3) == 1
    worker.join(timeout=10)
    assert not worker.is_alive()
    # Stopped from another thread (GUI shutdown, Ctrl-C): a cancellation, not a solver failure.
    assert result["outcome"].termination == "cancelled" and result["outcome"].returncode != 0

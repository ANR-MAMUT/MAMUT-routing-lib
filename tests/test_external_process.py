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


def test_progress_groups_that_are_not_numbers_are_skipped(tmp_path: Path) -> None:
    """A user pattern that matches other text: progress is a display, it never ends a solve."""
    import re

    from mamut_routing_lib.solvers.external.process import _scan_progress

    pattern = re.compile(r"^it (?P<iterations>\S+) pen (?P<penalty>\S+) cost (?P<cost>\S+)$", re.MULTILINE)
    log = "it n/a pen 0 cost 12\nit 3 pen none cost 11\nit 4 pen 0 cost eleven\nit 5 pen 1 cost 9\nit 6 pen 0 cost 10\n"
    assert _scan_progress(pattern, log, None, None) == (10.0, 6)  # 12 kept its cost; its iteration text is ignored

    solution = tmp_path / "solution.sol"
    outcome = run_process(
        [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(solution)],
        cwd=tmp_path,
        log_path=tmp_path / "solver.log",
        time_limit_s=10,
        # Matches "fake solver: mode=ok seed=0" too, with words where numbers are expected.
        progress_pattern=r"^(?P<penalty>\w+) (?:solver: mode=\w+ seed|cost)=(?P<cost>[0-9.]+)",
        poll_s=0.05,
    )
    assert outcome.termination == "exited" and outcome.returncode == 0 and solution.is_file()
    assert outcome.best_cost is None


# --------------------------------------------------------------------------- #
# The guard: a solver never outlives a parent that was killed
# --------------------------------------------------------------------------- #

#: A parent of its own: it runs the stand-in solver through ``run_process`` and writes the solver's pid.
_PARENT = """
import os, sys, threading, time
from pathlib import Path
from mamut_routing_lib.solvers.external import process

work, mode = Path(sys.argv[1]), sys.argv[2]

def record_pid():
    while not process._LIVE:
        time.sleep(0.01)
    (work / "solver.pid.tmp").write_text(str(next(iter(process._LIVE))))
    os.replace(work / "solver.pid.tmp", work / "solver.pid")

threading.Thread(target=record_pid, daemon=True).start()
process.run_process(
    [sys.executable, "-m", "mamut_routing_lib.solvers.external._fake_solver",
     "--problem", str(work / "problem.vrp"), "--solution", str(work / "s.sol")],
    cwd=work, log_path=work / "solver.log", time_limit_s=120, term_grace_s=0.3, poll_s=0.05,
    env={**os.environ, "FAKE_MODE": mode, "FAKE_PID_FILE": str(work / "grandchild.pid")},
)
"""


def _wait_until(condition, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


@pytest.mark.parametrize(
    ("signal_name", "mode"),
    [("SIGKILL", "sleep"), ("SIGTERM", "sleep"), ("SIGKILL", "ignore-sigterm"), ("SIGKILL", "spawn-grandchild")],
)
def test_a_killed_parent_leaves_no_solver_behind(tmp_path: Path, signal_name: str, mode: str) -> None:
    """SIGKILL and an unhandled SIGTERM run no exit hook, and the hard timeout dies with the parent."""
    import signal
    import subprocess

    _problem(tmp_path)
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, str(tmp_path), mode])
    pids: list[int] = []
    try:
        log, pid_file, grandchild_file = tmp_path / "solver.log", tmp_path / "solver.pid", tmp_path / "grandchild.pid"
        assert _wait_until(lambda: pid_file.exists() and log.exists() and "fake solver" in log.read_text()), "the solver never started"
        pids.append(int(pid_file.read_text()))
        if mode == "spawn-grandchild":
            assert _wait_until(lambda: grandchild_file.exists() and grandchild_file.read_text().strip()), "no grandchild"
            pids.append(int(grandchild_file.read_text()))
        time.sleep(0.3)  # the stand-in has installed its SIGTERM handler by now
        assert all(_alive(pid) for pid in pids)
        parent.send_signal(getattr(signal, signal_name))
        parent.wait(timeout=10)
        assert _wait_until(lambda: not any(_alive(pid) for pid in pids), timeout_s=10), f"still alive after {signal_name}: {pids}"
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_parent_death_during_guard_startup_never_runs_solver(tmp_path: Path) -> None:
    import signal
    import subprocess

    # Pause the runner exactly between creating the process group and installing its guard.
    parent_code = """
import os, signal, sys
from pathlib import Path
from mamut_routing_lib.solvers.external import process
work = Path(sys.argv[1])
def pause_startup(group, grace):
    (work / 'ready.tmp').write_text(str(group))
    os.replace(work / 'ready.tmp', work / 'ready')
    signal.pause()
process._start_guard = pause_startup
process.run_process(
    [sys.executable, '-c', "from pathlib import Path; import time; Path('started').touch(); time.sleep(5)"],
    cwd=work, log_path=work / 'log', time_limit_s=5,
)
"""
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(tmp_path)])
    group = None
    try:
        ready = tmp_path / "ready"
        assert _wait_until(ready.exists, timeout_s=10)
        group = int(ready.read_text())
        parent.kill()
        parent.wait(timeout=5)
        assert _wait_until(lambda: not _alive(group), timeout_s=3)
        assert not (tmp_path / "started").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if group is not None and _alive(group):
            os.killpg(group, signal.SIGKILL)


def test_guard_startup_failure_is_an_error_before_solver_execution(tmp_path: Path, monkeypatch) -> None:
    from mamut_routing_lib.solvers.external import process

    def unavailable(group, grace):
        raise OSError("guard process unavailable")

    monkeypatch.setattr(process, "_start_guard", unavailable)
    outcome = run_process(
        [sys.executable, "-c", "from pathlib import Path; Path('started').touch()"],
        cwd=tmp_path, log_path=tmp_path / "log", time_limit_s=2,
    )
    assert outcome.termination == "error" and "guard process unavailable" in outcome.error
    assert not (tmp_path / "started").exists() and live_process_count() == 0


def test_a_run_that_ends_releases_its_guard(tmp_path: Path) -> None:
    """The guard of a finished run exits without a signal: nothing is left running, and nothing is killed late."""
    from mamut_routing_lib.solvers.external import process as runner

    started: list = []
    start_guard = runner._start_guard

    def recording_start(group: int, term_grace_s: float):
        guard = start_guard(group, term_grace_s)
        started.append(guard)
        return guard

    runner._start_guard = recording_start
    try:
        outcome = run_process(
            [*FAKE, "--problem", str(_problem(tmp_path)), "--solution", str(tmp_path / "s.sol")],
            cwd=tmp_path,
            log_path=tmp_path / "solver.log",
            time_limit_s=10,
            poll_s=0.05,
        )
    finally:
        runner._start_guard = start_guard
    assert outcome.termination == "exited" and outcome.returncode == 0
    assert len(started) == 1 and started[0] is not None
    assert started[0].returncode == 0  # reaped, and it left by itself
    assert not runner._GUARDS and live_process_count() == 0


def test_termination_signals_become_an_interrupt() -> None:
    import signal

    from mamut_routing_lib.solvers.external.process import TerminationSignal, interrupt_on_termination

    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(TerminationSignal) as caught:
        with interrupt_on_termination():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)  # the handler runs here, in the main thread
    assert isinstance(caught.value, KeyboardInterrupt)
    assert (caught.value.signum, caught.value.name) == (signal.SIGTERM, "SIGTERM")
    assert signal.getsignal(signal.SIGTERM) is before  # restored


def test_termination_signals_are_left_alone_outside_the_main_thread() -> None:
    import signal

    from mamut_routing_lib.solvers.external.process import interrupt_on_termination

    before = signal.getsignal(signal.SIGTERM)
    seen: list = []

    def run() -> None:
        with interrupt_on_termination():
            seen.append(signal.getsignal(signal.SIGTERM))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert seen == [before]

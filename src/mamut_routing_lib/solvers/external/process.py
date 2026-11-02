"""Run one solver process: its own process group, a hard timeout, cooperative cancellation.

- The solver starts in a new session (POSIX) or process group (Windows), so a
  stop request or a timeout kills the whole tree -- a Java launcher, a shell
  wrapper, their children -- not just the immediate child. SIGTERM first, then
  SIGKILL after ``term_grace_s``.
- Output goes to a log file, never a pipe (a chatty solver would block on a
  full pipe). The runner tails the log every ``poll_s`` seconds and parses the
  descriptor's ``progress_pattern`` (named groups ``cost`` and optionally
  ``iterations`` / ``penalty``; a line with a non-zero penalty is not an
  incumbent).
- Every poll calls ``monitor.observe`` (elapsed time, incumbent in solver
  units), which is also where a stop request is noticed.
- Live process groups are registered; :func:`terminate_all` (also run at
  interpreter exit) kills them, so Ctrl-C in a CLI or a GUI shutdown leaves no
  orphan solver behind -- a solver in its own session does not receive the
  terminal's SIGINT.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from mamut_routing_lib.solvers.base import SolveMonitor

__all__ = ["ProcessOutcome", "kill_process_tree", "live_process_count", "run_process", "terminate_all"]

_IS_WINDOWS = sys.platform == "win32"
_LIVE: dict[int, subprocess.Popen] = {}
_LIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int | None
    termination: Literal["exited", "timeout", "cancelled", "error"]
    wall_time: float
    log_path: Path
    log_tail: str
    #: Best incumbent parsed from the progress lines, in solver units.
    best_cost: float | None = None
    iterations: int | None = None
    error: str | None = None


def live_process_count() -> int:
    with _LIVE_LOCK:
        return len(_LIVE)


def _register(process: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE[process.pid] = process


def _unregister(process: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE.pop(process.pid, None)


def kill_process_tree(process: subprocess.Popen, *, term_grace_s: float = 2.0) -> None:
    """Terminate the process and its group: SIGTERM, then SIGKILL after ``term_grace_s``."""
    if _IS_WINDOWS:  # pragma: no cover - no Windows CI; best effort
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            process.wait(timeout=term_grace_s)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + max(0.0, term_grace_s)
    while time.monotonic() < deadline:
        if process.poll() is not None and not _group_alive(process.pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - a zombie we cannot reap
        pass


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def terminate_all(term_grace_s: float = 1.0) -> int:
    """Kill every live solver process group started by this interpreter; returns how many."""
    with _LIVE_LOCK:
        processes = list(_LIVE.values())
    for process in processes:
        kill_process_tree(process, term_grace_s=term_grace_s)
        _unregister(process)
    return len(processes)


atexit.register(terminate_all)


def _read_new(log_path: Path, offset: int) -> tuple[str, int]:
    try:
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return "", offset
    return data.decode("utf-8", errors="replace"), offset + len(data)


def _tail(log_path: Path, limit: int = 4000) -> str:
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    time_limit_s: float,
    grace_s: float = 30.0,
    term_grace_s: float = 2.0,
    poll_s: float = 0.25,
    env: Mapping[str, str] | None = None,
    monitor: SolveMonitor | None = None,
    progress_pattern: str | None = None,
) -> ProcessOutcome:
    """Run ``argv`` until it exits, its budget plus ``grace_s`` runs out, or ``monitor`` asks to stop."""
    pattern = re.compile(progress_pattern, re.MULTILINE) if progress_pattern else None
    started = time.monotonic()
    deadline = started + float(time_limit_s) + max(0.0, float(grace_s))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    popen_kwargs: dict = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "env": dict(env) if env is not None else None,
    }
    if _IS_WINDOWS:  # pragma: no cover
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True
    best: float | None = None
    iterations: int | None = None
    offset = 0
    pending = ""
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(list(argv), stdout=log, **popen_kwargs)
        except OSError as error:
            return ProcessOutcome(
                returncode=None,
                termination="error",
                wall_time=time.monotonic() - started,
                log_path=log_path,
                log_tail="",
                error=f"cannot start {argv[0]}: {error}",
            )
        _register(process)
        termination: Literal["exited", "timeout", "cancelled", "error"] = "exited"
        try:
            while True:
                try:
                    process.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if pattern is not None:
                    chunk, offset = _read_new(log_path, offset)
                    if chunk:
                        text = pending + chunk
                        lines, _, pending = text.rpartition("\n")
                        for match in pattern.finditer(lines):
                            groups = match.groupdict()
                            penalty = groups.get("penalty")
                            if penalty is not None and float(penalty) != 0:
                                continue
                            try:
                                cost = float(groups["cost"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            if best is None or cost < best:
                                best = cost
                            if groups.get("iterations"):
                                iterations = int(float(groups["iterations"]))
                if monitor is not None and monitor.observe(best, iterations=iterations):
                    termination = "cancelled"
                    kill_process_tree(process, term_grace_s=term_grace_s)
                    break
                if time.monotonic() > deadline:
                    termination = "timeout"
                    kill_process_tree(process, term_grace_s=term_grace_s)
                    break
        except BaseException:
            # KeyboardInterrupt / SystemExit in the caller: never leave the solver running.
            kill_process_tree(process, term_grace_s=term_grace_s)
            raise
        finally:
            if not _IS_WINDOWS:
                # The solver exited, but children it spawned may still hold the group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            _unregister(process)
    if pattern is not None:  # progress lines written right before exit
        chunk, offset = _read_new(log_path, offset)
        for match in pattern.finditer(pending + chunk):
            groups = match.groupdict()
            penalty = groups.get("penalty")
            if penalty is not None and float(penalty) != 0:
                continue
            try:
                cost = float(groups["cost"])
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or cost < best:
                best = cost
    if monitor is not None and termination == "exited":
        # The last progress lines usually land right before exit: report the final incumbent.
        monitor.observe(best, iterations=iterations, force=True)
    return ProcessOutcome(
        returncode=process.returncode,
        termination=termination,
        wall_time=time.monotonic() - started,
        log_path=log_path,
        log_tail=_tail(log_path),
        best_cost=best,
        iterations=iterations,
    )

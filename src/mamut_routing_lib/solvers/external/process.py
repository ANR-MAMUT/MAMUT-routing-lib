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
- A parent that is killed runs none of this: SIGKILL, an unhandled SIGTERM
  (``timeout 60 mamut-routing solve``, a systemd stop) or a crash end it without
  the exit hook, and the hard timeout lives in its poll loop. Each registered
  solver therefore has a *guard* (POSIX): a small process that holds the read
  end of a pipe whose only writer is this process. When this process dies, for
  whatever reason, the guard reads end of file and kills the solver's process
  group. A launcher waits for the guard to be ready before it executes solver code.
  A run that ends normally writes one byte first, and the guard exits
  without killing. A process forked from this one without ``exec`` while a
  solve runs inherits the pipe, and the guard then waits for it too.
- :func:`interrupt_on_termination` turns SIGTERM and SIGHUP into the interrupt
  that Ctrl-C raises, for a command line that reports what it stopped.
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence

from mamut_routing_lib.solvers.base import SolveMonitor

__all__ = [
    "ProcessOutcome",
    "TerminationSignal",
    "interrupt_on_termination",
    "kill_process_tree",
    "live_process_count",
    "run_process",
    "terminate_all",
]

_IS_WINDOWS = sys.platform == "win32"
_LIVE: dict[int, subprocess.Popen] = {}
#: The guard of each live solver, by the solver's pid.
_GUARDS: dict[int, subprocess.Popen] = {}
_LIVE_LOCK = threading.Lock()
#: Pids that ``terminate_all`` stopped: their run is ``cancelled``, not a solver failure.
_TERMINATED: set[int] = set()

_LAUNCHER = Path(__file__).with_name("_process_launcher.py").resolve()


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


def _start_guard(group: int, term_grace_s: float) -> subprocess.Popen:
    """Start the POSIX parent-death guard and wait until it is listening."""
    guard = subprocess.Popen(
        [sys.executable, "-I", "-S", str(_LAUNCHER), "guard", str(group), str(max(0.0, term_grace_s))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=os.sep,
        start_new_session=True,
    )
    try:
        assert guard.stdout is not None
        with guard.stdout:
            if guard.stdout.read(1) != b"+":
                raise OSError("the parent-death guard failed to start")
    except BaseException:
        _release_guard(guard)
        raise
    return guard


def _release_guard(guard: subprocess.Popen) -> None:
    """Tell the guard that the run is over (one byte, then end of file), and reap it."""
    if guard.stdin is not None:
        try:
            guard.stdin.write(b".")
            guard.stdin.close()
        except OSError:  # the guard is already gone
            pass
    try:
        guard.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - a guard that does not read its pipe
        guard.kill()
        guard.wait()


@contextmanager
def _running_solver(argv: Sequence[str], term_grace_s: float, **popen_kwargs) -> Iterator[subprocess.Popen]:
    """Keep the solver supervised from its first instruction until its group is gone."""
    with ExitStack() as sockets:
        command = list(argv)
        if _IS_WINDOWS:  # pragma: no cover
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            if not sys.executable or getattr(sys, "frozen", False):
                raise OSError("a Python interpreter is required for the POSIX parent-death guard")
            control, child = socket.socketpair()
            sockets.enter_context(control)
            sockets.enter_context(child)
            command = [sys.executable, "-I", "-S", str(_LAUNCHER), "launch", str(child.fileno()), *argv]
            popen_kwargs.update(start_new_session=True, pass_fds=(child.fileno(),))

        process = subprocess.Popen(command, **popen_kwargs)
        guard = None
        registered = False
        try:
            if not _IS_WINDOWS:
                child.close()
                guard = _start_guard(process.pid, term_grace_s)
            with _LIVE_LOCK:
                if guard is not None:
                    # EOF before this byte ends the launcher without starting the solver.
                    control.sendall(b"+")
                    _GUARDS[process.pid] = guard
                _LIVE[process.pid] = process
                registered = True
            if guard is not None:
                # Waiting for exec must leave the registry available to other runs and to terminate_all.
                # exec closes the socket on success. On failure the launcher returns the OS error.
                try:
                    error = control.recv(8192)
                except ConnectionResetError:
                    with _LIVE_LOCK:
                        if process.pid not in _TERMINATED:
                            raise
                else:
                    if error:
                        raise OSError(error.decode(errors="replace"))
                control.close()
            yield process
        finally:
            if not _IS_WINDOWS:
                control.close()
                # Also remove children left behind by a solver that exited normally.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            if registered:
                _unregister(process)
            elif guard is not None:
                _release_guard(guard)
            process.wait()


def _unregister(process: subprocess.Popen) -> None:
    """Forget a solver whose process group is gone, and release its guard."""
    with _LIVE_LOCK:
        _LIVE.pop(process.pid, None)
        guard = _GUARDS.pop(process.pid, None)
    if guard is not None:
        _release_guard(guard)


def _was_terminated(process: subprocess.Popen) -> bool:
    with _LIVE_LOCK:
        if process.pid in _TERMINATED:
            _TERMINATED.discard(process.pid)
            return True
    return False


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
        _TERMINATED.update(process.pid for process in processes)
    for process in processes:
        kill_process_tree(process, term_grace_s=term_grace_s)
        _unregister(process)
    return len(processes)


atexit.register(terminate_all)


class TerminationSignal(KeyboardInterrupt):
    """SIGTERM or SIGHUP, raised as the interrupt of Ctrl-C by :func:`interrupt_on_termination`."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = int(signum)

    @property
    def name(self) -> str:
        """``SIGTERM`` or ``SIGHUP``."""
        return signal.Signals(self.signum).name


def _raise_termination(signum: int, frame: object) -> None:
    raise TerminationSignal(signum)


@contextmanager
def interrupt_on_termination() -> Iterator[None]:
    """Inside the block, SIGTERM and SIGHUP raise :class:`TerminationSignal` in the main thread.

    It is a ``KeyboardInterrupt``, so a command line that stops its solvers and reports on Ctrl-C
    does the same for ``timeout 60 <command>``, a systemd stop or a closed terminal, which
    otherwise end the process without a word. The former handlers are restored on exit. Python
    installs signal handlers from the main thread only: anywhere else this does nothing.
    """
    previous: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is None:  # no SIGHUP on Windows
                continue
            try:
                previous[signum] = signal.signal(signum, _raise_termination)
            except (OSError, ValueError):  # pragma: no cover - a signal this platform does not let us handle
                pass
    try:
        yield
    finally:
        for signum, handler in previous.items():
            # ``None`` is a handler that was not installed from Python: it cannot be put back.
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)  # type: ignore[arg-type]


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


def _scan_progress(
    pattern: re.Pattern[str], text: str, best: float | None, iterations: int | None
) -> tuple[float | None, int | None]:
    """Fold the progress lines of ``text`` into the running best cost and iteration count.

    A match whose groups are not numbers (a user ``progress_pattern`` that is too loose) is
    skipped: progress is a display, and reading it must never end a solve.
    """
    for match in pattern.finditer(text):
        groups = match.groupdict()
        try:
            penalty = groups.get("penalty")
            if penalty is not None and float(penalty) != 0:
                continue
            cost = float(groups["cost"])
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or cost < best:
            best = cost
        if groups.get("iterations"):
            try:
                iterations = int(float(groups["iterations"]))
            except (TypeError, ValueError, OverflowError):
                pass
    return best, iterations


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
    best: float | None = None
    iterations: int | None = None
    offset = 0
    pending = ""
    with log_path.open("wb") as log, ExitStack() as children:
        try:
            process = children.enter_context(_running_solver(argv, term_grace_s, stdout=log, **popen_kwargs))
        except OSError as error:
            return ProcessOutcome(
                returncode=None,
                termination="error",
                wall_time=time.monotonic() - started,
                log_path=log_path,
                log_tail="",
                error=f"cannot start {argv[0]}: {error}",
            )
        termination: Literal["exited", "timeout", "cancelled", "error"] = "exited"
        try:
            while True:
                try:
                    process.wait(timeout=poll_s)
                    if _was_terminated(process):
                        termination = "cancelled"  # terminate_all() from another thread (shutdown, Ctrl-C)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if pattern is not None:
                    chunk, offset = _read_new(log_path, offset)
                    if chunk:
                        text = pending + chunk
                        lines, _, pending = text.rpartition("\n")
                        best, iterations = _scan_progress(pattern, lines, best, iterations)
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
            _was_terminated(process)  # forget the pid whichever way the run ended
    if pattern is not None:  # progress lines written right before exit
        chunk, offset = _read_new(log_path, offset)
        best, iterations = _scan_progress(pattern, pending + chunk, best, iterations)
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

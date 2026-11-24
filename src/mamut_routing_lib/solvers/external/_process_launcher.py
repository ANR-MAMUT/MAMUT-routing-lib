"""POSIX solver startup and parent-death supervision. Run with the isolated interpreter."""

import os
import signal
import sys
import time


def guard(group: int, grace: float) -> None:
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, signal.SIG_IGN)
    os.write(1, b"+")
    os.close(1)
    if os.read(0, 1):
        return  # the caller cleaned up the group
    try:
        os.killpg(group, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            os.killpg(group, 0)
            time.sleep(0.05)
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def launch(control: int, argv: list[str]) -> None:
    if os.read(control, 1) != b"+":
        return  # the parent died before installing the guard
    os.set_inheritable(control, False)
    # Python ignores SIGPIPE on startup. Restore the subprocess default for native solvers.
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        os.write(control, str(error).encode(errors="replace")[:8192])
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "guard":
        guard(int(sys.argv[2]), float(sys.argv[3]))
    else:
        launch(int(sys.argv[2]), sys.argv[3:])

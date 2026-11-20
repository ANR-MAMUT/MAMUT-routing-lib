"""One KAYROS solve in a process of its own.

    python -m mamut_routing_lib.solvers._kayros_worker INSTANCE --time-limit T --seed N \
        --checkpoint incumbent.json --result result.json

KAYROS's search is a single native call that only comes back to Python on a new
incumbent, so nothing can stop it between two incumbents from inside the
process. ``KayrosSolver`` runs it here and kills this process to stop it.

- The incumbent hook only keeps the latest incumbent in memory: the search
  blocks on it.
- A writer thread saves that incumbent to ``--checkpoint`` (JSON, replaced
  atomically) several times a second, *then* prints
  ``incumbent cost=<value> iterations=<n>``. The parent reads those lines as
  live progress, so every cost it has seen is already on disk with its routes
  when it kills the process.
- The final solution, or the error that ended the solve, goes to ``--result``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

#: How often the writer thread saves a new incumbent (seconds).
CHECKPOINT_INTERVAL_S = 0.1


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` so that a reader, or a kill, never sees half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mamut_routing_lib.solvers._kayros_worker")
    parser.add_argument("instance")
    parser.add_argument("--time-limit", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    checkpoint_path, result_path = Path(args.checkpoint), Path(args.result)

    lock = threading.Lock()
    latest: dict[str, Any] = {}
    dirty = threading.Event()
    finished = threading.Event()

    def on_incumbent(incumbent: Any, routes: list[list[int]]) -> None:
        with lock:
            latest["value"] = float(incumbent.value)
            latest["iteration"] = int(getattr(incumbent, "iteration", 0) or 0)
            latest["routes"] = [list(route) for route in routes]
        dirty.set()

    def save_incumbent() -> None:
        if not dirty.is_set():
            return
        dirty.clear()
        with lock:
            snapshot = dict(latest)
        write_json_atomic(checkpoint_path, snapshot)
        print(f"incumbent cost={snapshot['value']!r} iterations={snapshot['iteration']}", flush=True)

    def writer() -> None:
        while not finished.wait(CHECKPOINT_INTERVAL_S):
            save_incumbent()
        save_incumbent()

    thread = threading.Thread(target=writer, name="kayros-checkpoint", daemon=True)
    thread.start()
    try:
        import kayros

        from mamut_routing_lib.td import load_td_instance

        loaded = load_td_instance(args.instance)
        print("solving", flush=True)
        solution = kayros.solve(loaded, time_limit=args.time_limit, seed=args.seed, on_incumbent=on_incumbent)
    except Exception as error:  # noqa: BLE001 - reported to the parent, which makes it a failed result
        print(f"error: {type(error).__name__}: {error}", flush=True)
        write_json_atomic(result_path, {"ok": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    finally:
        finished.set()
        thread.join(timeout=10)
    write_json_atomic(
        result_path,
        {
            "ok": True,
            "routes": [list(route) for route in solution.routes],
            "duration": float(solution.duration),
            "status": getattr(solution, "status", None),
            "iterations": getattr(solution, "iterations", None),
            "route_durations": [float(value) for value in (getattr(solution, "route_durations", None) or [])],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Environment facts recorded with every solve.

``machine`` (the CPU model) goes into BKS metadata, which is published; the
hostname is only ever written to local run archives (``include_host=True``).
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = ["cpu_model", "environment", "file_sha256", "lib_version", "package_version"]


@lru_cache(maxsize=1)
def cpu_model() -> str:
    """The CPU model name (``/proc/cpuinfo`` on Linux, ``platform.processor()`` elsewhere)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def lib_version() -> str:
    return package_version("mamut-routing-lib") or "unknown"


def environment(*, include_host: bool = False) -> dict[str, Any]:
    """Machine, platform, Python and library versions (and the hostname on request)."""
    facts: dict[str, Any] = {
        "machine": cpu_model(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "lib_version": lib_version(),
    }
    if include_host:
        facts["host"] = socket.gethostname()
    return facts


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str | None:
    """sha256 of a file's bytes, cached on (path, size, mtime); ``None`` if it does not exist."""
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return None
    key = (str(target.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _SHA_CACHE.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        cached = digest.hexdigest()
        _SHA_CACHE[key] = cached
    return cached

"""Environment facts recorded with every solve.

``machine`` (the CPU model) goes into BKS metadata, which is published; the
hostname is only ever written to local run archives (``include_host=True``).
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
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


_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?:\(([^)]*)\)|([^;]*))")
_CLAUSE = re.compile(r"^(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9]+)*)")


def _release(version: str) -> tuple[int, ...]:
    """The numeric release of a version (``0.13.0.dev0`` -> ``(0, 13, 0)``)."""
    match = re.match(r"\s*v?([0-9]+(?:\.[0-9]+)*)", version)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def declared_requirement(package: str, on: str) -> str | None:
    """The version clauses that the installed ``package`` declares on distribution ``on`` (``None`` when it declares none)."""
    try:
        requirements = metadata.requires(package) or []
    except metadata.PackageNotFoundError:
        return None
    wanted = re.sub(r"[-_.]+", "-", on).lower()
    for requirement in requirements:
        match = _REQUIREMENT.match(requirement)
        if match and re.sub(r"[-_.]+", "-", match.group(1)).lower() == wanted:
            return (match.group(2) or match.group(3) or "").strip() or None
    return None


def satisfies(version: str, clauses: str) -> bool | None:
    """Whether ``version`` meets comma-separated clauses such as ``>=0.6.0,<0.12``.

    Releases are compared on their numeric parts only, so a development or pre-release
    counts as its release. ``None`` when a clause is not a plain comparison.
    """
    current = _release(version)
    if not current:
        return None
    for clause in (part.strip() for part in clauses.split(",")):
        if not clause:
            continue
        match = _CLAUSE.match(clause)
        if not match or clause[match.end():].strip():  # a wildcard, a local version, an unknown operator
            return None
        operator, bound = match.group(1), _release(match.group(2))
        width = max(len(current), len(bound))
        left, right = current + (0,) * (width - len(current)), bound + (0,) * (width - len(bound))
        holds = {
            ">=": left >= right, "<=": left <= right, ">": left > right, "<": left < right,
            "==": left == right, "!=": left != right,
            "~=": left >= right and left[: len(bound) - 1] == bound[:-1],
        }[operator]
        if not holds:
            return False
    return True


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

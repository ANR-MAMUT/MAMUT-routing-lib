"""Provenance facts: file hashes that follow the file, and the identity of a JVM."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from mamut_routing_lib.solvers.provenance import file_sha256, java_release


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_file_sha256_of_a_missing_file_is_none(tmp_path: Path) -> None:
    assert file_sha256(tmp_path / "missing") is None


def test_file_sha256_follows_a_build_installed_with_the_same_size_and_mtime(tmp_path: Path) -> None:
    """`cp -p`, `install -p`, a reproducible build: size and modification time say nothing."""
    target = tmp_path / "solver"
    target.write_bytes(b"build-1")
    before = target.stat()
    assert file_sha256(target) == _sha(b"build-1")

    replacement = tmp_path / "solver.new"  # moved over the target: another inode
    replacement.write_bytes(b"build-2")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, target)
    after = target.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert file_sha256(target) == _sha(b"build-2")

    time.sleep(0.05)  # rewritten in place, modification time put back: the change time moved
    target.write_bytes(b"build-3")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert file_sha256(target) == _sha(b"build-3")


def test_file_sha256_without_the_cache_reads_the_file_and_remembers_nothing(tmp_path: Path) -> None:
    from mamut_routing_lib.solvers import provenance

    target = tmp_path / "problem.vrp"
    target.write_bytes(b"run-1")
    remembered = len(provenance._SHA_CACHE)
    assert file_sha256(target, cache=False) == _sha(b"run-1")
    assert len(provenance._SHA_CACHE) == remembered


def test_java_release_reads_the_jdk_of_the_launcher(tmp_path: Path) -> None:
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    java = jdk / "bin" / "java"
    java.write_text("#!/bin/sh\n")
    assert java_release(java) is None  # no release file
    (jdk / "release").write_text('IMPLEMENTOR="Eclipse Adoptium"\nJAVA_RUNTIME_VERSION="21.0.13+9-LTS"\nJAVA_VERSION="21.0.13"\n')
    assert java_release(java) == "21.0.13+9-LTS (Eclipse Adoptium)"
    link = tmp_path / "java"  # /usr/bin/java is a link to the JDK's launcher
    link.symlink_to(java)
    assert java_release(link) == "21.0.13+9-LTS (Eclipse Adoptium)"
    (jdk / "release").write_text('JAVA_VERSION="17.0.2"\n')
    assert java_release(java) == "17.0.2"

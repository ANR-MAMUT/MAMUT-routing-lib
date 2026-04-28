"""Opt-in real-network smoke test against the MAMUT-routing-dummy v0.0.1 release.

This file is skipped by default. To run it:

    MAMUT_ROUTING_TEST_NETWORK=1 pytest tests/test_remote_network.py -v

It performs a real GitHub API call and downloads ~1.8 MB from the
`ANR-MAMUT/MAMUT-routing-dummy` v0.0.1 release. Once the production artifacts
move to the `MAMUT-routing` repository, the `--repo` override below should be
the only line to change (or removed in favor of the lib default).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mamut_routing_lib.enums import BenchmarkName, ProblemType
from mamut_routing_lib.remote import (
    GitHubReleaseClient,
    GitHubReleaseSource,
    ReleaseArchiveScope,
)


pytestmark = pytest.mark.skipif(
    os.getenv("MAMUT_ROUTING_TEST_NETWORK") != "1",
    reason="real-network smoke; set MAMUT_ROUTING_TEST_NETWORK=1 to enable",
)


DUMMY_REPO = "ANR-MAMUT/MAMUT-routing-dummy"
DUMMY_TAG = "v0.0.1"
EXPECTED_SNAPSHOT_ID = "2026-04-24-621056e"
EXPECTED_ASSET_COUNT = 7
TARGET_FILENAME = "CVRP-Mamut2026-snapshot-2026-04-24-621056e.zip"


def test_real_release_download_cvrp_mamut2026(tmp_path: Path) -> None:
    client = GitHubReleaseClient(
        GitHubReleaseSource(repo_full_name=DUMMY_REPO, token=os.getenv("MAMUT_ROUTING_GITHUB_TOKEN"))
    )
    manifest = client.fetch_manifest(tag=DUMMY_TAG)

    assert manifest.snapshot_id == EXPECTED_SNAPSHOT_ID
    assert manifest.release_tag == DUMMY_TAG
    assert len(manifest.assets) == EXPECTED_ASSET_COUNT

    matching = manifest.select_assets(
        scope=ReleaseArchiveScope.PROBLEM_FAMILY,
        problem_type=ProblemType.CVRP,
        benchmark_name=BenchmarkName.MAMUT_2026,
    )
    assert len(matching) == 1
    asset = matching[0]
    assert asset.filename == TARGET_FILENAME
    assert asset.checksum_sha256 is not None
    assert asset.size_bytes is not None and asset.size_bytes > 0

    extracted_dir = client.download_asset(asset, tmp_path, extract=True)

    assert extracted_dir.is_dir()
    expected_subdir = extracted_dir / "benchmarks" / "CVRP" / "Mamut2026"
    assert expected_subdir.is_dir(), f"Expected directory missing: {expected_subdir}"
    extracted_files = list(expected_subdir.rglob("*"))
    assert any(p.is_file() for p in extracted_files), "Extracted archive contained no files"

    zip_path = tmp_path / TARGET_FILENAME
    assert zip_path.is_file()

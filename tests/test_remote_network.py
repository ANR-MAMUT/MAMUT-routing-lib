"""Opt-in real-network smoke test against the MAMUT-routing public release.

This file is skipped by default. To run it:

    MAMUT_ROUTING_TEST_NETWORK=1 pytest tests/test_remote_network.py -v

It performs a real GitHub API call and downloads ~1.6 MB from the
`ANR-MAMUT/MAMUT-routing` release.
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


RELEASE_REPO = "ANR-MAMUT/MAMUT-routing"
RELEASE_TAG = "snapshot-2026-05-22-28f9199"
EXPECTED_SNAPSHOT_ID = "2026-05-22-28f9199"
EXPECTED_ASSET_COUNT = 5
TARGET_FILENAME = "CVRP-Mamut2026-snapshot-2026-05-22-28f9199.zip"


def test_real_release_download_cvrp_mamut2026(tmp_path: Path) -> None:
    client = GitHubReleaseClient(
        GitHubReleaseSource(repo_full_name=RELEASE_REPO, token=os.getenv("MAMUT_ROUTING_GITHUB_TOKEN"))
    )
    manifest = client.fetch_manifest(tag=RELEASE_TAG)

    assert manifest.snapshot_id == EXPECTED_SNAPSHOT_ID
    assert manifest.release_tag == RELEASE_TAG
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

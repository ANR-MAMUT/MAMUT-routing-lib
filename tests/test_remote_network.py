"""Opt-in real-network smoke test against the MAMUT-routing public release.

This file is skipped by default. To run it:

    MAMUT_ROUTING_TEST_NETWORK=1 pytest tests/test_remote_network.py -v

Use MAMUT_ROUTING_TEST_RELEASE_TAG to pin a published family-first snapshot. By default the smoke test follows the latest release and downloads the Sintef2008 collection.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mamut_routing_lib.enums import BenchmarkName, ProblemType
from mamut_routing_lib.artifacts import discover_benchmark_instances
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

def test_real_release_download_sintef2008(tmp_path: Path) -> None:
    client = GitHubReleaseClient(
        GitHubReleaseSource(repo_full_name=RELEASE_REPO, token=os.getenv("MAMUT_ROUTING_GITHUB_TOKEN"))
    )
    tag = os.getenv("MAMUT_ROUTING_TEST_RELEASE_TAG")
    manifest = client.fetch_manifest(tag=tag)
    if tag:
        assert manifest.release_tag == tag
    matching = manifest.select_assets(
        scope=ReleaseArchiveScope.FAMILY_COLLECTION,
        problem_type=ProblemType.VRPTW,
        benchmark_name=BenchmarkName.SINTEF_2008,
    )
    assert len(matching) == 1
    asset = matching[0]
    assert asset.checksum_sha256 and asset.size_bytes and asset.size_bytes > 0
    extracted = client.download_asset(asset, tmp_path, extract=True, manifest=manifest)
    assert extracted == tmp_path / "Sintef2008"
    assert (extracted / "mamut-collection.json").is_file()
    assert discover_benchmark_instances(tmp_path, benchmark_names=[BenchmarkName.SINTEF_2008])
    assert (tmp_path / asset.filename).stat().st_size == asset.size_bytes

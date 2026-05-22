from __future__ import annotations

from pathlib import Path

from mamut_routing_lib.enums import BenchmarkName, ProblemType
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.remote import ReleaseArchiveManifest, ReleaseArchiveScope, load_release_manifest


def make_manifest_payload() -> dict:
    return {
        "schema_version": "1.0.0",
        "snapshot_id": "2026-04-24-deadbee",
        "published_at": "2026-04-24T12:00:00+00:00",
        "source_commit": "deadbeef1234",
        "source_branch": "main",
        "release_tag": "snapshot-2026-04-24",
        "assets": [
            {
                "scope": "problem_family",
                "filename": "CVRP-Mamut2026-snapshot-2026-04-24-deadbee.zip",
                "download_url": "https://example.invalid/CVRP-Mamut2026-snapshot-2026-04-24-deadbee.zip",
                "problem_type": "CVRP",
                "benchmark_name": "Mamut2026",
                "checksum_sha256": "abc123",
                "size_bytes": 123,
                "archive_root": "benchmarks/CVRP/Mamut2026",
            },
            {
                "scope": "problem_family",
                "filename": "VRPTW-Sintef2008-snapshot-2026-04-24-deadbee.zip",
                "download_url": "https://example.invalid/VRPTW-Sintef2008-snapshot-2026-04-24-deadbee.zip",
                "problem_type": "VRPTW",
                "benchmark_name": "Sintef2008",
                "checksum_sha256": "def456",
                "size_bytes": 456,
                "archive_root": "benchmarks/VRPTW/Sintef2008",
            },
        ],
    }


def test_load_release_manifest_from_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "snapshot-manifest.json"
    save_json_to_file(make_manifest_payload(), manifest_path)

    manifest = load_release_manifest(manifest_path)

    assert manifest.snapshot_id == "2026-04-24-deadbee"
    assert len(manifest.assets) == 2


def test_select_assets_filters_by_scope_and_family() -> None:
    manifest = ReleaseArchiveManifest(**make_manifest_payload())

    cvrp_family_assets = manifest.select_assets(
        scope=ReleaseArchiveScope.PROBLEM_FAMILY,
        problem_type=ProblemType.CVRP,
        benchmark_name=BenchmarkName.MAMUT_2026,
    )
    vrptw_family_assets = manifest.select_assets(
        scope=ReleaseArchiveScope.PROBLEM_FAMILY,
        problem_type=ProblemType.VRPTW,
        benchmark_name=BenchmarkName.SINTEF_2008,
    )

    assert [asset.filename for asset in cvrp_family_assets] == [
        "CVRP-Mamut2026-snapshot-2026-04-24-deadbee.zip"
    ]
    assert [asset.filename for asset in vrptw_family_assets] == [
        "VRPTW-Sintef2008-snapshot-2026-04-24-deadbee.zip"
    ]

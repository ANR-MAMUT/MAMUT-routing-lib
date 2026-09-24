from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.cli import app
from mamut_routing_lib.remote import ReleaseArchiveManifest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def manifest_payload() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def manifest(manifest_payload: dict[str, Any]) -> ReleaseArchiveManifest:
    return ReleaseArchiveManifest(**manifest_payload)


def _runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_cli_version_exits_before_remote_setup(flag: str) -> None:
    from importlib import metadata

    with patch("mamut_routing_lib.cli.GitHubReleaseClient") as mock_client_cls:
        result = _runner().invoke(app, [flag])

    expected_version = metadata.version("mamut-routing-lib")
    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"mamut-routing-lib {expected_version}"
    mock_client_cls.assert_not_called()


def test_cli_remote_list_renders_table_and_filters(manifest: ReleaseArchiveManifest) -> None:
    with patch("mamut_routing_lib.cli.GitHubReleaseClient") as mock_client_cls:
        mock_client_cls.return_value.fetch_manifest.return_value = manifest
        result = _runner().invoke(
            app,
            ["remote", "--repo", "acme/repo", "--tag", "v0.0.1", "list", "--problem-type", "CVRP"],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Snapshot 2026-04-24-deadbee" in result.stdout
    assert "CVRP-Poryos2026-snapshot-2026-04-24-deadbee.zip" in result.stdout
    assert "VRPTW-Sintef2008-snapshot-2026-04-24-deadbee.zip" not in result.stdout


def test_cli_remote_list_shows_all_when_no_filter(manifest: ReleaseArchiveManifest) -> None:
    with patch("mamut_routing_lib.cli.GitHubReleaseClient") as mock_client_cls:
        mock_client_cls.return_value.fetch_manifest.return_value = manifest
        result = _runner().invoke(app, ["remote", "--repo", "acme/repo", "list"])

    assert result.exit_code == 0, result.stdout + result.stderr
    for asset in manifest.assets:
        assert asset.filename in result.stdout


def test_cli_fetch_calls_download_asset_for_each_selected(tmp_path: Path, manifest: ReleaseArchiveManifest) -> None:
    fake_client = MagicMock()
    fake_client.fetch_manifest.return_value = manifest
    fake_client.download_asset.side_effect = lambda asset, dest, **_: Path(dest) / asset.filename

    with patch("mamut_routing_lib.cli.GitHubReleaseClient", return_value=fake_client):
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "remote",
                "--repo", "acme/repo",
                "fetch",
                "--problem-type", "CVRP",
                "--no-extract",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert fake_client.download_asset.call_count == 1
    called_asset = fake_client.download_asset.call_args.args[0]
    assert called_asset.filename == "CVRP-Poryos2026-snapshot-2026-04-24-deadbee.zip"
    assert fake_client.download_asset.call_args.kwargs["extract"] is False


def test_cli_fetch_by_filename(tmp_path: Path, manifest: ReleaseArchiveManifest) -> None:
    fake_client = MagicMock()
    fake_client.fetch_manifest.return_value = manifest
    fake_client.download_asset.side_effect = lambda asset, dest, **_: Path(dest) / asset.filename

    with patch("mamut_routing_lib.cli.GitHubReleaseClient", return_value=fake_client):
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "remote",
                "--repo", "acme/repo",
                "fetch",
                "VRPTW-Sintef2008-snapshot-2026-04-24-deadbee.zip",
                "--no-extract",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert fake_client.download_asset.call_count == 1
    assert fake_client.download_asset.call_args.args[0].filename == "VRPTW-Sintef2008-snapshot-2026-04-24-deadbee.zip"


def test_cli_fetch_unknown_filename_errors(tmp_path: Path, manifest: ReleaseArchiveManifest) -> None:
    fake_client = MagicMock()
    fake_client.fetch_manifest.return_value = manifest

    with patch("mamut_routing_lib.cli.GitHubReleaseClient", return_value=fake_client):
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "remote",
                "--repo", "acme/repo",
                "fetch",
                "does-not-exist.zip",
                "--no-extract",
            ],
        )

    assert result.exit_code != 0
    assert "Unknown asset filename" in (result.stderr + result.stdout)
    fake_client.download_asset.assert_not_called()


def test_cli_fetch_no_selection_errors(tmp_path: Path, manifest: ReleaseArchiveManifest) -> None:
    fake_client = MagicMock()
    fake_client.fetch_manifest.return_value = manifest

    with patch("mamut_routing_lib.cli.GitHubReleaseClient", return_value=fake_client):
        result = _runner().invoke(
            app,
            ["--benchmarks-dir", str(tmp_path), "remote", "--repo", "acme/repo", "fetch"],
        )

    assert result.exit_code == 2
    fake_client.download_asset.assert_not_called()


def _build_zip_with(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("inner.txt", content)
    return buffer.getvalue()


def test_cli_verify_classifies_ok_missing_mismatch(tmp_path: Path) -> None:
    ok_bytes = _build_zip_with(b"ok-content")
    mismatch_bytes = _build_zip_with(b"different-content")
    ok_sha = hashlib.sha256(ok_bytes).hexdigest()
    expected_sha_for_mismatch = hashlib.sha256(b"placeholder content for size").hexdigest()

    manifest_dict = {
        "schema_version": "1.0.0",
        "snapshot_id": "test-snap",
        "published_at": "2026-04-24T12:00:00+00:00",
        "source_commit": "deadbeef",
        "release_tag": "v0.0.1",
        "assets": [
            {
                "scope": "problem_family",
                "filename": "ok.zip",
                "download_url": "https://example.invalid/ok.zip",
                "problem_type": "CVRP",
                "benchmark_name": "Poryos2026",
                "checksum_sha256": ok_sha,
                "size_bytes": len(ok_bytes),
                "archive_root": "benchmarks/CVRP/Poryos2026",
            },
            {
                "scope": "problem_family",
                "filename": "mismatch.zip",
                "download_url": "https://example.invalid/mismatch.zip",
                "problem_type": "VRPTW",
                "benchmark_name": "Sintef2008",
                "checksum_sha256": expected_sha_for_mismatch,
                "size_bytes": len(mismatch_bytes),
                "archive_root": "benchmarks/VRPTW/Sintef2008",
            },
            {
                "scope": "problem_family",
                "filename": "missing.zip",
                "download_url": "https://example.invalid/missing.zip",
                "problem_type": "CVRP",
                "benchmark_name": "Poryos2026",
                "checksum_sha256": "0" * 64,
                "size_bytes": 1,
                "archive_root": "benchmarks/CVRP/Poryos2026",
            },
        ],
    }
    manifest = ReleaseArchiveManifest(**manifest_dict)

    (tmp_path / "ok.zip").write_bytes(ok_bytes)
    (tmp_path / "mismatch.zip").write_bytes(mismatch_bytes)

    fake_client = MagicMock()
    fake_client.fetch_manifest.return_value = manifest

    with patch("mamut_routing_lib.cli.GitHubReleaseClient", return_value=fake_client):
        result = _runner().invoke(
            app,
            ["--benchmarks-dir", str(tmp_path), "remote", "--repo", "acme/repo", "verify"],
        )

    assert result.exit_code == 1, result.stdout + result.stderr
    assert "OK        ok.zip" in result.stdout
    assert "MISMATCH  mismatch.zip" in result.stdout
    assert "MISSING   missing.zip" in result.stdout


def test_cli_manifest_outputs_valid_json(manifest: ReleaseArchiveManifest) -> None:
    fake_client = MagicMock()
    fake_client.fetch_manifest.return_value = manifest

    with patch("mamut_routing_lib.cli.GitHubReleaseClient", return_value=fake_client):
        result = _runner().invoke(app, ["remote", "--repo", "acme/repo", "manifest"])

    assert result.exit_code == 0, result.stdout + result.stderr
    parsed = json.loads(result.stdout)
    round_tripped = ReleaseArchiveManifest(**parsed)
    assert round_tripped.snapshot_id == manifest.snapshot_id
    assert len(round_tripped.assets) == len(manifest.assets)


def _local_release(tmp_path: Path) -> tuple[ReleaseArchiveManifest, type]:
    """A one-asset release served from disk by a GitHubReleaseClient subclass (no network)."""
    from mamut_routing_lib.remote import GitHubReleaseClient, GitHubReleaseSource

    source_dir = tmp_path / "release"
    source_dir.mkdir()
    zip_path = source_dir / "CVRP-Poryos2026-snapshot-x.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("benchmarks/CVRP/Poryos2026/n=2/a.vrp.json", b"{}")
    zip_path.write_bytes(buffer.getvalue())
    manifest = ReleaseArchiveManifest(
        snapshot_id="snap-x",
        published_at="2026-09-24T00:00:00Z",
        source_commit="abc",
        release_tag="snapshot-x",
        assets=[
            {
                "scope": "problem_family",
                "filename": zip_path.name,
                "download_url": f"file://{zip_path}",
                "problem_type": "CVRP",
                "benchmark_name": "Poryos2026",
                "checksum_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                "size_bytes": zip_path.stat().st_size,
                "archive_root": "benchmarks/CVRP/Poryos2026",
            }
        ],
    )

    class LocalClient(GitHubReleaseClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(GitHubReleaseSource("acme/repo"))

        def fetch_manifest(self, tag=None):
            return manifest

        def _download_file_once(self, url, destination_path, **_kwargs):
            destination_path.write_bytes(Path(url.removeprefix("file://")).read_bytes())

    return manifest, LocalClient


def test_cli_fetch_extracts_into_the_canonical_tree_and_verify_follows(tmp_path: Path) -> None:
    _, client_cls = _local_release(tmp_path)
    bench = tmp_path / "benchmarks"
    base = ["--benchmarks-dir", str(bench), "remote", "--repo", "acme/repo", "--tag", "snapshot-x"]
    with patch("mamut_routing_lib.cli.GitHubReleaseClient", client_cls):
        fetched = _runner().invoke(app, [*base, "fetch", "--all"])
        assert fetched.exit_code == 0, fetched.stdout + fetched.stderr
        assert (bench / "CVRP" / "Poryos2026" / "n=2" / "a.vrp.json").is_file()
        assert "archive kept at" in fetched.stdout
        assert "OK" in _runner().invoke(app, [*base, "verify"]).stdout
        (bench / "CVRP-Poryos2026-snapshot-x.zip").unlink()
        verified = _runner().invoke(app, [*base, "verify"])
        assert verified.exit_code == 0 and "EXTRACTED" in verified.stdout
        assert _runner().invoke(app, [*base, "fetch", "--all"]).exit_code == 0  # a stamped tree is replaced


def test_cli_fetch_refuses_to_replace_an_unstamped_directory(tmp_path: Path) -> None:
    _, client_cls = _local_release(tmp_path)
    bench = tmp_path / "benchmarks"
    (bench / "CVRP" / "Poryos2026").mkdir(parents=True)
    (bench / "CVRP" / "Poryos2026" / "mine.json").write_text("{}", encoding="utf-8")
    base = ["--benchmarks-dir", str(bench), "remote", "--repo", "acme/repo", "--tag", "snapshot-x", "fetch", "--all"]
    with patch("mamut_routing_lib.cli.GitHubReleaseClient", client_cls):
        refused = _runner().invoke(app, base)
        assert refused.exit_code == 2 and "--force" in refused.stderr
        assert (bench / "CVRP" / "Poryos2026" / "mine.json").is_file()
        forced = _runner().invoke(app, [*base, "--force"])
    assert forced.exit_code == 0, forced.stdout + forced.stderr
    assert not (bench / "CVRP" / "Poryos2026" / "mine.json").exists()

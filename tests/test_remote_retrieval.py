from __future__ import annotations

import io
import json
import http.client
import urllib.error
import zipfile
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest

from mamut_routing_lib.remote import (
    GitHubReleaseClient,
    GitHubReleaseSource,
    ReleaseArchiveAsset,
    ReleaseArchiveManifest,
    ReleaseArchiveScope,
    compute_sha256,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture_manifest_payload() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))


def _build_zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class _FakeHTTPResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(
        self,
        body: bytes,
        *,
        headers: dict[str, str] | None = None,
        final_url: str | None = None,
    ) -> None:
        self._stream = io.BytesIO(body)
        self.headers = headers or {}
        self._final_url = final_url

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            return self._stream.read()
        return self._stream.read(size)

    def geturl(self) -> str | None:
        return self._final_url

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stream.close()


def _patch_urlopen(
    responses_by_url: Mapping[
        str,
        _FakeHTTPResponse | Exception | list[_FakeHTTPResponse | Exception],
    ],
):
    seen_urls: list[str] = []
    seen_headers: list[dict[str, str]] = []

    def _fake_urlopen(request, *_args, **_kwargs):  # noqa: ANN001
        url = request.full_url if hasattr(request, "full_url") else request.get_full_url()
        seen_urls.append(url)
        seen_headers.append(dict(request.header_items()))
        result = responses_by_url[url]
        if isinstance(result, list):
            result = result.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return patch("mamut_routing_lib.remote.urllib.request.urlopen", side_effect=_fake_urlopen), seen_urls, seen_headers


def test_fetch_manifest_with_explicit_tag_uses_direct_release_asset_url() -> None:
    manifest_payload = _load_fixture_manifest_payload()
    expected_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/download/snapshot-2026-05-22/snapshot-manifest.json"
    responses = {expected_url: _FakeHTTPResponse(json.dumps(manifest_payload).encode("utf-8"))}

    patcher, seen_urls, seen_headers = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(
            GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing", token="ghp_secret")
        )
        manifest = client.fetch_manifest(tag="snapshot-2026-05-22")

    assert isinstance(manifest, ReleaseArchiveManifest)
    assert manifest.snapshot_id == manifest_payload["snapshot_id"]
    assert seen_urls == [expected_url]
    case_insensitive_headers = {k.lower(): v for k, v in seen_headers[0].items()}
    assert case_insensitive_headers.get("authorization") == "Bearer ghp_secret"
    assert "mamut-routing-lib" in case_insensitive_headers.get("user-agent", "")


def test_fetch_manifest_resolves_latest_tag_via_redirect() -> None:
    manifest_payload = _load_fixture_manifest_payload()
    latest_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/latest"
    redirected_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/tag/snapshot-2026-05-22"
    manifest_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/download/snapshot-2026-05-22/snapshot-manifest.json"
    responses = {
        latest_url: _FakeHTTPResponse(b"<html/>", final_url=redirected_url),
        manifest_url: _FakeHTTPResponse(json.dumps(manifest_payload).encode("utf-8")),
    }

    patcher, seen_urls, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing"))
        manifest = client.fetch_manifest()

    assert manifest.snapshot_id == manifest_payload["snapshot_id"]
    assert seen_urls == [latest_url, manifest_url]


def test_release_source_rejects_repo_without_owner() -> None:
    with pytest.raises(ValueError, match="owner/name"):
        GitHubReleaseSource(repo_full_name="MAMUT-routing")


def test_fetch_manifest_404_is_runtime_error() -> None:
    expected_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/download/v9.9.9/snapshot-manifest.json"
    http_error = urllib.error.HTTPError(
        url=expected_url, code=404, msg="Not Found", hdrs=None, fp=io.BytesIO(b"")
    )
    responses: dict[str, _FakeHTTPResponse | Exception] = {expected_url: http_error}

    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing"))
        with pytest.raises(RuntimeError, match="HTTP 404"):
            client.fetch_manifest(tag="v9.9.9")


def test_fetch_manifest_retries_transient_disconnect() -> None:
    manifest_payload = _load_fixture_manifest_payload()
    expected_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/download/snapshot-2026-05-22/snapshot-manifest.json"
    responses: dict[str, list[_FakeHTTPResponse | Exception]] = {
        expected_url: [
            http.client.RemoteDisconnected("transient disconnect"),
            _FakeHTTPResponse(json.dumps(manifest_payload).encode("utf-8")),
        ]
    }

    patcher, seen_urls, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(
            GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing"),
            retry_delay_seconds=0,
        )
        manifest = client.fetch_manifest(tag="snapshot-2026-05-22")

    assert manifest.snapshot_id == manifest_payload["snapshot_id"]
    assert seen_urls == [expected_url, expected_url]


def test_fetch_manifest_wraps_repeated_disconnect() -> None:
    expected_url = "https://github.com/ANR-MAMUT/MAMUT-routing/releases/download/snapshot-2026-05-22/snapshot-manifest.json"
    responses: dict[str, list[Exception]] = {
        expected_url: [
            http.client.RemoteDisconnected("first"),
            http.client.RemoteDisconnected("second"),
        ]
    }

    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(
            GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing"),
            retry_attempts=2,
            retry_delay_seconds=0,
        )
        with pytest.raises(RuntimeError, match="after 2 attempt"):
            client.fetch_manifest(tag="snapshot-2026-05-22")


def _make_archive_asset(*, filename: str, sha256: str, size_bytes: int, download_url: str) -> ReleaseArchiveAsset:
    return ReleaseArchiveAsset(
        scope=ReleaseArchiveScope.PROBLEM_FAMILY,
        filename=filename,
        download_url=download_url,
        problem_type=None,
        benchmark_name=None,
        checksum_sha256=sha256,
        size_bytes=size_bytes,
        archive_root="benchmarks/CVRP/Mamut2026",
    )


def test_download_asset_writes_file_and_verifies_sha256(tmp_path: Path) -> None:
    zip_bytes = _build_zip_bytes({"benchmarks/CVRP/Mamut2026/instance_0001.vrp.json": b"{\"id\": 1}"})
    fake_path = tmp_path / "src" / "asset.zip"
    fake_path.parent.mkdir(parents=True)
    fake_path.write_bytes(zip_bytes)
    expected_sha = compute_sha256(fake_path)

    download_url = "https://example.invalid/asset.zip"
    asset = _make_archive_asset(
        filename="CVRP-Mamut2026-snapshot-2026-04-24-deadbee.zip",
        sha256=expected_sha,
        size_bytes=len(zip_bytes),
        download_url=download_url,
    )

    responses = {download_url: _FakeHTTPResponse(zip_bytes)}
    patcher, _, _ = _patch_urlopen(responses)
    destination_dir = tmp_path / "dest"
    with patcher:
        result = GitHubReleaseClient(GitHubReleaseSource("acme/repo")).download_asset(asset, destination_dir)

    assert result == destination_dir / asset.filename
    assert result.exists()
    assert result.read_bytes() == zip_bytes


def test_download_asset_sha256_mismatch_raises_value_error(tmp_path: Path) -> None:
    zip_bytes = b"not a real zip but content with wrong sha"
    download_url = "https://example.invalid/asset.zip"
    asset = _make_archive_asset(
        filename="bogus.zip",
        sha256="0" * 64,
        size_bytes=len(zip_bytes),
        download_url=download_url,
    )

    responses = {download_url: _FakeHTTPResponse(zip_bytes)}
    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            GitHubReleaseClient(GitHubReleaseSource("acme/repo")).download_asset(asset, tmp_path)


def test_download_asset_extracts_zip_to_named_subdirectory(tmp_path: Path) -> None:
    zip_files = {
        "benchmarks/CVRP/Mamut2026/instance_0001.vrp.json": b"{\"id\": 1}",
        "benchmarks/CVRP/Mamut2026/instance_0002.vrp.json": b"{\"id\": 2}",
    }
    zip_bytes = _build_zip_bytes(zip_files)
    sha256 = __import__("hashlib").sha256(zip_bytes).hexdigest()
    download_url = "https://example.invalid/asset.zip"
    asset = _make_archive_asset(
        filename="CVRP-Mamut2026-snapshot-x.zip",
        sha256=sha256,
        size_bytes=len(zip_bytes),
        download_url=download_url,
    )

    responses = {download_url: _FakeHTTPResponse(zip_bytes)}
    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        result = GitHubReleaseClient(GitHubReleaseSource("acme/repo")).download_asset(
            asset, tmp_path, extract=True
        )

    assert result.is_dir()
    assert result.name == "CVRP-Mamut2026-snapshot-x"
    for path_in_zip in zip_files:
        assert (result / path_in_zip).is_file()


def test_download_asset_re_extract_overwrites_existing_dir(tmp_path: Path) -> None:
    zip_files = {"benchmarks/CVRP/file.txt": b"new content"}
    zip_bytes = _build_zip_bytes(zip_files)
    sha256 = __import__("hashlib").sha256(zip_bytes).hexdigest()
    download_url = "https://example.invalid/asset.zip"
    asset = _make_archive_asset(
        filename="archive.zip",
        sha256=sha256,
        size_bytes=len(zip_bytes),
        download_url=download_url,
    )

    stale_extract_dir = tmp_path / "archive"
    stale_extract_dir.mkdir()
    stale_file = stale_extract_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    responses = {download_url: _FakeHTTPResponse(zip_bytes)}
    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        result = GitHubReleaseClient(GitHubReleaseSource("acme/repo")).download_asset(
            asset, tmp_path, extract=True
        )

    assert result == stale_extract_dir
    assert not stale_file.exists()
    assert (result / "benchmarks/CVRP/file.txt").read_bytes() == b"new content"


def test_progress_callback_receives_increasing_byte_counts(tmp_path: Path) -> None:
    payload = b"x" * (3 * 1024 * 1024 + 7)  # 3 chunks + tail
    sha256 = __import__("hashlib").sha256(payload).hexdigest()
    download_url = "https://example.invalid/asset.bin"
    asset = _make_archive_asset(
        filename="payload.zip",
        sha256=sha256,
        size_bytes=len(payload),
        download_url=download_url,
    )

    progress_events: list[tuple[int, int | None]] = []

    def _on_progress(downloaded: int, total: int | None) -> None:
        progress_events.append((downloaded, total))

    responses = {download_url: _FakeHTTPResponse(payload)}
    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        GitHubReleaseClient(GitHubReleaseSource("acme/repo")).download_asset(
            asset, tmp_path, progress_callback=_on_progress
        )

    assert len(progress_events) >= 2
    assert progress_events[-1] == (len(payload), len(payload))
    assert all(total == len(payload) for _, total in progress_events)
    assert progress_events == sorted(progress_events, key=lambda evt: evt[0])

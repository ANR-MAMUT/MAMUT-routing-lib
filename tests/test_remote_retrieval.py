from __future__ import annotations

import io
import json
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

    def __init__(self, body: bytes, *, headers: dict[str, str] | None = None) -> None:
        self._stream = io.BytesIO(body)
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            return self._stream.read()
        return self._stream.read(size)

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stream.close()


def _make_release_metadata(
    *, manifest_filename: str, manifest_url: str, extra_assets: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    assets = [{"name": manifest_filename, "browser_download_url": manifest_url}]
    if extra_assets:
        assets.extend(extra_assets)
    return {"tag_name": "v0.0.1", "assets": assets}


def _patch_urlopen(responses_by_url: Mapping[str, _FakeHTTPResponse | Exception]):
    seen_urls: list[str] = []
    seen_headers: list[dict[str, str]] = []

    def _fake_urlopen(request, *_args, **_kwargs):  # noqa: ANN001
        url = request.full_url if hasattr(request, "full_url") else request.get_full_url()
        seen_urls.append(url)
        seen_headers.append(dict(request.header_items()))
        result = responses_by_url[url]
        if isinstance(result, Exception):
            raise result
        return result

    return patch("mamut_routing_lib.remote.urllib.request.urlopen", side_effect=_fake_urlopen), seen_urls, seen_headers


def test_fetch_manifest_uses_latest_endpoint_with_bearer_token() -> None:
    manifest_payload = _load_fixture_manifest_payload()
    release_metadata_url = "https://api.github.com/repos/ANR-MAMUT/MAMUT-routing-dummy/releases/latest"
    manifest_url = "https://example.invalid/snapshot-manifest.json"
    responses = {
        release_metadata_url: _FakeHTTPResponse(
            json.dumps(_make_release_metadata(manifest_filename="snapshot-manifest.json", manifest_url=manifest_url)).encode("utf-8")
        ),
        manifest_url: _FakeHTTPResponse(json.dumps(manifest_payload).encode("utf-8")),
    }

    patcher, seen_urls, seen_headers = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(
            GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing-dummy", token="ghp_secret")
        )
        manifest = client.fetch_manifest()

    assert isinstance(manifest, ReleaseArchiveManifest)
    assert manifest.snapshot_id == manifest_payload["snapshot_id"]
    assert seen_urls == [release_metadata_url, manifest_url]
    case_insensitive_first_call = {k.lower(): v for k, v in seen_headers[0].items()}
    assert case_insensitive_first_call.get("authorization") == "Bearer ghp_secret"
    assert "mamut-routing-lib" in case_insensitive_first_call.get("user-agent", "")


def test_fetch_manifest_with_explicit_tag_uses_tags_endpoint() -> None:
    manifest_payload = _load_fixture_manifest_payload()
    tag_url = "https://api.github.com/repos/ANR-MAMUT/MAMUT-routing-dummy/releases/tags/v0.0.1"
    manifest_url = "https://example.invalid/snapshot-manifest.json"
    responses = {
        tag_url: _FakeHTTPResponse(
            json.dumps(_make_release_metadata(manifest_filename="snapshot-manifest.json", manifest_url=manifest_url)).encode("utf-8")
        ),
        manifest_url: _FakeHTTPResponse(json.dumps(manifest_payload).encode("utf-8")),
    }

    patcher, seen_urls, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing-dummy"))
        client.fetch_manifest(tag="v0.0.1")

    assert seen_urls[0] == tag_url


def test_fetch_manifest_missing_manifest_asset_raises_file_not_found() -> None:
    release_metadata_url = "https://api.github.com/repos/ANR-MAMUT/MAMUT-routing-dummy/releases/latest"
    responses = {
        release_metadata_url: _FakeHTTPResponse(
            json.dumps({"assets": [{"name": "something-else.zip", "browser_download_url": "https://example.invalid/x"}]}).encode("utf-8")
        ),
    }

    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(GitHubReleaseSource(repo_full_name="ANR-MAMUT/MAMUT-routing-dummy"))
        with pytest.raises(FileNotFoundError, match="snapshot-manifest.json"):
            client.fetch_manifest()


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


def test_http_error_is_wrapped_as_runtime_error() -> None:
    release_url = "https://api.github.com/repos/acme/repo/releases/latest"
    http_error = urllib.error.HTTPError(
        url=release_url, code=404, msg="Not Found", hdrs=None, fp=io.BytesIO(b"")
    )
    responses: dict[str, _FakeHTTPResponse | Exception] = {release_url: http_error}

    patcher, _, _ = _patch_urlopen(responses)
    with patcher:
        client = GitHubReleaseClient(GitHubReleaseSource("acme/repo"))
        with pytest.raises(RuntimeError, match="HTTP 404"):
            client.fetch_manifest()

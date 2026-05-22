from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from mamut_routing_lib.enums import BenchmarkName, ProblemType
from mamut_routing_lib.json_utils import load_json_from_file


DEFAULT_RELEASE_REPO_ENV = "MAMUT_ROUTING_RELEASE_REPO"
DEFAULT_GITHUB_TOKEN_ENV = "MAMUT_ROUTING_GITHUB_TOKEN"
DEFAULT_MANIFEST_FILENAME = "snapshot-manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 0.5

ProgressCallback: TypeAlias = Callable[[int, int | None], None]


class ReleaseArchiveScope(str, Enum):
    PROBLEM_FAMILY = "problem_family"


class ReleaseArchiveAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: ReleaseArchiveScope
    filename: str
    download_url: str
    problem_type: ProblemType | None = None
    benchmark_name: BenchmarkName | None = None
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    archive_root: str | None = None


class ReleaseArchiveManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    snapshot_id: str
    published_at: str
    source_commit: str
    source_branch: str | None = None
    release_tag: str | None = None
    assets: list[ReleaseArchiveAsset] = Field(default_factory=list)

    def select_assets(
        self,
        *,
        scope: ReleaseArchiveScope | None = None,
        problem_type: ProblemType | None = None,
        benchmark_name: BenchmarkName | None = None,
    ) -> list[ReleaseArchiveAsset]:
        selected = self.assets
        if scope is not None:
            selected = [asset for asset in selected if asset.scope == scope]
        if problem_type is not None:
            selected = [asset for asset in selected if asset.problem_type == problem_type]
        if benchmark_name is not None:
            selected = [asset for asset in selected if asset.benchmark_name == benchmark_name]
        return selected


@dataclass(frozen=True)
class GitHubReleaseSource:
    repo_full_name: str
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME
    token: str | None = None

    def __post_init__(self) -> None:
        parts = self.repo_full_name.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"repo_full_name must be in 'owner/name' format, got: {self.repo_full_name!r}. "
                f"Example: 'ANR-MAMUT/MAMUT-routing'."
            )

    @classmethod
    def from_env(cls) -> "GitHubReleaseSource":
        repo_full_name = os.getenv(DEFAULT_RELEASE_REPO_ENV, "ANR-MAMUT/MAMUT-routing")
        token = os.getenv(DEFAULT_GITHUB_TOKEN_ENV)
        return cls(repo_full_name=repo_full_name, token=token)


def load_release_manifest(path: str | Path) -> ReleaseArchiveManifest:
    return ReleaseArchiveManifest(**load_json_from_file(path))


def compute_sha256(filepath: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(filepath: str | Path, expected_sha256: str) -> None:
    actual_sha256 = compute_sha256(filepath)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {filepath}: expected {expected_sha256}, got {actual_sha256}")


class GitHubReleaseClient:
    def __init__(
        self,
        source: GitHubReleaseSource | None = None,
        *,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.source = source or GitHubReleaseSource.from_env()
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.request_timeout_seconds = request_timeout_seconds

    def fetch_manifest(self, tag: str | None = None) -> ReleaseArchiveManifest:
        resolved_tag = tag if tag is not None else self._resolve_latest_tag()
        manifest_url = self._build_release_asset_url(resolved_tag, self.source.manifest_filename)
        payload = self._download_json(manifest_url)
        return ReleaseArchiveManifest(**payload)

    def _build_release_asset_url(self, tag: str, asset_filename: str) -> str:
        return f"https://github.com/{self.source.repo_full_name}/releases/download/{tag}/{asset_filename}"

    def _resolve_latest_tag(self) -> str:
        url = f"https://github.com/{self.source.repo_full_name}/releases/latest"
        with self._open_url(url) as response:
            final_url = response.geturl()
        marker = "/releases/tag/"
        if marker not in final_url:
            raise RuntimeError(
                f"Could not resolve latest release tag for {self.source.repo_full_name!r} from {final_url!r}. "
                f"Pass --tag explicitly."
            )
        return final_url.rsplit(marker, 1)[-1].split("?", 1)[0].rstrip("/")

    def download_asset(
        self,
        asset: ReleaseArchiveAsset,
        destination_dir: str | Path,
        *,
        extract: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        destination_root = Path(destination_dir)
        destination_root.mkdir(parents=True, exist_ok=True)
        destination_path = destination_root / asset.filename
        self._download_file(
            asset.download_url,
            destination_path,
            expected_total_bytes=asset.size_bytes,
            progress_callback=progress_callback,
        )

        if asset.checksum_sha256 is not None:
            verify_sha256(destination_path, asset.checksum_sha256)

        if not extract:
            return destination_path

        extract_dir = destination_root / destination_path.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination_path, "r") as archive:
            archive.extractall(extract_dir)
        return extract_dir

    def _download_json(self, url: str) -> dict:
        with self._open_url(url) as response:
            try:
                return json.loads(response.read().decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSON response from {url}: {exc}") from exc

    def _download_file(
        self,
        url: str,
        destination_path: Path,
        *,
        expected_total_bytes: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._download_file_once(
                    url,
                    destination_path,
                    expected_total_bytes=expected_total_bytes,
                    progress_callback=progress_callback,
                )
                return
            except Exception as exc:
                if isinstance(exc, RuntimeError) and not self._is_retryable_runtime_error(exc):
                    raise
                if not self._is_retryable_exception(exc) or attempt >= self.retry_attempts:
                    raise RuntimeError(
                        f"Failed to download {url} after {attempt} attempt(s): {exc}"
                    ) from exc
                if destination_path.exists():
                    destination_path.unlink()
                time.sleep(self.retry_delay_seconds * attempt)

    def _download_file_once(
        self,
        url: str,
        destination_path: Path,
        *,
        expected_total_bytes: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        with self._open_url(url) as response:
            total_bytes = expected_total_bytes
            if total_bytes is None:
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        total_bytes = int(content_length)
                    except ValueError:
                        total_bytes = None

            bytes_downloaded = 0
            with destination_path.open("wb") as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(bytes_downloaded, total_bytes)

    def _open_url(self, url: str):
        request = urllib.request.Request(url, headers=self._build_headers())
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.request_timeout_seconds)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Failed to fetch {url}: HTTP {exc.code}") from exc
            except Exception as exc:
                if not self._is_retryable_exception(exc) or attempt >= self.retry_attempts:
                    raise RuntimeError(
                        f"Failed to fetch {url} after {attempt} attempt(s): {exc}"
                    ) from exc
                time.sleep(self.retry_delay_seconds * attempt)

        raise RuntimeError(f"Failed to fetch {url}")

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                TimeoutError,
                ConnectionError,
                OSError,
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
            ),
        )

    @staticmethod
    def _is_retryable_runtime_error(exc: RuntimeError) -> bool:
        return exc.__cause__ is not None and GitHubReleaseClient._is_retryable_exception(exc.__cause__)

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "mamut-routing-lib",
        }
        if self.source.token:
            headers["Authorization"] = f"Bearer {self.source.token}"
        return headers

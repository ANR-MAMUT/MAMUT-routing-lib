from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mamut_routing_lib.enums import BenchmarkName, ProblemType
from mamut_routing_lib.json_utils import load_json_from_file


DEFAULT_RELEASE_REPO_ENV = "MAMUT_ROUTING_RELEASE_REPO"
DEFAULT_GITHUB_TOKEN_ENV = "MAMUT_ROUTING_GITHUB_TOKEN"
DEFAULT_MANIFEST_FILENAME = "snapshot-manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"


class ReleaseArchiveScope(str, Enum):
    PROBLEM = "problem"
    PROBLEM_FAMILY = "problem_family"
    FAMILY = "family"


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
    def __init__(self, source: GitHubReleaseSource | None = None) -> None:
        self.source = source or GitHubReleaseSource.from_env()

    def fetch_manifest(self, tag: str | None = None) -> ReleaseArchiveManifest:
        release_metadata = self._fetch_release_metadata(tag=tag)
        asset_metadata = self._find_asset_metadata(release_metadata["assets"], self.source.manifest_filename)
        payload = self._download_json(asset_metadata["browser_download_url"])
        return ReleaseArchiveManifest(**payload)

    def download_asset(
        self,
        asset: ReleaseArchiveAsset,
        destination_dir: str | Path,
        *,
        extract: bool = False,
    ) -> Path:
        destination_root = Path(destination_dir)
        destination_root.mkdir(parents=True, exist_ok=True)
        destination_path = destination_root / asset.filename
        self._download_file(asset.download_url, destination_path)

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

    def _fetch_release_metadata(self, tag: str | None = None) -> dict:
        if tag is None:
            url = f"https://api.github.com/repos/{self.source.repo_full_name}/releases/latest"
        else:
            url = f"https://api.github.com/repos/{self.source.repo_full_name}/releases/tags/{tag}"
        return self._download_json(url)

    @staticmethod
    def _find_asset_metadata(assets: list[dict], filename: str) -> dict:
        for asset in assets:
            if asset.get("name") == filename:
                return asset
        raise FileNotFoundError(f"Release asset not found: {filename}")

    def _download_json(self, url: str) -> dict:
        with self._open_url(url) as response:
            return json.loads(response.read().decode("utf-8"))

    def _download_file(self, url: str, destination_path: Path) -> None:
        with self._open_url(url) as response:
            with destination_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)

    def _open_url(self, url: str):
        request = urllib.request.Request(url, headers=self._build_headers())
        try:
            return urllib.request.urlopen(request)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch {url}: HTTP {exc.code}") from exc

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "mamut-routing-lib",
        }
        if self.source.token:
            headers["Authorization"] = f"Bearer {self.source.token}"
        return headers

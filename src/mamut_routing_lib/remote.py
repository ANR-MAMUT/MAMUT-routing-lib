"""Release archives on GitHub: the manifest contract and the download client.

A MAMUT-routing release ships one zip per classic (problem type, family) and
one per family-first collection, plus a ``snapshot-manifest.json``
(``ReleaseArchiveManifest``) listing every asset with its download URL,
sha256, size and ``archive_root``. Archive entries are paths relative to the
repository root (``benchmarks/...``); ``archive_root`` is the subtree an
archive covers (``benchmarks/<ProblemType>/<Family>`` or
``benchmarks/<Family>`` for a collection).

``GitHubReleaseClient`` reads the manifest of a tag (or of the latest
release), downloads assets with retries and verifies their checksum.
Extraction (``extract_release_archive``) lands every archive in the canonical
benchmarks tree: the ``archive_root`` subtree becomes
``<benchmarks-dir>/<archive_root without "benchmarks/">``, so
``discover_benchmark_instances(<benchmarks-dir>)`` works on a fetched tree
exactly as on a repository checkout. An extracted subtree carries a
``.mamut-release.json`` stamp; an existing subtree without one (a git
checkout, hand-made data) is never replaced unless forced. The
``mamut-routing remote`` commands are thin wrappers. Configuration:
``MAMUT_ROUTING_RELEASE_REPO`` (default ``ANR-MAMUT/MAMUT-routing``) and
``MAMUT_ROUTING_GITHUB_TOKEN`` (optional, for rate limits or private
releases).
"""

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
from pathlib import Path, PurePosixPath
from typing import Any, Callable, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from mamut_routing_lib.artifacts import RELEASE_STAGING_DIRNAME
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
#: Stamp written at the root of every subtree extracted by ``extract_release_archive``.
RELEASE_STAMP_FILENAME = ".mamut-release.json"

ProgressCallback: TypeAlias = Callable[[int, int | None], None]


class ReleaseArchiveScope(str, Enum):
    """What an archive covers: one problem type of one classic family, or a whole collection."""
    PROBLEM_FAMILY = "problem_family"
    #: One archive covering a whole family-first collection (all problem-type
    #: variants + the shared sidecars tree), e.g. Poryos2026. ``benchmark_name``
    #: identifies the family; ``problem_type`` is None.
    FAMILY_COLLECTION = "family_collection"


class ReleaseArchiveAsset(BaseModel):
    """One archive of a release: scope, ``filename``, ``download_url``, family identity, ``checksum_sha256``, ``size_bytes``, ``archive_root``."""
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
    """The ``snapshot-manifest.json`` of a release: snapshot identity (id, date, commit, tag) and its ``assets``."""
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
        """Assets matching every given filter (``scope``, ``problem_type``, ``benchmark_name``); no filter returns all.

        A ``problem_type`` filter also keeps the ``family_collection`` assets
        (``problem_type`` None): a collection archive ships every problem type
        of its family, so it may contain the requested one.
        """
        selected = self.assets
        if scope is not None:
            selected = [asset for asset in selected if asset.scope == scope]
        if problem_type is not None:
            selected = [
                asset
                for asset in selected
                if asset.problem_type == problem_type
                or (asset.scope == ReleaseArchiveScope.FAMILY_COLLECTION and asset.problem_type is None)
            ]
        if benchmark_name is not None:
            selected = [asset for asset in selected if asset.benchmark_name == benchmark_name]
        return selected


@dataclass(frozen=True)
class GitHubReleaseSource:
    """Where releases are read from: ``owner/name`` repository, manifest filename, optional API token."""
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
        """Source from ``MAMUT_ROUTING_RELEASE_REPO`` and ``MAMUT_ROUTING_GITHUB_TOKEN``."""
        repo_full_name = os.getenv(DEFAULT_RELEASE_REPO_ENV, "ANR-MAMUT/MAMUT-routing")
        token = os.getenv(DEFAULT_GITHUB_TOKEN_ENV)
        return cls(repo_full_name=repo_full_name, token=token)


def load_release_manifest(path: str | Path) -> ReleaseArchiveManifest:
    """Load and validate a manifest file."""
    return ReleaseArchiveManifest(**load_json_from_file(path))


def compute_sha256(filepath: str | Path) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(filepath: str | Path, expected_sha256: str) -> None:
    """Raise ``ValueError`` unless the file's sha256 equals ``expected_sha256``."""
    actual_sha256 = compute_sha256(filepath)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {filepath}: expected {expected_sha256}, got {actual_sha256}")


class ReleaseExtractionError(RuntimeError):
    """An archive cannot be extracted safely into the benchmarks tree."""


def release_target_relpath(archive_root: str) -> PurePosixPath:
    """Where an ``archive_root`` lands under the benchmarks dir (``benchmarks/`` stripped once)."""
    root = PurePosixPath(archive_root.rstrip("/"))
    if root.is_absolute() or ".." in root.parts or not root.parts:
        raise ReleaseExtractionError(f"invalid archive_root {archive_root!r}")
    parts = root.parts[1:] if root.parts[0] == "benchmarks" else root.parts
    if not parts:
        raise ReleaseExtractionError(f"archive_root {archive_root!r} would replace the whole benchmarks dir")
    return PurePosixPath(*parts)


def _infer_archive_root(names: list[str]) -> str:
    """The family-level root shared by every member (``benchmarks/<PT>/<Family>`` or ``benchmarks/<Family>``)."""
    parents = [PurePosixPath(name).parent.parts for name in names]
    if not parents:
        raise ReleaseExtractionError("empty archive")
    prefix: list[str] = []
    for level in zip(*parents):
        if len(set(level)) != 1:
            break
        prefix.append(level[0])
    if len(prefix) < 2 or prefix[0] != "benchmarks":
        raise ReleaseExtractionError("cannot infer archive_root: members do not share a benchmarks/<...> root")
    problem_types = {item.value for item in ProblemType}
    depth = 3 if prefix[1] in problem_types and len(prefix) >= 3 else 2
    return "/".join(prefix[:depth])


def read_release_stamp(directory: str | Path) -> dict[str, Any] | None:
    """The stamp of an extracted subtree, or None."""
    path = Path(directory) / RELEASE_STAMP_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def extract_release_archive(
    archive_path: str | Path,
    benchmarks_dir: str | Path,
    *,
    archive_root: str | None = None,
    stamp: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Extract a release archive into the canonical benchmarks tree; return the target directory.

    The members under ``archive_root`` (inferred from the members when None)
    land in ``<benchmarks_dir>/<archive_root without "benchmarks/">``. Every
    member is validated first (relative, no ``..``, inside ``archive_root``);
    extraction goes to ``<benchmarks_dir>/.mamut-staging/`` and the result is
    swapped in, so a failure leaves the previous subtree untouched. An
    existing target is replaced only when it carries a release stamp (a
    previous fetch) or with ``force``; ``stamp`` (plus ``archive_root``) is
    written to ``<target>/.mamut-release.json``.
    """
    archive_path = Path(archive_path)
    benchmarks_dir = Path(benchmarks_dir)
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        root = (archive_root or _infer_archive_root([info.filename for info in members])).rstrip("/")
        target = benchmarks_dir / release_target_relpath(root)
        prefix = f"{root}/"
        for info in members:
            name = info.filename
            member = PurePosixPath(name)
            if "\\" in name or member.is_absolute() or ".." in member.parts or not name.startswith(prefix):
                raise ReleaseExtractionError(f"unsafe or out-of-root archive member {name!r} (archive_root {root!r})")
        if target.exists() and not force and read_release_stamp(target) is None:
            raise ReleaseExtractionError(
                f"{target} exists and was not extracted by `remote fetch` (no {RELEASE_STAMP_FILENAME}); "
                "pass --force to replace it"
            )
        staging_parent = benchmarks_dir / RELEASE_STAGING_DIRNAME
        staging = staging_parent / f"{os.getpid()}-{archive_path.stem}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            for info in members:
                destination = staging.joinpath(*PurePosixPath(info.filename).relative_to(root).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
            stamp_payload = dict(stamp or {})
            stamp_payload["archive_root"] = root
            (staging / RELEASE_STAMP_FILENAME).write_text(
                json.dumps(stamp_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            previous = None
            if target.exists():
                previous = target.parent / f".{target.name}.old-{os.getpid()}"
                os.replace(target, previous)
            try:
                try:
                    os.replace(staging, target)
                except OSError:
                    shutil.move(str(staging), str(target))
            except BaseException:
                if previous is not None and not target.exists():
                    os.replace(previous, target)
                raise
            if previous is not None:
                shutil.rmtree(previous)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if staging_parent.is_dir() and not any(staging_parent.iterdir()):
                staging_parent.rmdir()
    return target


class GitHubReleaseClient:
    """Reads manifests and downloads assets of GitHub releases with retries and a request timeout."""
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
        """The manifest of ``tag``, or of the latest release when ``tag`` is ``None``."""
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
        manifest: ReleaseArchiveManifest | None = None,
        force: bool = False,
    ) -> Path:
        """Download ``asset`` into ``destination_dir``, verify its checksum, optionally extract.

        ``destination_dir`` is the benchmarks dir: the archive is kept at
        ``<destination_dir>/<filename>``. Returns the archive path, or with
        ``extract`` the canonical target directory (``extract_release_archive``,
        stamped with the snapshot of ``manifest`` when given; ``force``
        replaces an unstamped existing subtree). ``progress_callback``
        receives ``(bytes_so_far, total_or_None)``.
        """
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

        stamp = {
            "filename": asset.filename,
            "checksum_sha256": asset.checksum_sha256,
            "snapshot_id": manifest.snapshot_id if manifest is not None else None,
            "release_tag": manifest.release_tag if manifest is not None else None,
        }
        return extract_release_archive(
            destination_path,
            destination_root,
            archive_root=asset.archive_root,
            stamp=stamp,
            force=force,
        )

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

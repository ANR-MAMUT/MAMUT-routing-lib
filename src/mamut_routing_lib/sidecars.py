"""Generic sha-pinned sidecar references and benchmark-collection resolution.

A *collection* is a family-first benchmark tree (all problem-type variants of
one benchmark family plus their shared ``sidecars/`` tree) marked by a
``mamut-collection.json`` file at its root. Instance files inside a collection
reference their sidecars with paths **relative to the collection root**, each
paired with the sha256 of the sidecar's uncompressed canonical JSON bytes.
The Mamut2026 family (mounted at ``benchmarks/Mamut2026/``) is the first
collection; historic families keep the problem-type-first layout and
instance-relative sidecar paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

COLLECTION_MARKER_FILENAME = "mamut-collection.json"
COLLECTION_MARKER_FORMAT = "mamut-collection"
COLLECTION_MARKER_FORMAT_VERSION = 1

#: Safety bound for the marker walk-up (a benchmarks tree is never this deep).
_MAX_WALK_UP_LEVELS = 32


class CollectionResolutionError(ValueError):
    """Raised when a collection root or a collection-relative path cannot be resolved."""


def validate_relative_sidecar_path(value: str, field_name: str = "path") -> str:
    """A sidecar path must be relative, non-empty and free of parent escapes."""
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value.startswith("/"):
        raise ValueError(f"{field_name} must be a relative path, got {value!r}")
    if ".." in value.split("/"):
        raise ValueError(f"{field_name} must not contain '..' segments, got {value!r}")
    return value


class SidecarRef(BaseModel):
    """A sha-pinned reference to a sidecar file.

    ``path`` is relative to the collection root (for instances inside a
    collection) or to the instance directory (historic layouts).
    ``sha256`` pins the sidecar's uncompressed canonical JSON bytes, so it is
    stable across ``.json`` and ``.json.gz`` storage forms.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_sidecar_path(value)


class CollectionMarker(BaseModel):
    """Content of the ``mamut-collection.json`` marker at a collection root."""

    model_config = ConfigDict(extra="forbid")

    format: str = COLLECTION_MARKER_FORMAT
    format_version: int = COLLECTION_MARKER_FORMAT_VERSION
    family: str
    layout_version: int = 1

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        if value != COLLECTION_MARKER_FORMAT:
            raise ValueError(f"unexpected collection marker format: {value!r}")
        return value

    @field_validator("format_version")
    @classmethod
    def validate_format_version(cls, value: int) -> int:
        if value != COLLECTION_MARKER_FORMAT_VERSION:
            raise ValueError(f"unsupported collection marker format_version: {value!r}")
        return value


def load_collection_marker(marker_path: str | Path) -> CollectionMarker:
    path = Path(marker_path)
    with path.open("r", encoding="utf-8") as handle:
        return CollectionMarker(**json.load(handle))


def save_collection_marker(marker: CollectionMarker, collection_root: str | Path) -> Path:
    root = Path(collection_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / COLLECTION_MARKER_FILENAME
    payload = marker.model_dump(mode="json")
    target.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return target


def find_collection_root(start: str | Path) -> Path | None:
    """Walk up from ``start`` (a file or directory) to the nearest marker directory."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for _ in range(_MAX_WALK_UP_LEVELS):
        if (current / COLLECTION_MARKER_FILENAME).is_file():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


def require_collection_root(
    instance_path: str | Path,
    collection_root: str | Path | None = None,
) -> Path:
    """The collection root of an instance: explicit override, else marker walk-up."""
    if collection_root is not None:
        root = Path(collection_root)
        if not (root / COLLECTION_MARKER_FILENAME).is_file():
            raise CollectionResolutionError(
                f"explicit collection_root {root} has no {COLLECTION_MARKER_FILENAME}"
            )
        return root
    root = find_collection_root(instance_path)
    if root is None:
        raise CollectionResolutionError(
            f"no {COLLECTION_MARKER_FILENAME} found walking up from {instance_path}; "
            "pass collection_root= explicitly"
        )
    return root


def resolve_sidecar_ref(
    ref: SidecarRef,
    instance_path: str | Path,
    collection_root: str | Path | None = None,
) -> Path:
    """Absolute path of a collection-relative sidecar reference."""
    root = require_collection_root(instance_path, collection_root)
    return root / ref.path

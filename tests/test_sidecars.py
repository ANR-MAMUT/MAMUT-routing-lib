"""Tests for sha-pinned sidecar references and collection-root resolution."""

from __future__ import annotations

import pytest

from mamut_routing_lib.sidecars import (
    COLLECTION_MARKER_FILENAME,
    CollectionMarker,
    CollectionResolutionError,
    SidecarRef,
    find_collection_root,
    load_collection_marker,
    require_collection_root,
    resolve_sidecar_ref,
    save_collection_marker,
)


class TestSidecarRef:
    def test_valid_nested_relative_path(self):
        ref = SidecarRef(path="sidecars/lyon/n=10/base/base.geo.json.gz", sha256=None)
        assert ref.path.startswith("sidecars/")

    @pytest.mark.parametrize("bad", ["", "/abs/x.json", "a/../x.json", ".."])
    def test_invalid_paths_rejected(self, bad):
        with pytest.raises(ValueError):
            SidecarRef(path=bad)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            SidecarRef(path="x.json", surprise=1)


class TestCollectionMarker:
    def test_roundtrip(self, tmp_path):
        marker = CollectionMarker(family="Poryos2026")
        path = save_collection_marker(marker, tmp_path)
        assert path.name == COLLECTION_MARKER_FILENAME
        loaded = load_collection_marker(path)
        assert loaded.family == "Poryos2026"
        assert loaded.layout_version == 1

    def test_bad_format_rejected(self, tmp_path):
        target = tmp_path / COLLECTION_MARKER_FILENAME
        target.write_text('{"format": "something-else", "family": "X"}')
        with pytest.raises(ValueError):
            load_collection_marker(target)

    def test_unknown_version_rejected(self, tmp_path):
        target = tmp_path / COLLECTION_MARKER_FILENAME
        target.write_text('{"format": "mamut-collection", "format_version": 99, "family": "X"}')
        with pytest.raises(ValueError):
            load_collection_marker(target)


class TestRootResolution:
    def make_collection(self, tmp_path):
        root = tmp_path / "Poryos2026"
        save_collection_marker(CollectionMarker(family="Poryos2026"), root)
        deep = root / "TDVRP" / "lyon" / "n=10" / "base" / "sub"
        deep.mkdir(parents=True)
        instance = deep / "base-sub.vrp.json"
        instance.write_text("{}")
        return root, instance

    def test_walk_up_from_file_and_directory(self, tmp_path):
        root, instance = self.make_collection(tmp_path)
        assert find_collection_root(instance) == root
        assert find_collection_root(instance.parent) == root
        assert find_collection_root(root) == root

    def test_no_marker_returns_none(self, tmp_path):
        directory = tmp_path / "plain"
        directory.mkdir()
        assert find_collection_root(directory) is None

    def test_require_with_explicit_override(self, tmp_path):
        root, instance = self.make_collection(tmp_path)
        assert require_collection_root(instance, root) == root

    def test_require_rejects_bad_override(self, tmp_path):
        root, instance = self.make_collection(tmp_path)
        other = tmp_path / "not-a-collection"
        other.mkdir()
        with pytest.raises(CollectionResolutionError, match="has no"):
            require_collection_root(instance, other)

    def test_require_raises_without_marker(self, tmp_path):
        orphan = tmp_path / "orphan.vrp.json"
        orphan.write_text("{}")
        with pytest.raises(CollectionResolutionError, match="pass collection_root"):
            require_collection_root(orphan)

    def test_resolve_sidecar_ref(self, tmp_path):
        root, instance = self.make_collection(tmp_path)
        ref = SidecarRef(path="sidecars/lyon/n=10/base/base.road.json.gz")
        assert resolve_sidecar_ref(ref, instance) == root / ref.path

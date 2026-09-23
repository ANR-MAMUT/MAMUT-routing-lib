"""Discovery over release archives extracted by ``remote fetch``.

``remote fetch`` extracts ``<archive>.zip`` into ``<benchmarks-dir>/<archive stem>/``, and
the archive holds a ``benchmarks/`` tree: ``VRPTW-Sintef2008-snapshot-<id>/benchmarks/
VRPTW/Sintef2008/...`` for a classic family, ``Poryos2026-snapshot-<id>/benchmarks/
Poryos2026/...`` for a collection. Discovery must find the same instances, with the
same ids, as in a checkout, and ignore older snapshots of an archive.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mamut_routing_lib import discover_benchmark_instances, find_release_archive_trees
from mamut_routing_lib.artifacts import benchmark_tree_root, iter_benchmark_files
from mamut_routing_lib.cli import app
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.sidecars import CollectionMarker, save_collection_marker

NEW = "2026-09-23-70ca946"
OLD = "2026-07-06-172eb21"
BASE = "poryos-toyville-n2-hyb"


def _vrptw_payload(name: str, family: str, origin: str) -> dict:
    return {
        "instance_name": name,
        "instance_origin": origin,
        "benchmark_name": family,
        "num_customers": 2,
        "num_vehicles": 2,
        "vehicle_capacity": 10,
        "coordinates": [[0, 0], [1, 1], [2, 2]],
        "demands": [0, 4, 4],
        "depot": 0,
        "arc_costs": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "service_times": [0, 10, 10],
        "time_windows": [[0, 100], [0, 100], [0, 100]],
        "metadata": {},
    }


def _write_checkout(root: Path) -> Path:
    """A checkout's benchmarks/: a classic family, a subset-partitioned one, a collection."""
    save_json_to_file(_vrptw_payload("C101", "Sintef2008", "Solomon1987"), root / "VRPTW" / "Sintef2008" / "n=2" / "C101.vrp.json")
    save_json_to_file(
        _vrptw_payload("ORTEC-toy", "Ortec2022", "Ortec2022"),
        root / "VRPTW" / "Ortec2022" / "final" / "n=2" / "ORTEC-toy.vrp.json",
    )
    collection = root / "Poryos2026"
    save_collection_marker(CollectionMarker(family="Poryos2026"), collection)
    save_json_to_file(
        {
            "instance_name": BASE,
            "instance_origin": "OsmCvrpGen",
            "benchmark_name": "Poryos2026",
            "num_customers": 2,
            "num_vehicles": None,
            "vehicle_capacity": 10,
            "coordinates": [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
            "demands": [0, 4, 4],
            "depot": 0,
            "metric_variant": "euclidean",
            "arc_costs_source": {"model": "euclidean", "decimals": 3},
            "metadata": {"base_instance_name": BASE},
        },
        collection / "CVRP" / "euclidean" / "toyville" / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    return root


def _fetch(checkout: Path, fetched: Path, snapshot: str) -> None:
    """Lay out ``checkout``'s families the way ``remote fetch`` extracts their archives."""
    archives = {
        f"VRPTW-Sintef2008-snapshot-{snapshot}": "VRPTW/Sintef2008",
        f"VRPTW-Ortec2022-snapshot-{snapshot}": "VRPTW/Ortec2022",
        f"Poryos2026-snapshot-{snapshot}": "Poryos2026",
    }
    for stem, tree in archives.items():
        shutil.copytree(checkout / tree, fetched / stem / "benchmarks" / tree)
        (fetched / stem / "LICENSE").write_text("license\n")


@pytest.fixture()
def checkout(tmp_path: Path) -> Path:
    return _write_checkout(tmp_path / "checkout" / "benchmarks")


@pytest.fixture()
def fetched(tmp_path: Path, checkout: Path) -> Path:
    root = tmp_path / "fetched" / "benchmarks"
    _fetch(checkout, root, NEW)
    return root


def _ids(root: Path, **filters) -> list[str]:
    return sorted(item.instance_id for item in discover_benchmark_instances(root, **filters))


def test_fetched_archives_discover_like_a_checkout(checkout: Path, fetched: Path) -> None:
    assert _ids(fetched) == _ids(checkout)
    assert len(_ids(checkout)) == 3


def test_fetched_archives_keep_filters_and_layout_fields(fetched: Path) -> None:
    (ortec,) = discover_benchmark_instances(fetched, benchmark_names=["Ortec2022"])
    assert ortec.subset == "final"
    assert ortec.instance_path.is_relative_to(fetched / f"VRPTW-Ortec2022-snapshot-{NEW}" / "benchmarks")
    (poryos,) = discover_benchmark_instances(fetched, benchmark_names=["Poryos2026"])
    assert poryos.base_instance_name == BASE


def test_latest_snapshot_of_an_archive_wins(checkout: Path, fetched: Path) -> None:
    _fetch(checkout, fetched, OLD)
    items = discover_benchmark_instances(fetched)
    assert sorted(item.instance_id for item in items) == _ids(checkout)
    assert all(NEW in str(item.instance_path) for item in items)
    assert [tree.parent.name for tree in find_release_archive_trees(fetched)] == [
        f"Poryos2026-snapshot-{NEW}",
        f"VRPTW-Ortec2022-snapshot-{NEW}",
        f"VRPTW-Sintef2008-snapshot-{NEW}",
    ]


def test_checkout_and_archives_side_by_side(tmp_path: Path, checkout: Path) -> None:
    root = tmp_path / "mixed"
    shutil.copytree(checkout / "VRPTW" / "Sintef2008", root / "VRPTW" / "Sintef2008")
    _fetch(checkout, root, NEW)
    trees = {tree for tree, _ in iter_benchmark_files(root)}
    assert root in trees and len(trees) == 4
    # Sintef2008 twice: once from the checkout tree, once from its archive.
    assert _ids(root, benchmark_names=["Sintef2008"]) == ["vrptw-sintef2008-n2-C101"] * 2


def test_benchmark_tree_root(checkout: Path, fetched: Path) -> None:
    _fetch(checkout, fetched, OLD)
    new_file = fetched / f"VRPTW-Sintef2008-snapshot-{NEW}" / "benchmarks" / "VRPTW" / "Sintef2008" / "n=2" / "C101.vrp.json"
    old_file = fetched / f"VRPTW-Sintef2008-snapshot-{OLD}" / "benchmarks" / "VRPTW" / "Sintef2008" / "n=2" / "C101.vrp.json"
    assert benchmark_tree_root(new_file, fetched) == fetched / f"VRPTW-Sintef2008-snapshot-{NEW}" / "benchmarks"
    assert benchmark_tree_root(old_file, fetched) is None
    own_file = checkout / "VRPTW" / "Sintef2008" / "n=2" / "C101.vrp.json"
    assert benchmark_tree_root(own_file, checkout) == checkout
    assert benchmark_tree_root(own_file, fetched) is None


def test_cli_lists_fetched_archives_with_layout_ids(checkout: Path, fetched: Path) -> None:
    _fetch(checkout, fetched, OLD)
    result = CliRunner().invoke(app, ["--benchmarks-dir", str(fetched), "list", "--paths-only"])
    assert result.exit_code == 0, result.output
    paths = [line for line in result.output.splitlines() if line.strip()]
    assert len(paths) == 3 and all(NEW in path for path in paths)
    # The subset-partitioned id comes from the path layout, as in a checkout.
    ortec_id = _ids(checkout, benchmark_names=["Ortec2022"])[0]
    result = CliRunner().invoke(app, ["--benchmarks-dir", str(fetched), "list", "--instance-id", ortec_id])
    assert result.exit_code == 0, result.output
    assert ortec_id in result.output

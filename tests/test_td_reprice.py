"""Re-pricing stored TD BKS (``mamut_routing_lib.td.reprice``) and the relaxation-twin helpers."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mamut_routing_lib.cli import app
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.relaxation import check_relaxation_pair, relaxation_twin, structural_relaxation_issues
from mamut_routing_lib.td import (
    TD_CHECKER_CONTRACT,
    load_td_instance,
    materialize_instance_atfs,
)
from mamut_routing_lib.td.igp import load_instance_categories, materialize_selected_atfs_igp
from mamut_routing_lib.td.reprice import (
    iter_td_bks_files,
    load_td_instance_for_routes,
    reprice_td_bks,
    route_arcs,
)
from td_utils import write_toy_instance_files
from test_td_igp import write_igp_instance_files

# TOY1 route 0 -> 1 -> 2 -> 0 prices to 30.0 at departure 40.0 (see test_td_checker).
ROUTES = [[1, 2]]


def _write_bks(directory: Path, *, cost, durations, departures, optimality=None, objective="Duration") -> Path:
    metadata = {
        "authors": "Someone",
        "validated_cost": cost,
        "route_durations": durations,
        "route_departure_times": departures,
    }
    if optimality is not None:
        metadata["optimality"] = optimality
    path = directory / f"TOY1.bks.{objective}.json"
    save_json_to_file(
        {"instance_name": "TOY1", "routes": ROUTES, "cost": cost, "metadata": metadata, "objective_function": objective},
        path,
    )
    return path


def _stamp(proven_optimum):
    return {
        "proven": True,
        "prover": "a test prover",
        "certificate": "Optimal in a test.",
        "date": "2026-01-01",
        "proven_optimum": proven_optimum,
    }


def test_unchanged_bks_is_left_alone(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    path = _write_bks(tmp_path, cost=30.0, durations=[30.0], departures=[40.0])
    before = path.read_bytes()
    result = reprice_td_bks(path)
    assert result.action == "unchanged"
    assert result.changed_fields == []
    assert path.read_bytes() == before


def test_moved_cost_is_rewritten_and_recorded(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    path = _write_bks(tmp_path, cost=30.000000000000004, durations=[30.000000000000004], departures=[40.0])
    dry = reprice_td_bks(path, dry_run=True, note_date="2026-09-24")
    assert dry.action == "would-reprice"
    assert json.loads(path.read_text())["cost"] == 30.000000000000004

    result = reprice_td_bks(path, note_date="2026-09-24")
    assert result.action == "repriced"
    assert result.cost_ulps == 1
    assert set(result.changed_fields) == {"cost", "metadata.validated_cost", "metadata.route_durations"}
    stored = json.loads(path.read_text())
    assert stored["cost"] == 30.0
    assert stored["metadata"]["validated_cost"] == 30.0
    assert stored["metadata"]["route_durations"] == [30.0]
    assert stored["metadata"]["repriced"] == {
        "previous_cost": 30.000000000000004,
        "checker": stored["metadata"]["repriced"]["checker"],
        "contract": TD_CHECKER_CONTRACT,
        "date": "2026-09-24",
    }
    assert reprice_td_bks(path).action == "unchanged"


def test_derived_fields_alone_are_refreshed_without_a_repriced_marker(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    path = _write_bks(tmp_path, cost=30.0, durations=[30.0], departures=[39.999999999999996])
    result = reprice_td_bks(path)
    assert result.action == "repriced"
    assert result.changed_fields == ["metadata.route_departure_times"]
    stored = json.loads(path.read_text())
    assert stored["metadata"]["route_departure_times"] == [40.0]
    assert "repriced" not in stored["metadata"]


def test_stamp_follows_a_dust_move_with_a_note(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    path = _write_bks(
        tmp_path,
        cost=30.000000000000004,
        durations=[30.000000000000004],
        departures=[40.0],
        optimality=_stamp(30.000000000000004),
    )
    assert reprice_td_bks(path, note_date="2026-09-24").action == "repriced"
    optimality = json.loads(path.read_text())["metadata"]["optimality"]
    assert optimality["proven_optimum"] == 30.0
    assert "Re-certification under td-fold/2 is pending" in optimality["note"]


def test_stamp_conflict_is_refused(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    path = _write_bks(tmp_path, cost=31.0, durations=[31.0], departures=[40.0], optimality=_stamp(31.0))
    before = path.read_bytes()
    result = reprice_td_bks(path)
    assert result.action == "stamp-conflict"
    assert path.read_bytes() == before


def test_missing_sidecar_is_reported(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    (tmp_path / "TOY1.atf.json").unlink()
    path = _write_bks(tmp_path, cost=30.0, durations=[30.0], departures=[40.0])
    assert reprice_td_bks(path).action == "skipped-missing-sidecar"


def test_selected_arcs_match_the_full_igp_materialization(tmp_path: Path) -> None:
    instance_path = write_igp_instance_files(tmp_path)
    full = load_td_instance(instance_path)
    routes = [[1, 3], [2]]
    sparse = load_td_instance_for_routes(instance_path, routes)
    assert set(sparse.atfs.arcs) == route_arcs(routes)
    for key, atf in sparse.atfs.arcs.items():
        assert atf == full.atfs.arcs[key]
    categories = load_instance_categories(full.categories_path)
    assert materialize_selected_atfs_igp(full.instance, categories, {(0, 1)})[(0, 1)] == materialize_instance_atfs(
        full.instance, categories
    ).arcs[(0, 1)]


def test_cli_reports_and_exit_status(tmp_path: Path) -> None:
    write_toy_instance_files(tmp_path)
    path = _write_bks(tmp_path, cost=30.000000000000004, durations=[30.0], departures=[40.0])
    report = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(app, ["bks", "reprice-td", str(tmp_path), "--dry-run", "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert "would-reprice 1" in result.output
    assert json.loads(report.read_text())[0]["bks_path"] == str(path)
    (tmp_path / "TOY1.atf.json").unlink()
    assert runner.invoke(app, ["bks", "reprice-td", str(tmp_path)]).exit_code == 1
    assert runner.invoke(app, ["bks", "reprice-td", str(tmp_path), "--allow-missing-sidecars"]).exit_code == 0
    assert iter_td_bks_files([tmp_path]) == [path]


def _write_td_twins(root: Path, strict_cost: float, relaxed_cost: float) -> Path:
    strict_dir = root / "TDVRPTW" / "Dabia2013" / "n=2"
    relaxed_dir = root / "TDVRP" / "Dabia2013" / "n=2"
    strict_dir.mkdir(parents=True)
    relaxed_dir.mkdir(parents=True)
    write_toy_instance_files(strict_dir)
    write_toy_instance_files(relaxed_dir, with_time_windows=False)
    strict_bks = _write_bks(strict_dir, cost=strict_cost, durations=[strict_cost], departures=[40.0])
    _write_bks(relaxed_dir, cost=relaxed_cost, durations=[relaxed_cost], departures=[40.0])
    return strict_bks


def test_relaxation_twin_paths() -> None:
    assert relaxation_twin("b/TDVRPTW/Dabia2013/n=25/C101.vrp.json") == Path("b/TDVRP/Dabia2013/n=25/C101.vrp.json")
    assert relaxation_twin("b/P/VRPTW/fastest/lyon/n=10/base/base-tw-tight.vrp.json") == Path(
        "b/P/CVRP/fastest/lyon/n=10/base/base.vrp.json"
    )
    assert relaxation_twin("b/P/VRPTW/fastest/lyon/n=10/base/base.vrp.json") == Path(
        "b/P/CVRP/fastest/lyon/n=10/base/base.vrp.json"
    )
    assert relaxation_twin("b/VRPTW/Sintef2008/n=100/C101.vrp.json") is None
    assert relaxation_twin("b/TDVRP/Dabia2013/n=25/C101.vrp.json") is None


def test_relaxation_screen_and_pricing(tmp_path: Path) -> None:
    consistent = check_relaxation_pair(_write_td_twins(tmp_path / "a", 30.0, 30.0), priced=True)
    assert consistent is not None and not consistent.inverted and consistent.issues == []
    assert consistent.priced_cost == 30.0
    inverted = check_relaxation_pair(_write_td_twins(tmp_path / "b", 30.0, 31.0))
    assert inverted is not None and inverted.inverted


def test_hierarchical_objective_compares_routes_first(tmp_path: Path) -> None:
    strict_bks = _write_td_twins(tmp_path, 30.0, 30.0)
    relaxed_bks = relaxation_twin(strict_bks)
    for path, routes, cost in ((strict_bks, [[1], [2]], 20.0), (relaxed_bks, [[1, 2]], 25.0)):
        payload = json.loads(path.read_text())
        payload.update(routes=routes, cost=cost, objective_function="HierarchicalVehicleCost")
        save_json_to_file(payload, path.with_name("TOY1.bks.HierarchicalVehicleCost.json"))
    # One route at a higher cost beats two routes: not an inversion.
    check = check_relaxation_pair(strict_bks.with_name("TOY1.bks.HierarchicalVehicleCost.json"))
    assert check is not None and not check.inverted


def test_structural_differences_are_reported(tmp_path: Path) -> None:
    strict_bks = _write_td_twins(tmp_path, 30.0, 30.0)
    relaxed_instance = relaxation_twin(strict_bks.with_name("TOY1.vrp.json"))
    payload = json.loads(relaxed_instance.read_text())
    payload["demands"] = [0, 5, 4]
    save_json_to_file(payload, relaxed_instance)
    assert structural_relaxation_issues(strict_bks.with_name("TOY1.vrp.json"), relaxed_instance) == ["demands differs"]


def test_only_the_depot_window_must_fit_the_horizon(tmp_path: Path) -> None:
    strict_bks = _write_td_twins(tmp_path, 30.0, 30.0)
    strict_instance = strict_bks.with_name("TOY1.vrp.json")
    relaxed_instance = relaxation_twin(strict_instance)
    payload = json.loads(strict_instance.read_text())
    horizon_end = payload["horizon"][1]
    payload["time_windows"][1] = [payload["time_windows"][1][0], horizon_end + 2]
    save_json_to_file(payload, strict_instance)
    assert structural_relaxation_issues(strict_instance, relaxed_instance) == []
    payload["time_windows"][0] = [payload["time_windows"][0][0], horizon_end + 2]
    save_json_to_file(payload, strict_instance)
    assert structural_relaxation_issues(strict_instance, relaxed_instance) == ["the depot window leaves the horizon"]

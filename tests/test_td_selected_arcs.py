"""Route-arc loading (``mamut_routing_lib.td.selected_arcs``) and the relaxation-twin helpers."""

from __future__ import annotations

import json
from pathlib import Path

from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.relaxation import check_relaxation_pair, relaxation_twin, structural_relaxation_issues
from mamut_routing_lib.td import (
    load_td_instance,
    materialize_instance_atfs,
)
from mamut_routing_lib.td.igp import load_instance_categories, materialize_selected_atfs_igp
from mamut_routing_lib.td.selected_arcs import load_td_instance_for_routes, route_arcs
from td_utils import write_toy_instance_files
from test_td_igp import write_igp_instance_files

# TOY1 route 0 -> 1 -> 2 -> 0 prices to 30.0 at departure 40.0 (see test_td_checker).
ROUTES = [[1, 2]]


def _write_bks(directory: Path, *, cost, durations, departures, objective="Duration") -> Path:
    metadata = {
        "authors": "Someone",
        "validated_cost": cost,
        "route_durations": durations,
        "route_departure_times": departures,
    }
    path = directory / f"TOY1.bks.{objective}.json"
    save_json_to_file(
        {"instance_name": "TOY1", "routes": ROUTES, "cost": cost, "metadata": metadata, "objective_function": objective},
        path,
    )
    return path


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


def test_relaxation_compares_marked_fractions_on_display_ties_and_overflow(tmp_path):
    from fractions import Fraction

    for tag, numerator, display in (("tie", 2**53, float(2**53)), ("overflow", 2**1100, None)):
        strict = _write_td_twins(tmp_path / tag, display, display)
        relaxed = relaxation_twin(strict)
        for path, offset in ((strict, 0), (relaxed, 1)):
            payload = json.loads(path.read_text())
            payload["metadata"]["standard"] = {"authoritative_cost": {"numerator": str(numerator + offset), "denominator": "1"}}
            path.write_text(json.dumps(payload))
        assert check_relaxation_pair(strict).inverted
        check = check_relaxation_pair(strict, priced=True)
        assert check.inverted
        assert isinstance(check.priced_cost, Fraction) and check.priced_cost == 30
        assert check.as_dict()["priced_cost_exact"] == {"numerator": "30", "denominator": "1"}


def test_relaxation_priced_routes_can_improve_under_tied_displays(tmp_path):
    from fractions import Fraction

    strict = _write_td_twins(tmp_path, 30.0, 30.0)
    epsilon = Fraction(1, 2**55)
    for path, exact in ((strict, 30 + 2 * epsilon), (relaxation_twin(strict), 30 + epsilon)):
        payload = json.loads(path.read_text())
        payload["metadata"]["exact_evaluation"] = {
            "authoritative_cost": {"numerator": str(exact.numerator), "denominator": str(exact.denominator)}
        }
        path.write_text(json.dumps(payload))
    assert not check_relaxation_pair(strict).inverted
    assert check_relaxation_pair(strict, priced=True).inverted

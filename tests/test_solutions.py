"""External solutions: parsing, id normalisation, checking and BKS submission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.solutions import (
    SolutionFormatError,
    check_routes,
    normalize_routes,
    parse_lkh_mtsp_solution,
    parse_solution_text,
    preview_bks_update,
    read_solution_file,
    read_solution_text,
    submit_bks,
    write_cvrplib_solution,
)

FIXTURES = Path(__file__).parent / "fixtures" / "solvers"


def test_cvrplib_sol_with_and_without_hash() -> None:
    routes, cost = parse_solution_text("Route #1: 3 1\nRoute 2: 2\nCost 17\n")
    assert routes == [[3, 1], [2]] and cost == 17.0


@pytest.mark.parametrize(
    ("fixture", "cost"),
    [("hgs-cvrp.sol", 1.23457e06), ("ails2.sol", 1.2345678e7), ("filo2.sol", 21.0)],
)
def test_real_solver_cost_lines_parse_including_exponents(fixture: str, cost: float) -> None:
    routes, declared = parse_solution_text((FIXTURES / fixture).read_text())
    assert routes == [[2, 1], [3]]
    assert declared == pytest.approx(cost)


def test_lkh_mtsp_solution_drops_the_depot_and_empty_vehicles() -> None:
    routes, penalty, cost = parse_lkh_mtsp_solution((FIXTURES / "lkh3-mtsp.sol").read_text())
    assert routes == [[3, 2], [4]]
    assert (penalty, cost) == (0.0, 21.0)
    parsed = read_solution_text((FIXTURES / "lkh3-mtsp.sol").read_text(), 3)
    assert parsed.format == "lkh-mtsp"
    assert parsed.routes == [[2, 1], [3]]  # LKH node k is library node k-1


def test_lkh_penalty_means_infeasible() -> None:
    text = "p, Cost: 5_21\nThe tours traveled by the 1 salesmen are:\n1 2 3 4 1 (#3)  Cost: 21\n"
    with pytest.raises(SolutionFormatError, match="penalty"):
        read_solution_text(text, 3)


def test_json_solutions() -> None:
    assert parse_solution_text('{"routes": [[1, 2]], "cost": 3.5}') == ([[1, 2]], 3.5)
    assert parse_solution_text("[[2], [1]]") == ([[2], [1]], None)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "The solution is empty."),
        ("{nope", "Not valid JSON"),
        ('{"cost": 3}', "no 'routes' array"),
        ('{"routes": [3]}', "must be a list of stops"),
        ("Route #1: a b", "Unreadable stop id"),
        ("hello", "No 'Route #k"),
    ],
)
def test_parse_errors_keep_their_messages(text: str, message: str) -> None:
    with pytest.raises(SolutionFormatError, match=message.replace("(", r"\(")):
        parse_solution_text(text)


def test_normalize_auto_conventions() -> None:
    assert normalize_routes([[0, 2, 1, 0], [0, 3, 0]], 3) == [[2, 1], [3]]
    assert normalize_routes([[1, 0], [2]], 3) == [[2, 1], [3]]  # 0..n-1 covered once: zero-based
    # Forcing one-based reads the same file as customers 1 and 2 with a depot marker: 3 is missing.
    with pytest.raises(SolutionFormatError, match="never visited"):
        normalize_routes([[1, 0], [2]], 3, convention="one-based")
    assert normalize_routes([[0, 1], [2]], 3, convention="zero-based") == [[1, 2], [3]]


@pytest.mark.parametrize(
    ("routes", "message"),
    [
        ([[1, 1], [2, 3]], "more than once"),
        ([[1], [2]], "never visited"),
        ([[1, 9]], "fits neither"),
        ([[0, 0]], "no customer stops"),
    ],
)
def test_normalize_refuses_what_does_not_fit(routes, message: str) -> None:
    with pytest.raises(SolutionFormatError, match=message):
        normalize_routes(routes, 3)


def test_normalize_refuses_a_non_zero_depot() -> None:
    with pytest.raises(SolutionFormatError, match="depot"):
        normalize_routes([[1, 2]], 2, depot=1)


def test_writer_round_trips() -> None:
    text = write_cvrplib_solution([[2, 1], [3]], 21.5)
    assert text == "Route #1: 2 1\nRoute #2: 3\nCost 21.5\n"
    assert parse_solution_text(text) == ([[2, 1], [3]], 21.5)
    assert write_cvrplib_solution([[1]], 7.0).endswith("Cost 7\n")


def _write(instance, tmp_path: Path) -> Path:
    path = tmp_path / f"{instance.instance_name}.vrp.json"
    path.write_text(json.dumps(instance.model_dump(mode="json")), encoding="utf-8")
    return path


def test_check_routes_static(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    good = check_routes(toy_cvrp_instance, path, [[1, 2]])
    assert good.is_valid() and good.routing_cost == 14
    assert not check_routes(toy_cvrp_instance, path, [[1, 1, 2]]).is_valid()
    assert not check_routes(toy_cvrp_instance, path, [[1, 2]], declared_cost=13).is_valid()


def test_read_solution_file_normalises_against_the_instance(toy_cvrp_instance, tmp_path: Path) -> None:
    sol = tmp_path / "x.sol"
    sol.write_text("Route #1: 0 2 1 0\nCost 14\n")
    parsed = read_solution_file(sol, toy_cvrp_instance)
    assert parsed.routes == [[2, 1]] and parsed.declared_cost == 14 and parsed.format == "cvrplib-sol"


def test_submit_and_preview(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    assert preview_bks_update(path, [[1, 2]])["action"] == "would_create"
    with pytest.raises(ValueError, match="authors"):
        submit_bks(path, [[1, 2]], authors=" ")
    created = submit_bks(path, [[1], [2]], authors="Tester", metadata={"method": "manual"})
    assert created.action == "created"
    stored = load_bks(get_bks_path_for_instance(path, ObjectiveFunction.MONO_COST))
    assert stored.metadata["authors"] == "Tester" and stored.metadata["method"] == "manual"
    assert preview_bks_update(path, [[1, 2]])["action"] == "would_replace"
    assert submit_bks(path, [[1, 2]], authors="Tester").action == "replaced"
    tie = preview_bks_update(path, [[2, 1]])
    assert tie["action"] == "would_keep" and tie["tie"] is True
    assert preview_bks_update(path, [[1, 1]])["action"] == "invalid"


def test_submit_td(tmp_path: Path) -> None:
    from td_utils import write_toy_instance_files

    path = write_toy_instance_files(tmp_path, with_time_windows=True)
    preview = preview_bks_update(path, [[1], [2]])
    assert preview["objective_function"] == "Duration"
    assert preview["action"] == "would_create"
    assert submit_bks(path, [[1], [2]], authors="Tester").action == "created"
    assert preview_bks_update(path, [[1], [2]])["action"] == "would_keep"

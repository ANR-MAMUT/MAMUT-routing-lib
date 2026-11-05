"""`mamut-routing check` and `mamut-routing bks submit`: external solutions through the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
from mamut_routing_lib.cli import app
from mamut_routing_lib.enums import ObjectiveFunction


def _runner() -> CliRunner:
    return CliRunner()


def _write(instance, tmp_path: Path) -> Path:
    path = tmp_path / f"{instance.instance_name}.vrp.json"
    path.write_text(json.dumps(instance.model_dump(mode="json")), encoding="utf-8")
    return path


def _sol(tmp_path: Path, text: str, name: str = "found.sol") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_check_valid_solution_reports_the_checker_cost(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    result = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 1 2\nCost 14\n"))])
    assert result.exit_code == 0, result.output
    assert "verdict    valid" in result.stdout and "cost       14" in result.stdout


def test_check_json_and_declared_cost_mismatch(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    result = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 1 2\nCost 99\n")), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] and payload["cost"] == 14 and payload["declared_cost_matches"] is False
    assert "declares cost 99" in result.stderr


def test_check_invalid_routes_exit_1(toy_cvrp_instance, tmp_path: Path) -> None:
    toy = toy_cvrp_instance.model_copy(update={"vehicle_capacity": 5})
    instance = _write(toy, tmp_path)
    result = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 1 2\n"))])
    assert result.exit_code == 1
    assert "vehicle_capacity_exceeded" in result.stdout


def test_check_unreadable_or_misfit_files_exit_2(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    garbage = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "nothing here"))])
    assert garbage.exit_code == 2 and "No 'Route #k" in garbage.stderr
    misfit = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 1 2 3\n"))])
    assert misfit.exit_code == 2 and "fits neither" in misfit.stderr


def test_check_time_dependent_instance(tmp_path: Path) -> None:
    from td_utils import write_toy_instance_files

    instance = write_toy_instance_files(tmp_path, with_time_windows=True)
    result = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, '{"routes": [[1], [2]]}', "s.json")), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["valid"] and payload["objective_function"] == "Duration"


def test_submit_requires_authors(toy_cvrp_instance, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAMUT_ROUTING_BKS_AUTHORS", raising=False)
    instance = _write(toy_cvrp_instance, tmp_path)
    result = _runner().invoke(app, ["bks", "submit", str(instance), str(_sol(tmp_path, "Route #1: 1 2\n"))])
    assert result.exit_code == 2 and "--authors is required" in result.stderr
    assert not get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST).exists()


def test_submit_creates_keeps_and_records_provenance(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    sol = _sol(tmp_path, "Route #1: 1\nRoute #2: 2\n")
    args = ["bks", "submit", str(instance), str(sol), "--authors", "Jane Doe", "--method", "lkh3",
            "--seed", "3", "--meta", "campaign=test", "--meta", "runs=2"]
    first = _runner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert json.loads(first.stdout)["action"] == "created"
    stored = load_bks(get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST))
    assert stored.metadata["authors"] == "Jane Doe"
    assert stored.metadata["method"] == "lkh3" and stored.metadata["seed"] == 3
    assert stored.metadata["campaign"] == "test" and stored.metadata["runs"] == 2
    again = json.loads(_runner().invoke(app, args).stdout)
    assert again["action"] == "kept_existing" and again["tie"] is True


def test_submit_dry_run_writes_nothing_and_authors_from_env(toy_cvrp_instance, tmp_path: Path, monkeypatch) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    sol = _sol(tmp_path, "Route #1: 1 2\n")
    preview = _runner().invoke(app, ["bks", "submit", str(instance), str(sol), "--dry-run"])
    assert preview.exit_code == 0 and json.loads(preview.stdout)["action"] == "would_create"
    assert not get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST).exists()
    monkeypatch.setenv("MAMUT_ROUTING_BKS_AUTHORS", "Env Author")
    created = _runner().invoke(app, ["bks", "submit", str(instance), str(sol)])
    assert created.exit_code == 0, created.output
    assert load_bks(get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST)).metadata["authors"] == "Env Author"


def test_submit_rejects_an_invalid_solution(toy_cvrp_instance, tmp_path: Path) -> None:
    toy = toy_cvrp_instance.model_copy(update={"vehicle_capacity": 5})
    instance = _write(toy, tmp_path)
    result = _runner().invoke(
        app, ["bks", "submit", str(instance), str(_sol(tmp_path, "Route #1: 1 2\n")), "--authors", "X"]
    )
    assert result.exit_code == 1


@pytest.mark.parametrize("objective", ["hierarchicalvehiclecost"])
def test_submit_warns_when_the_objective_differs_from_the_family(toy_vrptw_instance, tmp_path: Path, objective) -> None:
    instance = _write(toy_vrptw_instance, tmp_path)
    result = _runner().invoke(
        app,
        ["bks", "submit", str(instance), str(_sol(tmp_path, "Route #1: 1\nRoute #2: 2\n")), "--authors", "X",
         "--objective", objective, "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "publishes its BKS under MonoCost" in result.stderr


@pytest.mark.parametrize("command", [["bks", "submit"], ["check"]])
@pytest.mark.parametrize("objective", ["Duration", "FleetCostDuration", "HierarchicalVehicleCost"])
def test_an_objective_that_does_not_apply_is_a_usage_error(toy_cvrp_instance, tmp_path: Path, command, objective) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    args = [*command, str(instance), str(_sol(tmp_path, "Route #1: 1 2\n")), "--objective", objective]
    if command[0] == "bks":
        args += ["--authors", "X"]
    result = _runner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert f"{objective} does not apply to CVRP instances (use MonoCost)" in result.stderr
    assert not list(tmp_path.glob("*.bks.*.json"))


def test_submit_bks_refuses_an_objective_that_does_not_apply(toy_cvrp_instance, tmp_path: Path) -> None:
    from td_utils import write_toy_instance_files

    from mamut_routing_lib.solutions import preview_bks_update, submit_bks

    instance = _write(toy_cvrp_instance, tmp_path)
    with pytest.raises(ValueError, match="Duration does not apply to CVRP instances"):
        submit_bks(instance, [[1, 2]], authors="X", objective_function=ObjectiveFunction.DURATION)
    with pytest.raises(ValueError, match="Duration does not apply to CVRP instances"):
        preview_bks_update(instance, [[1, 2]], objective_function=ObjectiveFunction.DURATION)
    td = write_toy_instance_files(tmp_path / "td", with_time_windows=True)
    with pytest.raises(ValueError, match="MonoCost does not apply to TDVRPTW instances"):
        submit_bks(td, [[1], [2]], authors="X", objective_function=ObjectiveFunction.MONO_COST)
    assert not list(tmp_path.rglob("*.bks.*.json"))


def test_check_says_when_it_reads_ids_as_zero_based(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    result = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 0 1\n"))])
    assert result.exit_code == 0, result.output
    assert "read as 0-based customer ids" in result.stderr and "zero-based ids" in result.stdout
    quiet = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 1 2\n", "b.sol"))])
    assert "0-based" not in quiet.stderr and "one-based ids" in quiet.stdout


def test_check_prints_the_exact_decimal_cost(toy_cvrp_instance, tmp_path: Path) -> None:
    # The route 0-1-2-0 sums 0.1 + 0.2 + 0.4 = 0.7000000000000001 in floats; its exact cost is 0.7.
    costs = [[0.0, 0.1, 0.4], [0.1, 0.0, 0.2], [0.4, 0.2, 0.0]]
    toy = toy_cvrp_instance.model_copy(update={"arc_costs": costs})
    instance = _write(toy, tmp_path)
    solution = _sol(tmp_path, "Route #1: 1 2\n")
    result = _runner().invoke(app, ["check", str(instance), str(solution)])
    assert result.exit_code == 0, result.output
    assert "cost       0.7\n" in result.stdout
    payload = json.loads(_runner().invoke(app, ["check", str(instance), str(solution), "--json"]).stdout)
    assert payload["exact_cost"] == "0.7" and payload["cost"] == 0.7000000000000001


def test_submit_meta_cannot_set_what_the_store_writes(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    solution = _sol(tmp_path, "Route #1: 1 2\n")
    result = _runner().invoke(
        app, ["bks", "submit", str(instance), str(solution), "--authors", "X", "--meta", "date=2020-01-01"]
    )
    assert result.exit_code == 2 and "written by the BKS store" in result.output
    assert not list(tmp_path.glob("*.bks.*.json"))


def test_dry_run_reports_a_stored_bks_that_fails_the_checker(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    solution = _sol(tmp_path, "Route #1: 1 2\n")
    created = _runner().invoke(app, ["bks", "submit", str(instance), str(solution), "--authors", "X"])
    assert created.exit_code == 0, created.output
    stored = next(tmp_path.glob("*.bks.MonoCost.json"))
    payload = json.loads(stored.read_text())
    payload["routes"] = [[1]]  # customer 2 no longer served
    stored.write_text(json.dumps(payload))
    preview = _runner().invoke(app, ["bks", "submit", str(instance), str(solution), "--dry-run"])
    assert preview.exit_code == 1 and json.loads(preview.stdout)["action"] == "stored_invalid"


def test_check_recognises_a_cost_in_scaled_units(toy_cvrp_instance, tmp_path: Path) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    checked = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, "Route #1: 1 2\n"))])
    cost = float(checked.stdout.split("cost")[-1].split()[0])
    scaled = _sol(tmp_path, f"Route #1: 1 2\nCost {cost * 1000:.0f}\n", "scaled.sol")
    result = _runner().invoke(app, ["check", str(instance), str(scaled)])
    assert result.exit_code == 0 and "x 1000" in result.stderr and "scaled integer units" in result.stderr

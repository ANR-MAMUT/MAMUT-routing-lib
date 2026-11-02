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

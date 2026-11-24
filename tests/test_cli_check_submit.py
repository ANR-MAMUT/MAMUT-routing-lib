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
@pytest.mark.parametrize("objective", ["Duration", "FleetCostDuration"])
def test_an_objective_that_does_not_apply_is_a_usage_error(toy_cvrp_instance, tmp_path: Path, command, objective) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    args = [*command, str(instance), str(_sol(tmp_path, "Route #1: 1 2\n")), "--objective", objective]
    if command[0] == "bks":
        args += ["--authors", "X"]
    result = _runner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert f"{objective} does not apply to CVRP instances (use MonoCost or HierarchicalVehicleCost)" in result.stderr
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


def test_check_prints_the_exact_binary64_cost(toy_cvrp_instance, tmp_path: Path) -> None:
    # The exact sum has an odd numerator and a power-of-two denominator.
    costs = [[0.0, 0.1, 0.4], [0.1, 0.0, 0.2], [0.4, 0.2, 0.0]]
    toy = toy_cvrp_instance.model_copy(update={"arc_costs": costs})
    instance = _write(toy, tmp_path)
    solution = _sol(tmp_path, "Route #1: 1 2\n")
    result = _runner().invoke(app, ["check", str(instance), str(solution)])
    assert result.exit_code == 0, result.output
    from fractions import Fraction
    exact = sum(map(Fraction, [0.1, 0.2, 0.4]))
    assert f"cost       {exact}\n" in result.stdout
    payload = json.loads(_runner().invoke(app, ["check", str(instance), str(solution), "--json"]).stdout)
    assert payload["exact_cost"] == str(exact) and payload["cost"] == float(exact)


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


@pytest.mark.parametrize("text", ["null", "42", '"abc"'])
def test_check_json_that_is_not_a_solution_exits_2(toy_cvrp_instance, tmp_path: Path, text: str) -> None:
    instance = _write(toy_cvrp_instance, tmp_path)
    result = _runner().invoke(app, ["check", str(instance), str(_sol(tmp_path, text, "found.json"))])
    assert result.exit_code == 2, result.output
    assert "must be an object with a 'routes' array" in result.stderr


@pytest.mark.parametrize('td', [False, True], ids=['static', 'td'])
def test_marked_bks_input_is_verified_on_check_preview_and_submit(toy_cvrp_instance, tmp_path, td):
    from td_utils import write_toy_instance_files

    instance = write_toy_instance_files(tmp_path) if td else _write(toy_cvrp_instance, tmp_path)
    external = _sol(tmp_path, 'Route #1: 1\nRoute #2: 2\n')
    runner = _runner()
    create = runner.invoke(app, ['bks', 'submit', str(instance), str(external), '--authors', 'Finder', '--backend', 'python'])
    assert create.exit_code == 0, create.output
    stored = next(tmp_path.glob('*.bks.*.json'))
    original = stored.read_bytes()
    marked = _sol(tmp_path, original.decode(), 'submission.json')
    commands = [ ['check', str(instance), str(marked), '--json'],
                 ['bks', 'submit', str(instance), str(marked), '--dry-run'],
                 ['bks', 'submit', str(instance), str(marked), '--authors', 'Finder'] ]
    for cmd in commands:
        valid = runner.invoke(app, [*cmd, '--backend', 'python'])
        assert valid.exit_code == 0, valid.output
    payload = json.loads(original)
    record = payload['metadata']['exact_evaluation']
    if td:
        # A later feasible departure has the same duration on the flat TD route.
        record['route_evaluations'][0]['departure_time'] = {'numerator': '1', 'denominator': '1'}
        mismatched = 'route_evaluations'
    else:
        record['authoritative_cost']['numerator'] = '999'
        mismatched = 'authoritative_cost'
    marked.write_text(json.dumps(payload))
    for cmd in commands:
        result = runner.invoke(app, [*cmd, '--backend', 'python'])
        assert result.exit_code == 1 and mismatched in result.output, result.output
        assert stored.read_bytes() == original
    payload = json.loads(original)
    payload['cost'] += 1
    marked.write_text(json.dumps(payload))
    for cmd in commands:
        check = runner.invoke(app, [*cmd, '--backend', 'python'])
        assert check.exit_code == 1 and 'cost' in check.output, check.output
    # Reordering complete routes or inserting depot markers cannot change the asserted route stream.
    payload = json.loads(original)
    payload['routes'].reverse()
    marked.write_text(json.dumps(payload))
    assert runner.invoke(app, [*commands[0], '--backend', 'python']).exit_code == 1
    payload['routes'][0].insert(0, 0)
    marked.write_text(json.dumps(payload))
    assert runner.invoke(app, [*commands[0], '--backend', 'python']).exit_code == 2
    payload = json.loads(original)
    payload['metadata']['standard'] = payload['metadata'].pop('exact_evaluation')
    marked.write_text(json.dumps(payload))
    for cmd in commands:
        result = runner.invoke(app, [*cmd, '--backend', 'python'])
        assert result.exit_code == 2 and 'manifest' in result.output, result.output
    assert stored.read_bytes() == original


@pytest.mark.parametrize(("text", "objective", "token", "encoding"), [
    ("Route #1: 1 2\nCost 1.4000e+01", "unspecified", "1.4000e+01", "decimal-text"),
    ("p, Cost: 0_9007199254740993\n1 2 3 1 (#2) Cost: 9007199254740993",
     "total-arc-cost", "9007199254740993", "integer"),
    ("Route #1: 1 2\nCost 1e999", "unspecified", "1e999", "decimal-text"),
])
def test_submit_preserves_standalone_solver_report(toy_cvrp_instance, tmp_path, text, objective, token, encoding):
    instance = _write(toy_cvrp_instance, tmp_path)
    result = _runner().invoke(app, ["bks", "submit", str(instance), str(_sol(tmp_path, text)), "--authors", "Finder"])
    assert result.exit_code == 0, result.output
    bks = load_bks(get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST))
    assert bks.cost == 14
    assert bks.metadata["solver_report"] == {"objective": objective, "value": token, "encoding": encoding}
    assert "solver_cost" not in bks.metadata


def test_submit_qualifies_raw_cost_with_unit_and_keeps_original_token(toy_cvrp_instance, tmp_path):
    instance = _write(toy_cvrp_instance, tmp_path)
    path = _sol(tmp_path, "Route #1: 1 2\nCost 1.4000E+02")
    report = {"objective": "total-arc-cost", "value": "1.4000E+02", "encoding": "decimal-text",
              "family_units_per_solver_unit": {"numerator": "1", "denominator": "10"}}
    args = ["bks", "submit", str(instance), str(path), "--authors", "Finder"]
    for mode in (["--dry-run"], []):
        invalid = _runner().invoke(app, [*args, *mode, "--meta", "solver_report=" + json.dumps(dict(report, value="140"))])
        assert invalid.exit_code == 1 and "original Cost token" in invalid.stderr
        invalid = _runner().invoke(app, [*args, *mode, "--meta", "solver_report=" + json.dumps(report),
                                         "--meta", "solver_cost=1"])
        assert invalid.exit_code == 1 and "differs" in invalid.stderr
    valid = _runner().invoke(app, [*args, "--meta", "solver_report=" + json.dumps(report)])
    assert valid.exit_code == 0, valid.output
    bks = load_bks(get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST))
    assert bks.metadata["solver_report"] == report and bks.metadata["solver_cost"] == 14.0


def test_submit_json_cost_is_independent_and_report_checked_in_dry_run(toy_cvrp_instance, tmp_path):
    instance = _write(toy_cvrp_instance, tmp_path)
    payload = {"routes": [[1, 2]], "cost": 19}
    path = _sol(tmp_path, json.dumps(payload), "candidate.json")
    args = ["bks", "submit", str(instance), str(path), "--authors", "Finder"]
    valid = _runner().invoke(app, args)
    assert valid.exit_code == 0, valid.output
    bks = load_bks(get_bks_path_for_instance(instance, ObjectiveFunction.MONO_COST))
    assert "solver_report" not in bks.metadata and "solver_cost" not in bks.metadata
    report = {"objective": "total-arc-cost", "encoding": "binary64", "value": "19"}
    path.write_text(json.dumps({**payload, "metadata": {"solver_report": report}}))
    for mode in (["--dry-run"], []):
        invalid = _runner().invoke(app, [*args, *mode])
        assert invalid.exit_code == 1 and "binary64" in invalid.stderr

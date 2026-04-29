from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.bks import BKSUpdateResult
from mamut_routing_lib.cli import app
from mamut_routing_lib.json_utils import save_json_to_file


pytest.importorskip("pyvrp")


def _runner() -> CliRunner:
    return CliRunner()


def _write_instance(tmp_path: Path, instance) -> Path:
    target = tmp_path / f"{instance.instance_id}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def _fake_method_result(*, instance_id: str, problem_type: str = "CVRP", routes=None):
    fake = MagicMock()
    fake.solver_is_feasible = True
    fake.routes = routes or [[1, 2]]
    fake.route_count = len(fake.routes)
    fake.solver_cost = 14
    fake.wall_time = 0.05
    fake.vehicle_penalty = 0
    fake.method = "hgs-v1"
    fake.problem_type = problem_type
    fake.instance_id = instance_id
    return fake


def test_solve_with_positional_path_calls_solver(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    fake_method_result = _fake_method_result(instance_id=toy_cvrp_instance.instance_id)
    fake_update = BKSUpdateResult(
        action="created",
        path=tmp_path / "fake.bks.MonoCost.json",
        previous_path=None,
        candidate_cost=14,
        candidate_num_routes=1,
    )

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_and_update_bks",
        return_value=(fake_method_result, fake_update),
    ) as mock_solve:
        result = _runner().invoke(
            app,
            ["solve", str(instance_path), "--time-limit-s", "1", "--seed", "7"],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    kwargs = mock_solve.call_args.kwargs
    assert kwargs["time_limit_s"] == 1
    assert kwargs["seed"] == 7
    assert toy_cvrp_instance.instance_id in result.stdout
    assert "feasible" in result.stdout
    assert "created" in result.stdout


def test_solve_with_filters_over_output_dir(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    fake_method_result = _fake_method_result(instance_id=toy_cvrp_instance.instance_id)
    fake_update = BKSUpdateResult(
        action="created",
        path=tmp_path / "fake.bks.MonoCost.json",
        previous_path=None,
        candidate_cost=14,
        candidate_num_routes=1,
    )

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_and_update_bks",
        return_value=(fake_method_result, fake_update),
    ) as mock_solve:
        result = _runner().invoke(
            app,
            [
                "--output-dir", str(tmp_path),
                "solve",
                "--problem-type", "CVRP",
                "--time-limit-s", "1",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    selected_instance = mock_solve.call_args.args[0]
    assert selected_instance.instance_id == toy_cvrp_instance.instance_id


def test_solve_no_save_bks_calls_solve_instance_only(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    fake_method_result = _fake_method_result(instance_id=toy_cvrp_instance.instance_id)

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance", return_value=fake_method_result
    ) as mock_solve_instance, patch(
        "mamut_routing_lib.solvers.pyvrp.solve_and_update_bks"
    ) as mock_solve_and_update:
        result = _runner().invoke(
            app,
            ["solve", str(instance_path), "--time-limit-s", "1", "--no-save-bks"],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    mock_solve_and_update.assert_not_called()
    mock_solve_instance.assert_called_once()
    assert "skipped" in result.stdout


def test_solve_no_selection_errors_with_exit_2(tmp_path: Path) -> None:
    result = _runner().invoke(
        app,
        ["--output-dir", str(tmp_path), "solve", "--time-limit-s", "1"],
    )
    assert result.exit_code == 2
    assert "No instances selected" in (result.stderr + result.stdout) or "does not exist" in (result.stderr + result.stdout)

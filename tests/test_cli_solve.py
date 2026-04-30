from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.bks import BKSUpdateResult
from mamut_routing_lib.cli import app
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.models import BenchmarkInstance


pytest.importorskip("pyvrp")


def _runner() -> CliRunner:
    return CliRunner()


def _write_instance(tmp_path: Path, instance) -> Path:
    target = tmp_path / f"{instance.instance_name}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def _write_historical_instance(benchmarks_dir: Path, instance: BenchmarkInstance) -> Path:
    target = (
        benchmarks_dir
        / "VRPTW"
        / instance.benchmark_name.value
        / f"n={instance.num_customers}"
        / f"{instance.instance_name}.vrp.json"
    )
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def _make_historical_vrptw(*, benchmark_name: str) -> BenchmarkInstance:
    return BenchmarkInstance(
        instance_name="C101",
        instance_origin="Solomon1987",
        benchmark_name=benchmark_name,
        num_customers=2,
        num_vehicles=2,
        vehicle_capacity=10,
        coordinates=[(0, 0), (1, 1), (2, 2)],
        demands=[0, 1, 2],
        service_times=[0, 10, 10],
        time_windows=[(0, 100), (0, 100), (0, 100)],
        depot=0,
        arc_costs=[
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ],
        metadata={"metric_variant": "euclidean", "authors": "Marius M. Solomon"},
    )


def _fake_method_result(*, instance_name: str, problem_type: str = "CVRP", routes=None):
    fake = MagicMock()
    fake.solver_is_feasible = True
    fake.routes = routes or [[1, 2]]
    fake.route_count = len(fake.routes)
    fake.solver_cost = 14
    fake.wall_time = 0.05
    fake.vehicle_penalty = 0
    fake.method = "hgs-v1"
    fake.problem_type = problem_type
    fake.instance_id = instance_name
    return fake


def test_solve_with_positional_path_calls_solver(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    fake_method_result = _fake_method_result(instance_name=toy_cvrp_instance.instance_name)
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
    assert toy_cvrp_instance.instance_name in result.stdout
    assert "feasible" in result.stdout
    assert "created" in result.stdout


def test_solve_with_filters_over_benchmarks_dir(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    fake_method_result = _fake_method_result(instance_name=toy_cvrp_instance.instance_name)
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
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--problem-type", "CVRP",
                "--time-limit-s", "1",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    selected_instance = mock_solve.call_args.args[0]
    assert selected_instance.instance_name == toy_cvrp_instance.instance_name


def test_solve_filters_by_derived_instance_id(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    fake_method_result = _fake_method_result(instance_name=toy_vrptw_instance.instance_name, problem_type="VRPTW")
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
                "--benchmarks-dir",
                str(tmp_path),
                "solve",
                "--instance-id",
                "vrptw-mamut2026-fastest-testville-n2-mamut-n2-testvrptw",
                "--time-limit-s",
                "1",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    selected_instance = mock_solve.call_args.args[0]
    assert selected_instance.instance_name == toy_vrptw_instance.instance_name
    assert "vrptw-mamut2026-fastest-testville-n2-mamut-n2-testvrptw" in result.stdout
    assert toy_vrptw_instance.instance_name in result.stdout


def test_solve_no_save_bks_calls_solve_instance_only(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    fake_method_result = _fake_method_result(instance_name=toy_cvrp_instance.instance_name)

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


def test_solve_warns_when_solving_sintef_with_mono_cost(tmp_path: Path) -> None:
    benchmarks_dir = tmp_path / "benchmarks"
    _write_historical_instance(benchmarks_dir, _make_historical_vrptw(benchmark_name="Sintef2008"))
    fake_method_result = _fake_method_result(instance_name="C101", problem_type="VRPTW")

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance", return_value=fake_method_result
    ):
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(benchmarks_dir),
                "solve",
                "--benchmark-name", "Sintef2008",
                "--time-limit-s", "1",
                "--no-save-bks",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Warning:" in result.stderr
    assert "Sintef2008" in result.stderr
    assert "base objective is HierarchicalVehicleCost" in result.stderr
    assert "solving with MonoCost" in result.stderr


def test_solve_warns_when_solving_dimacs_with_hierarchical_cost(tmp_path: Path) -> None:
    benchmarks_dir = tmp_path / "benchmarks"
    _write_historical_instance(benchmarks_dir, _make_historical_vrptw(benchmark_name="Dimacs2021"))
    fake_method_result = _fake_method_result(instance_name="C101", problem_type="VRPTW")

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance", return_value=fake_method_result
    ):
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(benchmarks_dir),
                "solve",
                "--benchmark-name", "Dimacs2021",
                "--objective", "hierarchicalvehiclecost",
                "--time-limit-s", "1",
                "--no-save-bks",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Warning:" in result.stderr
    assert "Dimacs2021" in result.stderr
    assert "base objective is MonoCost" in result.stderr
    assert "solving with HierarchicalVehicleCost" in result.stderr


def test_solve_does_not_warn_for_mamut_objective_choice(tmp_path: Path, toy_vrptw_instance) -> None:
    _write_instance(tmp_path, toy_vrptw_instance)
    fake_method_result = _fake_method_result(instance_name=toy_vrptw_instance.instance_name, problem_type="VRPTW")

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance", return_value=fake_method_result
    ):
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--benchmark-name", "Mamut2026",
                "--objective", "hierarchicalvehiclecost",
                "--time-limit-s", "1",
                "--no-save-bks",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Warning:" not in result.stderr


def test_solve_rejects_hierarchical_objective_against_cvrp_in_batch(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_and_update_bks"
    ) as mock_solve_and_update:
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--objective", "hierarchicalvehiclecost",
                "--time-limit-s", "1",
            ],
        )

    assert result.exit_code == 2
    mock_solve_and_update.assert_not_called()


def test_solve_no_selection_errors_with_exit_2(tmp_path: Path) -> None:
    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "solve", "--time-limit-s", "1"],
    )
    assert result.exit_code == 2
    assert "No instances selected" in (result.stderr + result.stdout) or "does not exist" in (result.stderr + result.stdout)

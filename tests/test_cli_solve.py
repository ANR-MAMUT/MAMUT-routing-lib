from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
from mamut_routing_lib.cli import app
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.models import BenchmarkInstance


pytest.importorskip("pyvrp")

from mamut_routing_lib.solvers.pyvrp import MethodResult  # noqa: E402


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
    routes = routes or [[1, 2]]
    return MethodResult(
        method="pyvrp-ils-v1",
        problem_type=problem_type,
        objective_function="MonoCost",
        instance_id=instance_name,
        seed=7,
        time_limit_s=1,
        wall_time=0.05,
        solver_is_feasible=True,
        routes=routes,
        solver_cost=14,
        route_count=len(routes),
        vehicle_penalty=0,
        metadata={},
    )


def test_solve_with_positional_path_calls_solver(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    fake_method_result = _fake_method_result(instance_name=toy_cvrp_instance.instance_name)

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance",
        return_value=fake_method_result,
    ) as mock_solve:
        result = _runner().invoke(
            app,
            ["solve", str(instance_path), "--time-limit-s", "1", "--seed", "7", "--save-bks", "--authors", "Tester"],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    kwargs = mock_solve.call_args.kwargs
    assert kwargs["time_limit_s"] == 1
    assert kwargs["seed"] == 7
    assert toy_cvrp_instance.instance_name in result.stdout
    assert "feasible" in result.stdout
    assert "created" in result.stdout
    stored = load_bks(get_bks_path_for_instance(instance_path, ObjectiveFunction.MONO_COST))
    assert stored.metadata["authors"] == "Tester"
    assert stored.metadata["solver"] == "pyvrp" and stored.metadata["seed"] == 7


def test_solve_with_filters_over_benchmarks_dir(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    fake_method_result = _fake_method_result(instance_name=toy_cvrp_instance.instance_name)

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance",
        return_value=fake_method_result,
    ) as mock_solve:
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--problem-type", "CVRP",
                "--time-limit-s", "1",
                "--save-bks", "--authors", "Tester",
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

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance",
        return_value=fake_method_result,
    ) as mock_solve:
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir",
                str(tmp_path),
                "solve",
                "--instance-id",
                "vrptw-poryos2026-fastest-testville-n2-poryos-n2-testvrptw",
                "--time-limit-s",
                "1",
                "--save-bks",
                "--authors",
                "Tester",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    selected_instance = mock_solve.call_args.args[0]
    assert selected_instance.instance_name == toy_vrptw_instance.instance_name
    assert "vrptw-poryos2026-fastest-testville-n2-poryos-n2-testvrptw" in result.stdout
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


def test_solve_multi_instance_accepts_jobs_and_keeps_output_order(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    def _fake_solve(instance, **_kwargs):
        return _fake_method_result(
            instance_name=instance.instance_name,
            problem_type=instance.metadata.problem_type,
        )

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance", side_effect=_fake_solve) as mock_solve:
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--time-limit-s", "1",
                "--no-save-bks",
                "--jobs", "2",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 2
    assert "jobs=2" in result.stdout
    assert result.stdout.index(toy_cvrp_instance.instance_name) < result.stdout.index(
        toy_vrptw_instance.instance_name
    )
    assert "skipped" in result.stdout


def test_solve_multi_instance_jobs_one_processes_all(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    def _fake_solve(instance, **_kwargs):
        return _fake_method_result(
            instance_name=instance.instance_name,
            problem_type=instance.metadata.problem_type,
        )

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance", side_effect=_fake_solve) as mock_solve:
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--time-limit-s", "1",
                "--no-save-bks",
                "--jobs", "1",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 2
    assert "jobs=1" in result.stdout
    assert toy_cvrp_instance.instance_name in result.stdout
    assert toy_vrptw_instance.instance_name in result.stdout


def test_solve_single_instance_jobs_keeps_single_instance_behavior(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)
    fake_method_result = _fake_method_result(instance_name=toy_cvrp_instance.instance_name)

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance", return_value=fake_method_result) as mock_solve:
        result = _runner().invoke(
            app,
            [
                "solve",
                str(instance_path),
                "--time-limit-s", "1",
                "--no-save-bks",
                "--jobs", "8",
            ],
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    mock_solve.assert_called_once()
    assert "Solving 1 instance(s)" in result.stdout
    assert "jobs=" not in result.stdout
    assert toy_cvrp_instance.instance_name in result.stdout


def test_solve_display_rejected_for_multi_instance(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance") as mock_solve:
        result = _runner().invoke(
            app,
            [
                "--benchmarks-dir", str(tmp_path),
                "solve",
                "--time-limit-s", "1",
                "--no-save-bks",
                "--display",
            ],
        )

    assert result.exit_code == 2
    mock_solve.assert_not_called()
    assert "--display is only supported" in (result.stderr + result.stdout)


def test_solve_duplicate_positional_paths_rejected_for_multi_instance(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance") as mock_solve:
        result = _runner().invoke(
            app,
            [
                "solve",
                str(instance_path),
                str(instance_path),
                "--time-limit-s", "1",
                "--no-save-bks",
                "--jobs", "2",
            ],
        )

    assert result.exit_code == 2
    mock_solve.assert_not_called()
    assert "duplicate instance path" in (result.stderr + result.stdout)


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
            color=True,
        )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "\x1b[" in result.stderr
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
                "--benchmark-name", "Poryos2026",
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


def test_solve_reports_a_failing_instance_and_keeps_the_batch(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    _write_instance(tmp_path, toy_cvrp_instance)
    _write_instance(tmp_path, toy_vrptw_instance)

    def _fake_solve(instance, **_kwargs):
        if instance.instance_name == toy_vrptw_instance.instance_name:
            raise RuntimeError("solver exploded")
        return _fake_method_result(instance_name=instance.instance_name)

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance", side_effect=_fake_solve):
        result = _runner().invoke(
            app, ["--benchmarks-dir", str(tmp_path), "solve", "--time-limit-s", "1", "--no-save-bks", "--jobs", "2"]
        )

    assert result.exit_code == 1, result.stdout + result.stderr
    assert isinstance(result.exception, SystemExit)
    assert "feasible" in result.stdout and "error" in result.stdout
    assert "RuntimeError: solver exploded" in result.stderr
    assert "Summary: feasible 1 · infeasible 0 · errors 1" in result.stdout


def test_solve_single_instance_error_is_a_clean_exit(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)
    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance", side_effect=ValueError("bad matrix")):
        result = _runner().invoke(app, ["solve", str(instance_path), "--time-limit-s", "1", "--no-save-bks"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "ValueError: bad matrix" in result.stderr


def test_solve_skips_scanned_time_dependent_instances(tmp_path: Path, toy_cvrp_instance) -> None:
    from td_utils import write_toy_instance_files

    _write_instance(tmp_path, toy_cvrp_instance)
    td_dir = tmp_path / "TDVRPTW" / "Dabia2013" / "n=2"
    td_dir.mkdir(parents=True)
    write_toy_instance_files(td_dir)

    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance",
        side_effect=lambda instance, **_: _fake_method_result(instance_name=instance.instance_name),
    ) as mock_solve:
        result = _runner().invoke(app, ["--benchmarks-dir", str(tmp_path), "solve", "--time-limit-s", "1", "--no-save-bks"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert mock_solve.call_count == 1
    assert "skipping 1 time-dependent instance" in result.stderr
    assert "skipped (TD) 1" in result.stdout


def test_solve_refuses_an_explicit_time_dependent_path(tmp_path: Path) -> None:
    from td_utils import write_toy_instance_files

    instance_path = write_toy_instance_files(tmp_path)
    result = _runner().invoke(app, ["solve", str(instance_path), "--time-limit-s", "1", "--no-save-bks"])
    assert result.exit_code == 2
    assert "time-dependent" in result.stderr


def test_solve_handles_unmaterialized_distances_sidecars(tmp_path: Path) -> None:
    from collection_utils import write_toy_collection

    root = write_toy_collection(tmp_path)
    for sidecar in (root / "Poryos2026" / "sidecars").rglob("*.distances-*.json.gz"):
        sidecar.unlink()
    fastest = next((root / "Poryos2026" / "CVRP" / "fastest").rglob("*.vrp.json"))

    def _fake_solve(instance, **_kwargs):
        return _fake_method_result(instance_name=instance.instance_name)

    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance", side_effect=_fake_solve) as mock_solve:
        scanned = _runner().invoke(
            app,
            ["--benchmarks-dir", str(root), "solve", "--problem-type", "CVRP", "--time-limit-s", "1", "--no-save-bks"],
        )
        explicit = _runner().invoke(app, ["solve", str(fastest), "--time-limit-s", "1", "--no-save-bks"])

    assert scanned.exit_code == 0, scanned.stdout + scanned.stderr
    assert "skipping 1 instance(s) whose distances sidecar" in scanned.stderr
    assert "skipped (sidecar) 1" in scanned.stdout
    assert explicit.exit_code == 1
    assert "distances sidecar not in the tree" in explicit.stderr
    assert mock_solve.call_count == 1  # only the euclidean CVRP of the scan


def test_solve_rejects_hierarchical_objective_on_collection_cvrp(tmp_path: Path) -> None:
    from collection_utils import write_toy_collection

    root = write_toy_collection(tmp_path)
    euclidean = next((root / "Poryos2026" / "CVRP" / "euclidean").rglob("*.vrp.json"))
    result = _runner().invoke(
        app, ["solve", str(euclidean), "--objective", "hierarchicalvehiclecost", "--time-limit-s", "1"]
    )
    assert result.exit_code == 2
    assert "VRPTW-only" in result.output


def test_solve_writes_no_bks_by_default(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)
    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance",
        return_value=_fake_method_result(instance_name=toy_cvrp_instance.instance_name),
    ):
        result = _runner().invoke(app, ["solve", str(instance_path), "--time-limit-s", "1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout
    assert not get_bks_path_for_instance(instance_path, ObjectiveFunction.MONO_COST).exists()


def test_solve_save_bks_needs_authors(tmp_path: Path, toy_cvrp_instance, monkeypatch) -> None:
    monkeypatch.delenv("MAMUT_ROUTING_BKS_AUTHORS", raising=False)
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)
    with patch("mamut_routing_lib.solvers.pyvrp.solve_instance") as mock_solve:
        result = _runner().invoke(app, ["solve", str(instance_path), "--time-limit-s", "1", "--save-bks"])
    assert result.exit_code == 2
    assert "--save-bks needs --authors" in result.stderr
    mock_solve.assert_not_called()


def test_solve_unknown_solver_is_a_usage_error(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)
    result = _runner().invoke(app, ["solve", str(instance_path), "--solver", "nope", "--time-limit-s", "1"])
    assert result.exit_code == 2
    assert "unknown solver 'nope'" in result.output


def test_solve_reports_the_checker_cost_not_the_scaled_solver_objective(tmp_path: Path) -> None:
    """Collection instances solve in milli-units; the COST column is the checker's cost."""
    from collection_utils import write_toy_collection

    root = write_toy_collection(tmp_path)
    euclidean = next((root / "Poryos2026" / "CVRP" / "euclidean").rglob("*.vrp.json"))
    result = _runner().invoke(app, ["solve", str(euclidean), "--time-limit-s", "1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    row = next(line for line in result.stdout.splitlines() if "feasible" in line)
    cost = float(row.split()[3])
    assert cost < 1000, f"COST looks scaled x1000: {row}"


def test_solve_rows_infeasible_when_the_checker_rejects(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance.model_copy(update={"vehicle_capacity": 5}))
    with patch(
        "mamut_routing_lib.solvers.pyvrp.solve_instance",
        return_value=_fake_method_result(instance_name=toy_cvrp_instance.instance_name),
    ):
        result = _runner().invoke(app, ["solve", str(instance_path), "--time-limit-s", "1"])
    assert result.exit_code == 1
    assert "infeasible" in result.stdout
    assert "vehicle_capacity_exceeded" in result.stderr

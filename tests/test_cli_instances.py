from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mamut_routing_lib.cli import app
from mamut_routing_lib.json_utils import save_json_to_file


def _runner() -> CliRunner:
    return CliRunner()


def _write(tmp_path: Path, instance) -> Path:
    target = tmp_path / f"{instance.instance_name}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def test_list_lists_filtered_with_summary(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    cvrp_path = _write(tmp_path, toy_cvrp_instance)
    _write(tmp_path, toy_vrptw_instance)

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "list", "--problem-type", "CVRP"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "INSTANCE_NAME" in result.stdout
    assert "cvrp-mamut2026-fastest-testville-n2-mamut-n2-testcvrp" in result.stdout
    assert toy_cvrp_instance.instance_name in result.stdout
    assert toy_vrptw_instance.instance_name not in result.stdout
    assert "PATH" not in result.stdout
    assert str(cvrp_path.resolve()) not in result.stdout
    assert "Summary:" in result.stdout
    assert "Scanned        : 2" in result.stdout
    assert "Total          : 1" in result.stdout
    assert "CVRP=1" in result.stdout


def test_list_show_path_includes_path_column(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write(tmp_path, toy_cvrp_instance)

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "list", "--show-path", "--no-summary"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "PATH" in result.stdout
    assert str(instance_path.resolve()) in result.stdout


def test_list_paths_only_emits_one_path_per_line(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    cvrp_path = _write(tmp_path, toy_cvrp_instance)
    vrptw_path = _write(tmp_path, toy_vrptw_instance)

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "list", "--paths-only"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert sorted(lines) == sorted([str(cvrp_path.resolve()), str(vrptw_path.resolve())])
    assert "Summary:" not in result.stdout
    assert "INSTANCE_ID" not in result.stdout


def test_list_no_summary_skips_recap(tmp_path: Path, toy_cvrp_instance) -> None:
    _write(tmp_path, toy_cvrp_instance)

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "list", "--no-summary"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_cvrp_instance.instance_name in result.stdout
    assert "Summary:" not in result.stdout


def test_list_filters_by_derived_instance_id(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write(tmp_path, toy_cvrp_instance)
    _write(tmp_path, toy_vrptw_instance)

    result = _runner().invoke(
        app,
        [
            "--benchmarks-dir",
            str(tmp_path),
            "list",
            "--instance-id",
            "vrptw-mamut2026-fastest-testville-n2-mamut-n2-testvrptw",
            "--no-summary",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_vrptw_instance.instance_name in result.stdout
    assert toy_cvrp_instance.instance_name not in result.stdout


def test_list_filters_by_stored_instance_name(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    _write(tmp_path, toy_cvrp_instance)
    _write(tmp_path, toy_vrptw_instance)

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "list", "--instance-name", toy_cvrp_instance.instance_name, "--no-summary"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_cvrp_instance.instance_name in result.stdout
    assert toy_vrptw_instance.instance_name not in result.stdout


def test_list_empty_selection_exits_2(tmp_path: Path) -> None:
    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(tmp_path), "list"],
    )
    assert result.exit_code == 2
    assert "No instances matched" in (result.stderr + result.stdout)


def test_list_with_positional_path(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write(tmp_path, toy_cvrp_instance)

    result = _runner().invoke(app, ["list", str(instance_path)])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_cvrp_instance.instance_name in result.stdout
    assert "Total          : 1" in result.stdout

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mamut_routing_lib.cli import app
from mamut_routing_lib.json_utils import save_json_to_file


def _runner() -> CliRunner:
    return CliRunner()


def _write(tmp_path: Path, instance) -> Path:
    target = tmp_path / f"{instance.instance_id}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def test_instances_lists_filtered_with_summary(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    _write(tmp_path, toy_cvrp_instance)
    _write(tmp_path, toy_vrptw_instance)

    result = _runner().invoke(
        app,
        ["--output-dir", str(tmp_path), "instances", "--problem-type", "CVRP"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_cvrp_instance.instance_id in result.stdout
    assert toy_vrptw_instance.instance_id not in result.stdout
    assert "Summary:" in result.stdout
    assert "Total          : 1" in result.stdout
    assert "CVRP=1" in result.stdout


def test_instances_paths_only_emits_one_path_per_line(
    tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance
) -> None:
    cvrp_path = _write(tmp_path, toy_cvrp_instance)
    vrptw_path = _write(tmp_path, toy_vrptw_instance)

    result = _runner().invoke(
        app,
        ["--output-dir", str(tmp_path), "instances", "--paths-only"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert sorted(lines) == sorted([str(cvrp_path.resolve()), str(vrptw_path.resolve())])
    assert "Summary:" not in result.stdout
    assert "INSTANCE_ID" not in result.stdout


def test_instances_no_summary_skips_recap(tmp_path: Path, toy_cvrp_instance) -> None:
    _write(tmp_path, toy_cvrp_instance)

    result = _runner().invoke(
        app,
        ["--output-dir", str(tmp_path), "instances", "--no-summary"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_cvrp_instance.instance_id in result.stdout
    assert "Summary:" not in result.stdout


def test_instances_empty_selection_exits_2(tmp_path: Path) -> None:
    result = _runner().invoke(
        app,
        ["--output-dir", str(tmp_path), "instances"],
    )
    assert result.exit_code == 2
    assert "No instances matched" in (result.stderr + result.stdout)


def test_instances_with_positional_path(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write(tmp_path, toy_cvrp_instance)

    result = _runner().invoke(app, ["instances", str(instance_path)])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert toy_cvrp_instance.instance_id in result.stdout
    assert "Total          : 1" in result.stdout

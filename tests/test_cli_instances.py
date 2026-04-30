from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mamut_routing_lib.cli import app, _local_instance_record
from mamut_routing_lib.enums import MetricVariant
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.models import BenchmarkInstance


def _runner() -> CliRunner:
    return CliRunner()


def _write(tmp_path: Path, instance) -> Path:
    target = tmp_path / f"{instance.instance_name}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def _make_historical_vrptw_with_metric_metadata() -> BenchmarkInstance:
    return BenchmarkInstance(
        instance_name="C101",
        instance_origin="Solomon1987",
        benchmark_name="Sintef2008",
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


def test_local_instance_record_preserves_metadata_metric_variant_for_historical_layout(
    tmp_path: Path,
) -> None:
    """4-part historical layouts (Sintef/Dimacs) don't carry a metric segment in the path,
    but enriched metadata may carry `metric_variant`. Path overrides metadata only for
    fields the path actually provides — historical paths must not clobber the metadata's
    `metric_variant`/`place_slug` to None.
    """
    benchmarks_dir = tmp_path / "benchmarks"
    instance_path = (
        benchmarks_dir / "VRPTW" / "Sintef2008" / "n=2" / "C101.vrp.json"
    )
    instance = _make_historical_vrptw_with_metric_metadata()
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    record = _local_instance_record(instance_path, instance, benchmarks_dir)

    assert record.metric_variant == MetricVariant.EUCLIDEAN
    # ID stays in 4-part historical form: no metric segment in the path → no metric in ID.
    assert record.instance_id == "vrptw-sintef2008-n2-C101"


def test_list_displays_metadata_metric_variant_for_historical_instance(tmp_path: Path) -> None:
    benchmarks_dir = tmp_path / "benchmarks"
    instance_path = (
        benchmarks_dir / "VRPTW" / "Sintef2008" / "n=2" / "C101.vrp.json"
    )
    instance = _make_historical_vrptw_with_metric_metadata()
    save_json_to_file(instance.model_dump(mode="json"), instance_path)

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(benchmarks_dir), "list", "--no-summary"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    # The METRIC column should print 'euclidean' for this row, not '-'.
    rows = [line for line in result.stdout.splitlines() if "C101" in line]
    assert rows, result.stdout
    assert any("euclidean" in row for row in rows), result.stdout

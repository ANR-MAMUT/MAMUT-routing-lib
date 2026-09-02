from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mamut_routing_lib.cli import app
from mamut_routing_lib.cvrplib import instance_to_vrp_text
from mamut_routing_lib.json_utils import save_json_to_file

from td_utils import write_toy_instance_files


def _runner() -> CliRunner:
    return CliRunner()


def _write(directory: Path, instance) -> Path:
    target = directory / f"{instance.instance_name}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), target)
    return target


def test_export_single_path_writes_sibling_then_reports_exists_until_forced(tmp_path: Path, toy_cvrp_instance) -> None:
    source = _write(tmp_path, toy_cvrp_instance)

    result = _runner().invoke(app, ["export", "vrp", str(source)])
    assert result.exit_code == 0, result.stdout + result.stderr
    output = tmp_path / "poryos-n2-testcvrp.vrp"
    assert output.read_text(encoding="utf-8") == instance_to_vrp_text(toy_cvrp_instance)
    assert "written" in result.stdout and "Written        : 1" in result.stdout

    output.write_text("stale", encoding="utf-8")
    again = _runner().invoke(app, ["export", "vrp", str(source)])
    assert again.exit_code == 0, again.stdout + again.stderr
    assert "exists" in again.stdout and "Existing       : 1" in again.stdout
    assert output.read_text(encoding="utf-8") == "stale"

    forced = _runner().invoke(app, ["export", "vrp", str(source), "--force"])
    assert forced.exit_code == 0, forced.stdout + forced.stderr
    assert output.read_text(encoding="utf-8").startswith("NAME : poryos-n2-testcvrp\n")


def test_export_output_dir_mirrors_the_benchmarks_tree_and_filters(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance) -> None:
    benchmarks = tmp_path / "benchmarks"
    cvrp_path = _write(benchmarks / "CVRP" / "Poryos2026" / "n=2", toy_cvrp_instance)
    _write(benchmarks / "VRPTW" / "Poryos2026" / "n=2", toy_vrptw_instance)
    out = tmp_path / "out"

    result = _runner().invoke(
        app,
        ["--benchmarks-dir", str(benchmarks), "export", "vrp", "--problem-type", "CVRP", "--output-dir", str(out), "--jobs", "2"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    exported = out / "CVRP" / "Poryos2026" / "n=2" / "poryos-n2-testcvrp.vrp"
    assert exported.is_file()
    assert not (out / "VRPTW").exists()
    assert not cvrp_path.with_suffix("").exists()
    assert "Written        : 1" in result.stdout

    vrptw = _runner().invoke(
        app,
        ["--benchmarks-dir", str(benchmarks), "export", "vrp", "--output-dir", str(out), "--quiet"],
    )
    assert vrptw.exit_code == 0, vrptw.stdout + vrptw.stderr
    assert (out / "VRPTW" / "Poryos2026" / "n=2" / "poryos-n2-testvrptw.vrp").is_file()
    assert "INSTANCE_ID" not in vrptw.stdout
    assert "Existing       : 1" in vrptw.stdout and "Written        : 1" in vrptw.stdout


def test_export_refuses_explicit_time_dependent_paths_but_skips_scanned_ones(tmp_path: Path, toy_cvrp_instance) -> None:
    benchmarks = tmp_path / "benchmarks"
    td_path = write_toy_instance_files(benchmarks / "TDVRPTW" / "Dabia2013" / "n=2")
    _write(benchmarks / "CVRP" / "Poryos2026" / "n=2", toy_cvrp_instance)

    explicit = _runner().invoke(app, ["export", "vrp", str(td_path), "--output-dir", str(tmp_path / "out")])
    assert explicit.exit_code == 2
    assert "time-dependent" in explicit.stderr or "time-dependent" in explicit.stdout

    scan = _runner().invoke(app, ["--benchmarks-dir", str(benchmarks), "export", "vrp", "--output-dir", str(tmp_path / "out")])
    assert scan.exit_code == 0, scan.stdout + scan.stderr
    combined = scan.stdout + scan.stderr
    assert "skipping 1 time-dependent instance(s)" in combined
    assert "Skipped (TD)   : 1" in scan.stdout
    assert (tmp_path / "out" / "CVRP" / "Poryos2026" / "n=2" / "poryos-n2-testcvrp.vrp").is_file()
    assert not list((tmp_path / "out").rglob("TOY1*"))


def test_export_unsupported_option_for_instance_is_reported_and_fails(tmp_path: Path, toy_cvrp_instance) -> None:
    source = _write(tmp_path, toy_cvrp_instance)  # metric fastest, CVRP

    solomon = _runner().invoke(app, ["export", "vrp", str(source), "--format", "solomon", "--output-dir", str(tmp_path / "o")])
    assert solomon.exit_code == 1, solomon.stdout + solomon.stderr
    assert "unsupported" in solomon.stdout and "VRPTW-only" in solomon.stdout
    assert not list((tmp_path / "o").rglob("*")) if (tmp_path / "o").exists() else True

    euc = _runner().invoke(app, ["export", "vrp", str(source), "--edge-weight-type", "euc_2d", "--output-dir", str(tmp_path / "e")])
    assert euc.exit_code == 1, euc.stdout + euc.stderr
    assert "only meaningful for euclidean-metric" in euc.stdout


def test_export_rejects_unknown_format_and_edge_weight_values(tmp_path: Path, toy_cvrp_instance) -> None:
    source = _write(tmp_path, toy_cvrp_instance)
    bad_format = _runner().invoke(app, ["export", "vrp", str(source), "--format", "tsplib"])
    assert bad_format.exit_code == 2
    bad_weights = _runner().invoke(app, ["export", "vrp", str(source), "--edge-weight-type", "GEO"])
    assert bad_weights.exit_code == 2
    missing = _runner().invoke(app, ["export", "vrp", str(tmp_path / "nope.vrp.json")])
    assert missing.exit_code == 2

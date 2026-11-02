"""`mamut-routing solvers list` and `solve --solver` with a configured external solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mamut_routing_lib.cli import app

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

CONFIG = """schema_version = 1
[solvers.fake]
label = "Fake"
argv = ["{python}", "-m", "mamut_routing_lib.solvers.external._fake_solver", "--params", "{params}"]
runtime = "python"
input_format = "lkh"
output_format = "lkh-mtsp"
problem_types = ["CVRP", "VRPTW"]
"""


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "solvers.toml"
    path.write_text(CONFIG)
    return path


def _instance(instance, tmp_path: Path) -> Path:
    path = tmp_path / f"{instance.instance_name}.vrp.json"
    path.write_text(json.dumps(instance.model_dump(mode="json")))
    return path


def test_solvers_list_json_includes_configured_solvers(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["solvers", "list", "--json", "--solvers-config", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    names = [row["name"] for row in payload["solvers"]]
    assert names[:6] == ["pyvrp", "kayros", "filo2", "hgs-cvrp", "ails2", "lkh3"] and "fake" in names
    fake = next(row for row in payload["solvers"] if row["name"] == "fake")
    assert fake["available"] and fake["source"].endswith("solvers.toml")
    assert payload["config"].endswith("solvers.toml")


def test_solvers_list_reports_a_broken_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 1\n[solvers.filo2]\nexecutabel = "x"\n')
    result = CliRunner().invoke(app, ["solvers", "list", "--solvers-config", str(bad)])
    assert result.exit_code == 2 and "unknown key" in result.stderr


def test_solve_with_a_configured_external_solver_and_kept_artifacts(tmp_path: Path, toy_vrptw_instance) -> None:
    instance = _instance(toy_vrptw_instance, tmp_path)
    artifacts = tmp_path / "artifacts"
    result = CliRunner().invoke(
        app,
        ["solve", str(instance), "--solver", "fake", "--solvers-config", str(_config(tmp_path)),
         "--time-limit-s", "5", "--keep-artifacts", str(artifacts), "--save-bks", "--authors", "Tester"],
    )
    assert result.exit_code == 0, result.output
    assert "solver=fake" in result.stdout and "created" in result.stdout
    kept = list(artifacts.rglob("problem.par"))
    assert len(kept) == 1 and (kept[0].parent / "solver.log").is_file()
    bks = json.loads(next(tmp_path.glob("*.bks.MonoCost.json")).read_text())
    assert bks["metadata"]["solver"] == "fake" and bks["metadata"]["input_sha256"]


def test_solve_with_an_unavailable_external_solver_is_a_usage_error(tmp_path: Path, toy_cvrp_instance, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    result = CliRunner().invoke(app, ["solve", str(_instance(toy_cvrp_instance, tmp_path)), "--solver", "lkh3"])
    assert result.exit_code == 2
    assert "not available" in result.stderr and "solvers.toml" in result.stderr


def test_solve_skips_instances_a_solver_cannot_take(tmp_path: Path) -> None:
    from collection_utils import write_toy_collection

    root = write_toy_collection(tmp_path)
    config = tmp_path / "x.toml"
    config.write_text(CONFIG.replace('problem_types = ["CVRP", "VRPTW"]', 'problem_types = ["CVRP"]'))
    result = CliRunner().invoke(
        app,
        ["--benchmarks-dir", str(root), "solve", "--solver", "fake", "--solvers-config", str(config),
         "--benchmark-name", "Poryos2026", "--time-limit-s", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "skipped (unsupported) 1" in result.stdout  # the VRPTW variant
    assert "skipped (TD) 2" in result.stdout

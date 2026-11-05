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


def test_solvers_list_reports_a_broken_table_and_keeps_the_others(tmp_path: Path) -> None:
    bad = _config(tmp_path)
    bad.write_text(bad.read_text() + '[solvers.filo2]\nexecutabel = "x"\n')
    result = CliRunner().invoke(app, ["solvers", "list", "--solvers-config", str(bad)])
    assert result.exit_code == 0, result.output
    filo2 = next(line for line in result.stdout.splitlines() if line.startswith("filo2 "))
    assert "config error" in filo2 and "did you mean 'executable'" in filo2
    fake = next(line for line in result.stdout.splitlines() if line.startswith("fake "))
    assert " yes " in fake


def test_solvers_list_refuses_a_file_that_is_not_toml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 1\n[solvers.filo2\n')
    result = CliRunner().invoke(app, ["solvers", "list", "--solvers-config", str(bad)])
    assert result.exit_code == 2 and "bad.toml" in result.stderr


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


def _interrupt_cli(args: list[str], *, ready, env: dict[str, str], timeout_s: float = 30.0):
    """Run the CLI in a subprocess, send SIGINT once ``ready()`` holds, return (rc, output, seconds after SIGINT)."""
    import os
    import signal
    import subprocess
    import time

    child = subprocess.Popen(
        [sys.executable, "-c", "from mamut_routing_lib.cli import app; app()", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env},
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        deadline = time.monotonic() + timeout_s
        while not ready():
            assert child.poll() is None, child.stdout.read().decode()
            assert time.monotonic() < deadline, "the solves never started"
            time.sleep(0.1)
        time.sleep(0.5)
        child.send_signal(signal.SIGINT)
        sent = time.monotonic()
        output, _ = child.communicate(timeout=timeout_s)
        return child.returncode, output.decode(), time.monotonic() - sent
    finally:
        if child.poll() is None:
            child.kill()


def _solver_processes(marker: Path) -> list[str]:
    import shutil
    import subprocess

    if shutil.which("pgrep") is None:
        pytest.skip("pgrep is needed to look for orphaned solvers")
    found = subprocess.run(["pgrep", "-af", str(marker)], capture_output=True, text=True)
    return [line for line in found.stdout.splitlines() if "_fake_solver" in line]


@pytest.mark.parametrize("batch", [False, True], ids=["one-instance", "parallel-batch"])
def test_ctrl_c_stops_running_external_solves_at_once(tmp_path: Path, toy_cvrp_instance, toy_vrptw_instance, batch) -> None:
    paths = [_instance(toy_cvrp_instance, tmp_path)]
    if batch:
        paths.append(_instance(toy_vrptw_instance, tmp_path))
    artifacts = tmp_path / "artifacts"
    args = ["solve", *map(str, paths), "--solver", "fake", "--solvers-config", str(_config(tmp_path)),
            "--time-limit-s", "120", "--keep-artifacts", str(artifacts), "--save-bks", "--authors", "Tester"]
    if batch:
        args += ["--jobs", "2"]
    rc, output, after = _interrupt_cli(
        args,
        ready=lambda: len(list(artifacts.rglob("solver.log"))) == len(paths),
        env={"FAKE_MODE": "sleep"},
    )
    assert rc == 130, output
    assert after < 10, f"Ctrl-C took {after:.1f}s"
    assert "every running solve was stopped and wrote no BKS" in output
    assert not list(tmp_path.glob("*.bks.*.json"))
    assert _solver_processes(artifacts) == []


def test_ctrl_c_stops_a_pyvrp_solve_at_once(tmp_path: Path, toy_cvrp_instance) -> None:
    import time

    pytest.importorskip("pyvrp")
    path = _instance(toy_cvrp_instance, tmp_path)
    launched = time.monotonic()
    rc, output, after = _interrupt_cli(
        ["solve", str(path), "--time-limit-s", "120", "--save-bks", "--authors", "Tester"],
        ready=lambda: time.monotonic() - launched > 4,  # PyVRP leaves no file behind while it solves
        env={},
    )
    assert rc == 130, output
    assert after < 10, f"Ctrl-C took {after:.1f}s"
    assert not list(tmp_path.glob("*.bks.*.json"))


# --------------------------------------------------------------------------- #
# Usability: saved solutions, gap to the BKS, flag aliases, starter config
# --------------------------------------------------------------------------- #


def test_solve_output_dir_keeps_solutions_that_check_and_submit_read_back(tmp_path: Path, toy_cvrp_instance) -> None:
    instance = _instance(toy_cvrp_instance, tmp_path)
    config = _config(tmp_path)
    out = tmp_path / "solutions"
    result = CliRunner().invoke(
        app,
        ["solve", str(instance), "--solver", "fake", "--solvers-config", str(config), "--time-limit", "5", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "GAP" in result.stdout and "no BKS" in result.stdout and "not saved" in result.stdout
    assert "Solutions: 1 written" in result.stdout
    assert "beat the stored BKS (or have none)" in result.stderr and "bks submit" in result.stderr
    [written] = list(out.glob("*.json"))
    assert written.name.endswith(".fake-seed42.json")
    payload = json.loads(written.read_text())
    assert payload["metadata"]["validation"] == "valid" and payload["metadata"]["bks_metadata"]["solver"] == "fake"
    assert not list(tmp_path.glob("*.bks.*.json"))  # nothing stored by default

    checked = CliRunner().invoke(app, ["check", str(instance), str(written)])
    assert checked.exit_code == 0, checked.output
    submitted = CliRunner().invoke(app, ["bks", "submit", str(instance), str(written), "--authors", "Tester"])
    assert submitted.exit_code == 0, submitted.output and json.loads(submitted.stdout)["action"] == "created"
    bks = json.loads(next(tmp_path.glob("*.bks.MonoCost.json")).read_text())
    assert bks["metadata"]["solver"] == "fake" and bks["metadata"]["seed"] == 42 and bks["metadata"]["time_limit_s"] == 5

    again = CliRunner().invoke(
        app, ["solve", str(instance), "--solver", "fake", "--solvers-config", str(config), "--time-limit-s", "5"]
    )
    assert again.exit_code == 0, again.output
    row = next(line for line in again.stdout.splitlines() if "feasible" in line)
    assert "0.00%" in row  # same routes as the stored BKS
    assert "The routes were not kept: pass --output-dir" in again.stderr


def test_solve_accepts_the_tools_flag_names(tmp_path: Path, toy_cvrp_instance) -> None:
    instance = _instance(toy_cvrp_instance, tmp_path)
    result = CliRunner().invoke(
        app,
        ["solve", str(instance), "--solver", "fake", "--solvers-config", str(_config(tmp_path)),
         "--time-limit", "5", "--update-bks", "--authors", "Tester"],
    )
    assert result.exit_code == 0, result.output
    assert "created" in result.stdout and list(tmp_path.glob("*.bks.MonoCost.json"))


def test_solve_help_shows_the_pyvrp_extra() -> None:
    result = CliRunner().invoke(app, ["solve", "--help"], env={"COLUMNS": "200"})
    assert 'mamut-routing-lib[pyvrp]' in result.stdout


def test_gap_to_bks_text() -> None:
    from mamut_routing_lib.cli import _gap_to_bks
    from mamut_routing_lib.enums import ObjectiveFunction

    mono, hier = ObjectiveFunction.MONO_COST, ObjectiveFunction.HIERARCHICAL_VEHICLE_COST
    assert _gap_to_bks(101.0, 3, (100.0, 3), mono) == ("+1.00%", False)
    assert _gap_to_bks(99.0, 3, (100.0, 3), mono) == ("-1.00%", True)
    assert _gap_to_bks(100.0, 3, (100.0, 3), mono) == ("0.00%", False)
    assert _gap_to_bks(99.9999, 3, (100.0, 3), mono) == ("-<0.01%", True)
    assert _gap_to_bks(90.0, 4, (100.0, 3), hier) == ("+1 veh", False)
    assert _gap_to_bks(120.0, 2, (100.0, 3), hier) == ("-1 veh", True)
    assert _gap_to_bks(50.0, 2, None, mono) == ("no BKS", True)


def test_solvers_init_writes_a_starter_and_never_overwrites(tmp_path: Path, monkeypatch) -> None:
    from mamut_routing_lib.solvers.external.config import default_config_path, load_config

    monkeypatch.delenv("MAMUT_ROUTING_SOLVERS_CONFIG", raising=False)
    result = CliRunner().invoke(app, ["solvers", "init"])
    assert result.exit_code == 0, result.output
    target = default_config_path()
    assert target.is_file() and "[solvers.filo2]" in target.read_text() and "not read by default" not in result.stderr
    assert load_config(target).errors == {}
    listed = CliRunner().invoke(app, ["solvers", "list"])
    assert listed.exit_code == 0 and f"solvers config: {target.resolve()}" in listed.stdout
    again = CliRunner().invoke(app, ["solvers", "init"])
    assert again.exit_code == 2 and "--force" in again.stderr
    elsewhere = tmp_path / "elsewhere.toml"
    other = CliRunner().invoke(app, ["solvers", "init", str(elsewhere)])
    assert other.exit_code == 0 and elsewhere.is_file() and "not read by default" in other.stderr
    printed = CliRunner().invoke(app, ["solvers", "init", "--print"])
    assert printed.exit_code == 0 and printed.stdout.startswith("# MAMUT-routing solvers config")


def test_solvers_list_explains_how_to_enable_once(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    result = CliRunner().invoke(app, ["solvers", "list"], env={"MAMUT_ROUTING_SOLVERS_CONFIG": ""})
    assert result.exit_code == 0, result.output
    table, _, enable = result.stdout.partition("To enable")
    assert "solvers.toml" not in table.split("\n", 1)[1]  # the path is printed once, not per row
    assert enable.count("\n  ails2 ") == 1 and enable.count("solvers.ails2.jar") == 1  # said once (it was twice)
    assert "patch_ails2.py" in enable and "hgs-cvrp-explicit-matrix.patch" in enable


def test_workbench_cvrp_labelled_mamut2026_loads(tmp_path: Path, toy_cvrp_instance) -> None:
    """Workbench instances generated before the Mamut2026 -> Poryos2026 rename are embedded CVRP payloads."""
    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.models import BenchmarkInstanceCVRP

    payload = toy_cvrp_instance.model_dump(mode="json")
    payload["benchmark_name"] = "Mamut2026"
    payload["metadata"] = {"metric_variant": "euclidean", "place_slug": "lorient"}
    path = tmp_path / "lorient_poi-n2-k1_euclidean.vrp.json"
    path.write_text(json.dumps(payload))
    assert isinstance(load_benchmark_instance(path), BenchmarkInstanceCVRP)

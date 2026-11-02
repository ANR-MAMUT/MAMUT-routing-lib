"""ExternalSolver end to end with the fake executable: inputs, scaling, outputs, failures, provenance."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers import InstanceNotSupportedError, get_solver, run_solver, solver_names
from mamut_routing_lib.solvers.base import SolveMonitor, SolverCapabilities
from mamut_routing_lib.solvers.external.adapter import ExternalSolver
from mamut_routing_lib.solvers.external.catalog import BUILTIN_DESCRIPTORS
from mamut_routing_lib.solvers.external.descriptor import ExternalSolverDescriptor

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

FAKE = ("{python}", "-m", "mamut_routing_lib.solvers.external._fake_solver")
CVRP_MONO = {ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST})}

FAKE_X = ExternalSolverDescriptor(
    name="fake-x",
    label="FakeX",
    method="fake-x",
    argv=(*FAKE, "--problem", "{problem}", "--solution", "{solution}", "--seed", "{seed}", "--time-limit", "{time_limit_s}"),
    input_format="cvrplib-x",
    output_format="cvrplib-sol",
    capabilities=SolverCapabilities(objectives=CVRP_MONO, input="coordinates", symmetric_only=True, fleet_limit=False),
    runtime="python",
    distance_model="float",
    coordinate_scales=(1.0,),
    progress_pattern=r"progress cost=(?P<cost>[0-9.]+)",
    env={"FAKE_DISTANCE": "float"},
    timeout_grace_s=0.3,
    timeout_grace_fraction=0.0,
    term_grace_s=0.3,
)

FAKE_LKH = ExternalSolverDescriptor(
    name="fake-lkh",
    label="FakeLKH",
    method="fake-lkh",
    argv=(*FAKE, "--params", "{params}"),
    input_format="lkh",
    output_format="lkh-mtsp",
    capabilities=SolverCapabilities(
        objectives={
            ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST}),
            ProblemType.VRPTW: frozenset({ObjectiveFunction.MONO_COST}),
        },
        input="explicit",
        integer_costs=True,
    ),
    runtime="python",
    distance_model="integer",
    timeout_grace_s=0.3,
    timeout_grace_fraction=0.0,
    term_grace_s=0.3,
)


def _env(monkeypatch, **values: str) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _write(instance, tmp_path: Path, name: str | None = None) -> Path:
    path = tmp_path / f"{name or instance.instance_name}.vrp.json"
    path.write_text(json.dumps(instance.model_dump(mode="json")), encoding="utf-8")
    return path


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    from collection_utils import write_toy_collection

    return write_toy_collection(tmp_path)


def _variant(root: Path, problem: str, metric: str) -> Path:
    return next((root / "Poryos2026" / problem / metric).rglob("*.vrp.json"))


def test_coordinate_solver_on_a_euclidean_collection_instance(collection: Path) -> None:
    solved = run_solver(ExternalSolver(FAKE_X), _variant(collection, "CVRP", "euclidean"), time_limit_s=5, seed=4)
    result = solved.result
    assert result.status == "feasible" and solved.valid
    assert result.solver == "fake-x" and result.cost_exact is False
    assert result.provenance["scale"]["mode"] == "coordinates"
    assert result.solver_cost == pytest.approx(solved.cost, abs=1e-3)
    assert result.provenance["input_sha256"] and result.provenance["argv"][0] == sys.executable
    assert "host" not in result.provenance
    assert solved.bks_metadata()["input_sha256"] == result.provenance["input_sha256"]


def test_coordinate_solver_refuses_a_road_metric(collection: Path) -> None:
    with pytest.raises(InstanceNotSupportedError, match="coordinates"):
        run_solver(ExternalSolver(FAKE_X), _variant(collection, "CVRP", "fastest"), time_limit_s=1)


def test_lkh_style_solver_scales_exactly_and_cross_checks_its_cost(collection: Path) -> None:
    solved = run_solver(ExternalSolver(FAKE_LKH), _variant(collection, "CVRP", "fastest"), time_limit_s=5)
    result = solved.result
    assert solved.valid
    assert result.cost_exact is True and result.cost_scale == 1000
    assert result.provenance["problem_type_line"] == "CVRP"
    assert result.provenance["reported_cost_matches_checker"] is True
    assert result.solver_cost == pytest.approx(solved.cost)


def test_lkh_style_solver_on_vrptw_writes_cvrptw(collection: Path) -> None:
    solved = run_solver(ExternalSolver(FAKE_LKH), _variant(collection, "VRPTW", "fastest"), time_limit_s=5)
    assert solved.valid
    assert solved.result.provenance["problem_type_line"] == "CVRPTW"


def test_asymmetric_matrix_goes_to_lkh_as_acvrp(toy_cvrp_instance, tmp_path: Path) -> None:
    asym = toy_cvrp_instance.model_copy(update={"arc_costs": [[0, 5, 6], [7, 0, 3], [6, 4, 0]]})
    solved = run_solver(ExternalSolver(FAKE_LKH), _write(asym, tmp_path), time_limit_s=5)
    assert solved.valid and solved.result.provenance["problem_type_line"] == "ACVRP"
    assert solved.result.provenance["vehicles"] >= 1


def test_symmetric_only_solver_refuses_an_asymmetric_euclidean_claim(toy_cvrp_instance, tmp_path: Path) -> None:
    """A matrix declared euclidean but asymmetric is caught at solve time."""
    meta = {**toy_cvrp_instance.metadata.model_dump(mode="json"), "metric_variant": "euclidean"}
    asym = toy_cvrp_instance.model_copy(update={"arc_costs": [[0, 1, 3], [2, 0, 1], [3, 1, 0]], "metadata": meta})
    with pytest.raises(InstanceNotSupportedError):
        run_solver(ExternalSolver(FAKE_X), _write(asym, tmp_path), time_limit_s=1)


@pytest.mark.parametrize(
    ("mode", "status", "message"),
    [
        ("crash", "failed", "no solution file"),
        ("no-output", "failed", "no solution file"),
        ("garbage", "failed", "unreadable solution file"),
    ],
)
def test_solver_failures_are_results_not_exceptions(collection: Path, monkeypatch, mode, status, message) -> None:
    _env(monkeypatch, FAKE_MODE=mode)
    solved = run_solver(ExternalSolver(FAKE_LKH), _variant(collection, "CVRP", "fastest"), time_limit_s=5)
    assert solved.result.status == status and message in solved.result.error
    assert not solved.valid


def test_zero_based_ids_are_understood(collection: Path, monkeypatch) -> None:
    _env(monkeypatch, FAKE_MODE="zero-based")
    solved = run_solver(ExternalSolver(FAKE_X), _variant(collection, "CVRP", "euclidean"), time_limit_s=5)
    assert solved.valid


def test_a_wrong_reported_cost_is_flagged_and_ignored(collection: Path, monkeypatch) -> None:
    _env(monkeypatch, FAKE_MODE="wrong-cost")
    solved = run_solver(ExternalSolver(FAKE_LKH), _variant(collection, "CVRP", "fastest"), time_limit_s=5)
    assert solved.valid
    assert solved.result.provenance["reported_cost_matches_checker"] is False
    assert solved.cost == pytest.approx(solved.result.solver_cost - 1.0)  # 1000 scaled units = 1.0


def test_nonzero_exit_with_a_solution_keeps_the_solution(collection: Path, monkeypatch) -> None:
    _env(monkeypatch, FAKE_MODE="nonzero-with-solution")
    solved = run_solver(ExternalSolver(FAKE_X), _variant(collection, "CVRP", "euclidean"), time_limit_s=5)
    assert solved.valid and "exited with code 7" in solved.result.error


def test_timeout_and_cancellation(collection: Path, monkeypatch) -> None:
    _env(monkeypatch, FAKE_MODE="sleep")
    path = _variant(collection, "CVRP", "fastest")
    timed_out = run_solver(ExternalSolver(FAKE_LKH), path, time_limit_s=0.3)
    assert timed_out.result.status == "timeout" and not timed_out.valid

    monitor = SolveMonitor(None, interval_s=0.0)
    timer = threading.Timer(0.3, monitor.request_stop)
    timer.start()
    cancelled = run_solver(ExternalSolver(FAKE_LKH), path, time_limit_s=60, monitor=monitor)
    timer.cancel()
    assert cancelled.result.status == "cancelled" and cancelled.result.stopped_early


def test_live_progress_is_reported_in_instance_units(collection: Path, monkeypatch) -> None:
    _env(monkeypatch, FAKE_MODE="slow", FAKE_SLEEP="0.6")
    ticks = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    solved = run_solver(ExternalSolver(FAKE_X), _variant(collection, "CVRP", "euclidean"), time_limit_s=5, monitor=monitor)
    assert solved.valid
    assert {tick.phase for tick in ticks} >= {"solving"}
    incumbents = [tick.best_cost for tick in ticks if tick.best_cost is not None]
    assert incumbents and min(incumbents) == pytest.approx(solved.cost, abs=1e-3)


def test_keep_artifacts(collection: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "artifacts"
    solved = run_solver(
        ExternalSolver(FAKE_LKH), _variant(collection, "CVRP", "fastest"), time_limit_s=5, workdir=workdir, keep_artifacts=True
    )
    assert solved.valid
    assert (workdir / "problem.vrp").is_file() and (workdir / "problem.par").is_file() and (workdir / "solver.log").is_file()
    assert solved.result.provenance["artifacts_dir"] == str(workdir.resolve())


def test_a_temporary_workdir_is_removed(collection: Path) -> None:
    solved = run_solver(ExternalSolver(FAKE_LKH), _variant(collection, "CVRP", "fastest"), time_limit_s=5)
    assert "artifacts_dir" not in solved.result.provenance


def test_builtins_are_registered_and_explain_missing_executables(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    names = solver_names()
    for name in ("filo2", "hgs-cvrp", "ails2", "lkh3"):
        assert name in names
        availability = get_solver(name).availability()
        assert not availability.available
        assert "solvers.toml" in availability.hint
    assert "patch" in BUILTIN_DESCRIPTORS["ails2"].requires_note


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_builtin_argv_templates(collection: Path, tmp_path: Path, toy_cvrp_instance) -> None:
    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.solvers.base import SolveRequest

    path = _variant(collection, "CVRP", "euclidean")
    request = SolveRequest(
        instance=load_benchmark_instance(path), instance_path=path, objective=ObjectiveFunction.MONO_COST,
        time_limit_s=12.5, seed=3,
    )
    filo = ExternalSolver(BUILTIN_DESCRIPTORS["filo2"].with_overrides(executable=str(_executable(tmp_path, "filo2"))))
    (tmp_path / "w1").mkdir()
    prepared = filo.prepare(request, tmp_path / "w1")
    argv = filo.build_argv(prepared, request)
    assert argv[-2:] == ["--optimization-seconds", "12"]
    assert prepared.solution.name == "problem.vrp_seed-3.vrp.sol"
    assert prepared.plan.unit == "mm"
    header = prepared.problem.read_text().splitlines()[:6]
    assert header[5] == "CAPACITY : 10" and not any(line.startswith("VEHICLES") for line in header)

    jar = tmp_path / "AILSII.jar"
    jar.write_text("")
    ails = ExternalSolver(BUILTIN_DESCRIPTORS["ails2"].with_overrides(jar=str(jar), java=sys.executable))
    (tmp_path / "w2").mkdir()
    prepared = ails.prepare(request, tmp_path / "w2")
    argv = ails.build_argv(prepared, request)
    assert argv[:4] == [sys.executable, "-Xmx4g", "-XX:+UseSerialGC", "-XX:ActiveProcessorCount=1"]
    assert argv[argv.index("-limit") + 1] == "12.5" and argv[argv.index("-out") + 1] == str(prepared.solution)

    fleet = toy_cvrp_instance.model_copy(update={"num_vehicles": 2})
    fixed = _write(fleet, tmp_path)
    meta = json.loads(fixed.read_text())
    hgs = ExternalSolver(BUILTIN_DESCRIPTORS["hgs-cvrp"].with_overrides(executable=str(_executable(tmp_path, "hgs"))))
    fleet_request = SolveRequest(
        instance=load_benchmark_instance(fixed), instance_path=fixed, objective=ObjectiveFunction.MONO_COST,
        time_limit_s=5, seed=1,
    )
    (tmp_path / "w3").mkdir()
    prepared = hgs.prepare(fleet_request, tmp_path / "w3")
    assert hgs.build_argv(prepared, fleet_request)[-2:] == ["-veh", "2"]
    assert meta["num_vehicles"] == 2


FAKE_MATRIX = FAKE_X.with_overrides(
    name="fake-matrix",
    label="FakeMatrix",
    method="fake-matrix",
    input_format="cvrplib-matrix",
    capabilities=SolverCapabilities(objectives=CVRP_MONO, input="explicit", symmetric_only=False),
    matrix_scales=(1.0, 0.1),
)


def test_full_matrix_solver_takes_asymmetric_road_metrics(collection: Path, toy_cvrp_instance, tmp_path: Path) -> None:
    asym = toy_cvrp_instance.model_copy(update={"arc_costs": [[0, 5, 6], [7, 0, 3], [6, 4, 0]]})
    for path in (_variant(collection, "CVRP", "fastest"), _write(asym, tmp_path)):
        solved = run_solver(ExternalSolver(FAKE_MATRIX), path, time_limit_s=5)
        assert solved.valid, solved.result.error
        assert solved.result.cost_exact is True and solved.result.provenance["scale"]["rounding"] == "float"
        assert solved.result.solver_cost == pytest.approx(solved.cost)
        assert "reported_cost_matches_checker" not in solved.result.provenance  # float solver: no exact cross-check


def test_builtin_hgs_reads_the_full_matrix(collection: Path, tmp_path: Path, monkeypatch) -> None:
    from mamut_routing_lib.artifacts import load_benchmark_instance
    from mamut_routing_lib.solvers.base import SolveRequest, profile_instance

    hgs = ExternalSolver(BUILTIN_DESCRIPTORS["hgs-cvrp"].with_overrides(executable=str(_executable(tmp_path, "hgs"))))
    path = _variant(collection, "CVRP", "fastest")
    instance = load_benchmark_instance(path)
    assert hgs.supports(profile_instance(instance, path), ObjectiveFunction.MONO_COST).supported
    request = SolveRequest(instance=instance, instance_path=path, objective=ObjectiveFunction.MONO_COST, time_limit_s=5)
    (tmp_path / "w").mkdir()
    prepared = hgs.prepare(request, tmp_path / "w")
    lines = prepared.problem.read_text().splitlines()
    assert "EDGE_WEIGHT_FORMAT : FULL_MATRIX" in lines and lines[lines.index("EDGE_WEIGHT_SECTION") + 1] == "0 120.5 240.25"
    assert prepared.plan.factor == 1.0 and prepared.plan.exact
    argv = hgs.build_argv(prepared, request)
    assert argv[1:3] == [str(prepared.problem), str(prepared.solution)] and "-round" in argv
    monkeypatch.setenv("PATH", "/nonexistent")
    missing = ExternalSolver(BUILTIN_DESCRIPTORS["hgs-cvrp"]).availability()
    assert not missing.available and "hgs-cvrp-explicit-matrix.patch" in missing.hint

"""run_solver: vetting, solving through the registry, and the checker's verdict on the routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyvrp")

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.objectives import base_objective, default_objective
from mamut_routing_lib.solvers import run_solver
from mamut_routing_lib.solvers.base import (
    Availability,
    InstanceNotSupportedError,
    SolveMonitor,
    SolveResult,
    SolverCapabilities,
    SolverUnavailableError,
    Support,
)
from mamut_routing_lib.solvers.pyvrp import solve_cvrp


def _write(instance, tmp_path: Path) -> Path:
    path = tmp_path / f"{instance.instance_name}.vrp.json"
    path.write_text(json.dumps(instance.model_dump(mode="json")), encoding="utf-8")
    return path


def test_run_solver_pyvrp_checks_the_routes(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    solved = run_solver("pyvrp", path, time_limit_s=1, seed=3)
    assert solved.result.solver == "pyvrp"
    assert solved.result.status == "feasible"
    assert solved.valid
    assert solved.cost == solved.check.routing_cost
    assert solved.result.solver_cost == pytest.approx(solved.cost)
    assert solved.objective is ObjectiveFunction.MONO_COST
    metadata = solved.bks_metadata(campaign="x")
    assert metadata["solver"] == "pyvrp" and metadata["campaign"] == "x"
    assert "host" not in metadata and metadata["machine"]


def test_run_solver_vrptw_hierarchical_strips_the_vehicle_penalty(toy_vrptw_instance, tmp_path: Path) -> None:
    path = _write(toy_vrptw_instance, tmp_path)
    solved = run_solver(
        "pyvrp",
        path,
        time_limit_s=2,
        seed=1,
        objective_function=ObjectiveFunction.HIERARCHICAL_VEHICLE_COST,
    )
    assert solved.valid
    assert solved.result.solver_cost == pytest.approx(solved.cost)
    assert solved.result.metadata["vehicle_penalty"] > 0


def test_run_solver_refuses_an_objective_the_solver_does_not_offer(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    with pytest.raises(InstanceNotSupportedError, match="HierarchicalVehicleCost"):
        run_solver("pyvrp", path, time_limit_s=1, objective_function=ObjectiveFunction.HIERARCHICAL_VEHICLE_COST)


class _Unavailable:
    name = "ghost"
    label = "Ghost"
    capabilities = SolverCapabilities(objectives={ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST})})

    def availability(self):
        return Availability(False, reason="not installed", hint="install it")

    def supports(self, profile, objective):
        return Support.ok()

    def solve(self, request, monitor=None):  # pragma: no cover
        raise AssertionError("must not be called")


def test_run_solver_refuses_an_unavailable_solver_before_any_work(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    with pytest.raises(SolverUnavailableError, match="not installed .*install it"):
        run_solver(_Unavailable(), path, time_limit_s=1)


class _Bogus(_Unavailable):
    name = "bogus"

    def availability(self):
        return Availability(True)

    def solve(self, request, monitor=None):
        return SolveResult(
            solver="bogus", method="bogus", problem_type="CVRP", objective_function="MonoCost",
            instance_id="x", seed=0, time_limit_s=1, wall_time=0.0, status="feasible",
            routes=[[1, 1, 2]], solver_cost=1.0,
        )


def test_the_checker_has_the_last_word_on_a_solver_claim(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    solved = run_solver(_Bogus(), path, time_limit_s=1)
    assert solved.result.status == "feasible"
    assert not solved.valid
    assert solved.cost is None
    assert solved.to_json()["ok"] is False


def test_monitor_budget_and_clock_are_set_by_run_solver(toy_cvrp_instance, tmp_path: Path) -> None:
    path = _write(toy_cvrp_instance, tmp_path)
    ticks = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    run_solver("pyvrp", path, time_limit_s=1, monitor=monitor)
    assert ticks and ticks[0].budget_s == 1.0


def test_warm_start_routes_seed_a_mono_cost_solve(toy_cvrp_instance) -> None:
    result = solve_cvrp(toy_cvrp_instance, time_limit_s=1, seed=2, initial_routes=[[1], [2]])
    assert result.solver_is_feasible
    assert result.metadata["warm_start"] is True
    assert result.metadata["cost_exact"] is True


def test_a_partial_warm_start_is_refused_with_its_reason(toy_cvrp_instance, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1 customer\\(s\\) missing \\(first: 2\\)"):
        solve_cvrp(toy_cvrp_instance, time_limit_s=1, seed=2, initial_routes=[[1]])
    solved = run_solver("pyvrp", _write(toy_cvrp_instance, tmp_path), time_limit_s=1, initial_routes=[[1, 1], [2]])
    assert solved.result.status == "failed" and "visited more than once" in solved.result.error


def test_objective_defaults(toy_cvrp_instance, toy_vrptw_instance) -> None:
    assert default_objective(toy_vrptw_instance) is ObjectiveFunction.MONO_COST
    assert base_objective(toy_cvrp_instance) is ObjectiveFunction.MONO_COST
    sintef = toy_vrptw_instance.model_copy(update={"benchmark_name": "Sintef2008"})
    assert base_objective(sintef) is ObjectiveFunction.HIERARCHICAL_VEHICLE_COST
    assert default_objective(sintef) is ObjectiveFunction.MONO_COST
    # EURO Meets NeurIPS 2022 ranked total cost: every published Ortec2022 BKS is MonoCost.
    ortec = toy_vrptw_instance.model_copy(update={"benchmark_name": "Ortec2022"})
    assert base_objective(ortec) is ObjectiveFunction.MONO_COST

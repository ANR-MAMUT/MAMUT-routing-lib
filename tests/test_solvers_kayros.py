"""The KAYROS adapter with a stand-in kayros module: checked results, live incumbents, cancellation."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.solvers import InstanceNotSupportedError, get_solver, run_solver
from mamut_routing_lib.solvers.base import SolveMonitor


@dataclass
class _Incumbent:
    value: float
    seconds: float
    iteration: int
    origin: str


@dataclass
class _Solution:
    routes: list
    duration: float
    route_durations: list
    status: str
    iterations: int


def _fake_kayros(*, loop_until_stopped: bool = False, explode: bool = False) -> types.ModuleType:
    module = types.ModuleType("kayros")
    module.__spec__ = importlib.machinery.ModuleSpec("kayros", None)

    def solve(instance, params=None, *, time_limit=None, seed=0, on_incumbent=None):
        from mamut_routing_lib.td import LoadedTDInstance

        assert isinstance(instance, LoadedTDInstance)
        if explode:
            raise RuntimeError("internal/checker disagreement")
        routes = [[1], [2]]
        iteration = 0
        while True:
            iteration += 1
            if on_incumbent is not None:
                on_incumbent(_Incumbent(100.0 - iteration, 0.01 * iteration, iteration, "aco"), routes)
            if not loop_until_stopped or iteration > 10_000:
                break
        return _Solution(routes=routes, duration=99.0, route_durations=[50.0, 49.0], status="time_limit", iterations=iteration)

    module.solve = solve
    return module


@pytest.fixture
def td_instance(tmp_path: Path) -> Path:
    from td_utils import write_toy_instance_files

    return write_toy_instance_files(tmp_path, with_time_windows=True)


def test_kayros_is_listed_and_unavailable_without_the_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kayros", None)
    availability = get_solver("kayros").availability()
    assert not availability.available and "kayros" in availability.hint


def test_kayros_results_are_checked_by_the_td_checker(monkeypatch, td_instance: Path) -> None:
    monkeypatch.setitem(sys.modules, "kayros", _fake_kayros())
    ticks = []
    monitor = SolveMonitor(lambda tick: ticks.append(tick) or False, interval_s=0.0)
    solved = run_solver("kayros", td_instance, time_limit_s=1, seed=5, monitor=monitor)
    assert solved.objective is ObjectiveFunction.DURATION
    assert solved.result.status == "feasible" and solved.valid
    assert solved.cost == solved.check.routing_cost  # the lib's TD checker, not KAYROS's own number
    assert solved.result.metadata["status"] == "time_limit"
    assert ticks and ticks[-1].best_cost == 99.0


def test_kayros_stop_request_keeps_the_last_incumbent(monkeypatch, td_instance: Path) -> None:
    monkeypatch.setitem(sys.modules, "kayros", _fake_kayros(loop_until_stopped=True))
    seen = []

    def on_tick(tick):
        seen.append(tick)
        return len(seen) >= 3

    solved = run_solver("kayros", td_instance, time_limit_s=10, monitor=SolveMonitor(on_tick, interval_s=0.0))
    assert solved.result.status == "cancelled" and solved.result.stopped_early
    assert solved.result.routes == [[1], [2]]


def test_kayros_failure_is_a_result(monkeypatch, td_instance: Path) -> None:
    monkeypatch.setitem(sys.modules, "kayros", _fake_kayros(explode=True))
    solved = run_solver("kayros", td_instance, time_limit_s=1)
    assert solved.result.status == "failed" and "disagreement" in solved.result.error


def test_kayros_refuses_static_instances_and_fleet_cost(monkeypatch, td_instance: Path, toy_cvrp_instance, tmp_path: Path) -> None:
    import json

    monkeypatch.setitem(sys.modules, "kayros", _fake_kayros())
    static = tmp_path / "static.vrp.json"
    static.write_text(json.dumps(toy_cvrp_instance.model_dump(mode="json")))
    with pytest.raises(InstanceNotSupportedError, match="CVRP"):
        run_solver("kayros", static, time_limit_s=1)
    with pytest.raises(InstanceNotSupportedError, match="FleetCostDuration"):
        run_solver("kayros", td_instance, time_limit_s=1, objective_function=ObjectiveFunction.FLEET_COST_DURATION)

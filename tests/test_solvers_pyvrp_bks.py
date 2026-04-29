from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyvrp")

from mamut_routing_lib.artifacts import get_bks_path_for_instance
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.solvers.pyvrp import solve_and_update_bks


def _write_instance(tmp_path: Path, instance) -> Path:
    instance_path = tmp_path / f"{instance.instance_id}.vrp.json"
    save_json_to_file(instance.model_dump(mode="json"), instance_path)
    return instance_path


def test_solve_and_update_bks_creates_bks_when_none_exists(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    method_result, update_result = solve_and_update_bks(
        toy_cvrp_instance,
        instance_path=instance_path,
        time_limit_s=2,
        seed=1,
    )

    assert method_result.solver_is_feasible
    assert update_result is not None
    assert update_result.action == "created"
    expected_bks_path = get_bks_path_for_instance(instance_path, ObjectiveFunction.MONO_COST)
    assert expected_bks_path.exists()


def test_solve_and_update_bks_repeated_run_keeps_or_replaces(tmp_path: Path, toy_cvrp_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_cvrp_instance)

    _, first_update = solve_and_update_bks(
        toy_cvrp_instance, instance_path=instance_path, time_limit_s=2, seed=1
    )
    _, second_update = solve_and_update_bks(
        toy_cvrp_instance, instance_path=instance_path, time_limit_s=2, seed=1
    )

    assert first_update is not None and first_update.action == "created"
    assert second_update is not None
    assert second_update.action in {"kept_existing", "replaced"}


def test_solve_and_update_bks_handles_vrptw_hierarchical(tmp_path: Path, toy_vrptw_instance) -> None:
    instance_path = _write_instance(tmp_path, toy_vrptw_instance)

    method_result, update_result = solve_and_update_bks(
        toy_vrptw_instance,
        instance_path=instance_path,
        time_limit_s=2,
        seed=1,
        objective_function=ObjectiveFunction.HIERARCHICAL_VEHICLE_COST,
    )
    assert method_result.solver_is_feasible
    assert update_result is not None
    assert update_result.action == "created"
    expected_path = get_bks_path_for_instance(
        instance_path, ObjectiveFunction.HIERARCHICAL_VEHICLE_COST
    )
    assert expected_path.exists()

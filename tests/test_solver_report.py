from fractions import Fraction
import json
import math

import pytest

from mamut_routing_lib.models import BenchmarkSolution
from mamut_routing_lib.solvers.report import make_solver_report, solver_cost_display, solver_report_metadata


def test_display_rounds_the_scaled_report_only_once():
    report = make_solver_report("total-arc-cost", "9007199254740993", encoding="integer", unit=Fraction(1, 3))
    assert solver_cost_display(report) == 3002399751580331.0
    assert solver_cost_display(report) != float(report["value"]) / 3


def test_binary64_and_decimal_sources_keep_distinct_rounding_and_json_types():
    binary = make_solver_report("total-route-duration", 0.1, encoding="binary64", unit=Fraction(3))
    printed = make_solver_report("total-arc-cost", "1.00E-1", encoding="decimal-text", unit=Fraction(3))
    assert solver_cost_display(binary) == math.nextafter(0.3, math.inf)
    assert solver_cost_display(printed) == 0.3
    assert json.loads(json.dumps(binary))["value"] == 0.1
    assert json.loads(json.dumps(printed))["value"] == "1.00E-1"


def test_no_display_for_missing_unit_composite_without_distance_or_overflow():
    reports = [
        make_solver_report("total-arc-cost", "10", encoding="integer"),
        make_solver_report("vehicle-penalized-cost", "250", encoding="integer", unit=Fraction(1)),
        make_solver_report("total-arc-cost", "1e1000000000", encoding="decimal-text", unit=Fraction(1)),
    ]
    for report in reports:
        assert solver_report_metadata(report) == {"solver_report": report}
    tiny = make_solver_report("total-arc-cost", "1e-1000000000", encoding="decimal-text", unit=Fraction(1))
    assert solver_cost_display(tiny) == 0.0


def test_large_rational_unit_and_compensating_decimal_exponent():
    report = make_solver_report("total-arc-cost", "1e-5000", encoding="decimal-text", unit=Fraction(10**5000))
    assert solver_cost_display(report) == 1.0


def test_checked_writer_carries_report_and_rejects_a_conflicting_display(tmp_path, toy_cvrp_instance):
    from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks
    from mamut_routing_lib.enums import ObjectiveFunction
    from mamut_routing_lib.solutions import submit_bks

    path = tmp_path / "toy.vrp.json"
    path.write_text(toy_cvrp_instance.model_dump_json())
    report = make_solver_report("total-arc-cost", "1.90E2", encoding="decimal-text", unit=Fraction(1, 10))
    submit_bks(path, [[1], [2]], authors="Example", metadata={"solver_report": report})
    stored = load_bks(get_bks_path_for_instance(path, ObjectiveFunction.MONO_COST))
    assert stored.metadata["solver_cost"] == 19.0
    assert stored.metadata["solver_report"] == report
    assert stored.cost != 19
    with pytest.raises(ValueError, match="solver_cost differs"):
        submit_bks(path, [[1], [2]], authors="Example", metadata={"solver_report": report, "solver_cost": 9})


def test_legacy_metadata_is_readable_and_new_bks_does_not_republish_it(toy_cvrp_instance):
    from mamut_routing_lib.bks import create_bks_from_solution
    from mamut_routing_lib.enums import ObjectiveFunction

    legacy = {"cost_exact": True, "solver_cost": 19, "solver_cost_raw": 1900, "solver_reported_cost": 1900}
    solution = BenchmarkSolution(instance_name=toy_cvrp_instance.instance_name, routes=[[1], [2]], metadata=legacy)
    assert solution.metadata == legacy
    candidate = create_bks_from_solution(toy_cvrp_instance, solution, ObjectiveFunction.MONO_COST,
                                         authors="Example", metadata=legacy)
    assert legacy.keys().isdisjoint(candidate.metadata)


def test_preserve_report_attribution_on_direct_submission(toy_cvrp_instance):
    from mamut_routing_lib.bks import create_bks_from_solution
    from mamut_routing_lib.enums import ObjectiveFunction

    report = make_solver_report("total-arc-cost", "14", encoding="integer", unit=Fraction(1))
    attribution = {"solver": "my-solver", "solver_version": "1.0", "solver_report": report}
    solution = BenchmarkSolution(instance_name=toy_cvrp_instance.instance_name, routes=[[1, 2]], metadata=attribution)
    candidate = create_bks_from_solution(toy_cvrp_instance, solution, ObjectiveFunction.MONO_COST, authors="Example")
    for key, value in attribution.items():
        assert candidate.metadata[key] == value
    with pytest.raises(ValueError, match="Conflicting supplied solver"):
        create_bks_from_solution(toy_cvrp_instance, solution, ObjectiveFunction.MONO_COST,
                                 authors="Example", metadata={"solver": "other-solver"})


@pytest.mark.parametrize("encoding,value", [("binary64", "0.1"), ("binary64", True), ("integer", 1),
                                            ("decimal-text", "NaN"), ("binary64", float("inf"))])
def test_invalid_value_encoding_is_rejected(encoding, value):
    report = make_solver_report("total-arc-cost", value, encoding=encoding, unit=Fraction(1))
    with pytest.raises(ValueError, match="solver_report"):
        solver_cost_display(report)

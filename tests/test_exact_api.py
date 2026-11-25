from fractions import Fraction
import json

import pytest

from mamut_routing_lib import STATIC_PROFILE, TD_PROFILE, evaluate_exact, prepare_exact
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_types import ExactEvaluation
from td_utils import write_toy_instance_files


def test_public_dispatch_and_lossless_json(tmp_path):
    static = tmp_path / "static.vrp.json"
    static.write_text(json.dumps({
        "instance_name": "toy", "instance_origin": "Solomon1987", "benchmark_name": "Poryos2026",
        "coordinates": [[0, 0], [1, 0]], "num_customers": 1, "demands": [0, 1],
        "vehicle_capacity": 1, "arc_costs": [[0, 0.1], [0.2, 0]],
    }))
    result = evaluate_exact(static, [[1]], profile=STATIC_PROFILE, backend="python")
    assert result.cost == Fraction.from_float(0.1) + Fraction.from_float(0.2)
    assert "route_evaluations" not in result.as_dict()
    with pytest.raises(ValueError, match="profile"):
        prepare_exact(static, profile=TD_PROFILE, backend="python")

    td = write_toy_instance_files(tmp_path, gzip_sidecar=True)
    prepared = prepare_exact(td, backend="python", profile=TD_PROFILE)
    result = prepared.evaluate([[2], [1]])
    assert result.routes == ((1,), (2,))
    assert result.cost == sum(w.duration for w in result.route_evaluations) == 45
    assert result.as_dict()["route_evaluations"][0]["departure_time"] == {"numerator": "0", "denominator": "1"}


def test_integer_string_encoding_beyond_cpython_format_limit():
    cost = Fraction(10**5000 + 1, 3)
    result = ExactEvaluation(TD_PROFILE, "python-fraction-partition", ObjectiveFunction.DURATION, (), cost)
    data = result.as_dict()
    assert data["authoritative_cost"] == {"numerator": "1" + "0" * 4999 + "1", "denominator": "3"}
    assert data["cost"] is None and data["display_overflow"] is True


@pytest.mark.parametrize("backend", ["native", "python"])
def test_static_public_checkers_and_costs_follow_family_profile(backend):
    from mamut_routing_lib import check_solution, check_vrptw_solution, compute_route_cost, compute_solution_cost
    from mamut_routing_lib import static_v012
    from mamut_routing_lib.models import BenchmarkSolution
    from test_exact_static import instance

    if backend == "native":
        pytest.importorskip("mamut_routing_lib._static_exact")
    options = {} if backend == "native" else {"backend": backend}
    deadline = 2**53
    model = instance([[0, deadline], [0.5, 0]], windows=[(0, deadline)] * 2,
                     services=[0, 0], instance_name="deadline")
    candidate = BenchmarkSolution(instance_name="deadline", routes=[[1]])
    assert static_v012.check_solution(model, candidate).is_valid()
    assert check_solution(model, candidate, **options).status.value == "time_window_violated"
    assert check_vrptw_solution(model, candidate, **options).status.value == "time_window_violated"
    expected = Fraction(deadline) + Fraction(1, 2)
    assert compute_route_cost(model, [1], **options) == expected
    assert compute_solution_cost(model, candidate, **options) == expected
    model.time_windows = [(0, deadline + 2)] * 2
    verified = check_solution(model, candidate, **options)
    assert verified.is_valid() and verified.routing_cost == deadline
    assert verified.exact_evaluation["authoritative_cost"] == {"numerator": str(2 * deadline + 1), "denominator": "2"}
    assert verified.exact_evaluation["profile"] == STATIC_PROFILE


@pytest.mark.parametrize("backend", ["native", "python"])
def test_td_public_checkers_use_exact_witnesses_and_family_fleet_objective(tmp_path, backend):
    from mamut_routing_lib import td
    from mamut_routing_lib.enums import BenchmarkName
    from mamut_routing_lib.models import BenchmarkSolution
    from test_exact_td import example

    if backend == "native":
        pytest.importorskip("mamut_routing_lib._td_exact")
    options = {} if backend == "native" else {"backend": backend}
    instance, atfs = example()
    instance.benchmark_name = BenchmarkName.BLAUTH_2024
    atfs.benchmark_name = instance.benchmark_name.value
    loaded = td.LoadedTDInstance(instance, atfs, tmp_path / "TOY1.vrp.json", None)
    candidate = BenchmarkSolution(instance_name=instance.instance_name, routes=[[2], [1]])
    result = td.check_td_solution(loaded, candidate, **options)
    assert result.exact_evaluation["profile"] == td.TD_CHECKER_CONTRACT == TD_PROFILE
    assert result.exact_evaluation["objective"] == "FleetCostDuration"
    assert result.routing_cost == float(Fraction(79, 6))
    assert td.compute_solution_cost(instance, atfs, candidate.routes, **options) == Fraction(79, 6)
    bks = td.create_td_bks_from_solution(loaded, candidate, authors="test", **options)
    assert bks.objective_function == ObjectiveFunction.FLEET_COST_DURATION
    assert td.check_td_solution(loaded, bks, **options).is_valid()
    witness = td.compute_route_duration(instance, atfs, [1], **options)
    assert (witness.duration, witness.departure_time) == (Fraction(14, 3), Fraction(10, 3))
    assert result.route_evaluations[0] == witness
    assert result.model_dump(mode="json")["route_evaluations"][0]["departure_time"] == float(Fraction(10, 3))


def test_exact_point_interpolation_at_a_jump_and_adjacent_rationals():
    from mamut_routing_lib.td import NDCPWLF, PWLFError

    arc = NDCPWLF([0.0, 1.0, 1.0, 2.0], [1.0, 3.0, 4.0, 8.0])
    delta = Fraction(1, 2**60)
    assert arc.evaluate_exact(1 - delta) == 3 - 2 * delta
    assert arc.evaluate_exact(Fraction(1)) == 3
    assert arc.evaluate_exact(1 + delta) == 4 + 4 * delta
    with pytest.raises(PWLFError):
        arc.evaluate_exact(2 + delta)


@pytest.mark.parametrize(("n", "d"), [
    ("2", "4"), ("0", "2"), ("01", "2"), ("1", "02"), ("1", "0"), ("-0", "1"),
    ("-1", "2"), ("1", "-2"), ("+1", "2"), ("1.9", "2"), ("1e1", "2"),
    ("１", "2"), ("1\n", "2"), (1, "2"),
])
def test_cost_readers_reject_noncanonical_result_fields(n, d):
    from mamut_routing_lib.cli import _rational_cost
    from mamut_routing_lib.relaxation import _stored_cost

    record = {"authoritative_cost": {"numerator": n, "denominator": d}}
    with pytest.raises(ValueError, match="canonical result fraction"):
        _rational_cost(record)
    with pytest.raises(ValueError, match="canonical result fraction"):
        _stored_cost({"cost": 0.5, "metadata": {"standard": record}})


def test_result_fraction_readers_preserve_long_coprime_values_and_zero():
    from mamut_routing_lib.cli import _rational_cost
    from mamut_routing_lib.relaxation import _stored_cost

    scale = 10**5000
    for n, d, value in [("1" + "0" * 4999 + "1", "1" + "0" * 5000, Fraction(scale + 1, scale)),
                        ("0", "1", Fraction(0))]:
        record = {"authoritative_cost": {"numerator": n, "denominator": d}}
        assert _rational_cost(record) == value
        assert _stored_cost({"metadata": {"exact_evaluation": record}}) == value

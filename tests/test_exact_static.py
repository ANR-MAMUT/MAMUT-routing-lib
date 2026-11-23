"""Conformance witnesses for the static binary64-exact-v1 backends."""

from fractions import Fraction
import sys

import pytest

from mamut_routing_lib.checker import SolutionCheckStatus
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_static import prepare_static
from mamut_routing_lib.exact_types import ExactRouteError
from mamut_routing_lib.models import BenchmarkInstance, BenchmarkInstanceCVRP


@pytest.fixture(params=["python", "native"])
def backend(request):
    if request.param == "native":
        pytest.importorskip("mamut_routing_lib._static_exact")
    return request.param


def instance(matrix, *, windows=None, services=None, **fields):
    # Construct already-materialized data, also allowing malformed field
    # values to exercise the evaluator's own input validation.
    vertices = len(matrix)
    data = dict(num_customers=vertices-1, depot=0, vehicle_capacity=10,
                num_vehicles=None, demands=[0] * vertices, arc_costs=matrix,
                benchmark_name="Poryos2026")
    data.update(fields)
    if windows is None:
        return BenchmarkInstanceCVRP.model_construct(**data)
    return BenchmarkInstance.model_construct(**data, time_windows=windows, service_times=services)


def test_binary64_operands_and_canonical_routes(backend):
    model = instance([[0, 0.1, 0.2], [1, 0, 0], [0, 0, 0]])
    prepared = prepare_static(model, backend=backend)
    model.arc_costs[0][1] = 99
    model.demands[1] = 11
    result = prepared.evaluate([[2], [1]])
    assert result.cost == 1 + Fraction.from_float(0.1) + Fraction.from_float(0.2)
    assert result.routes == ((1,), (2,))
    assert result.profile == "binary64-exact-v1"
    assert result.backend == {"python": "python-fraction", "native": "cpp-dyadic-prepared"}[backend]
    assert result.objective == ObjectiveFunction.MONO_COST


def test_inclusive_deadline_waiting_and_depot_opening(backend):
    matrix = [[0, 3/8, 0], [0, 0, 5/4], [1/2, 0, 0]]
    model = instance(matrix, windows=[(0, 5/2), (1/2, 8), (0, 2)], services=[0, 1/4, 0])
    result = prepare_static(model, backend=backend, objective_function="HierarchicalVehicleCost").evaluate([[1, 2]])
    assert result.cost == Fraction(17, 8)
    assert result.objective == ObjectiveFunction.HIERARCHICAL_VEHICLE_COST
    model.time_windows[0] = (1, 5/2)
    with pytest.raises(ExactRouteError) as exc:
        prepare_static(model, backend=backend).evaluate([[1, 2]])
    assert exc.value.status == SolutionCheckStatus.TIME_WINDOW_VIOLATED


def test_fractional_excess_over_depot_deadline(backend):
    model = instance([[0, 2**53], [0.5, 0]], windows=[(0, 2**53)] * 2, services=[0, 0])
    with pytest.raises(ExactRouteError) as exc:
        prepare_static(model, backend=backend).evaluate([[1]])
    assert exc.value.status == SolutionCheckStatus.TIME_WINDOW_VIOLATED


@pytest.mark.parametrize("outward,back", [
    (sys.float_info.max, sys.float_info.max),
    (1.0, float.fromhex("0x0.0000000000001p-1022")),
    (2**53+1, -0.0),
])
def test_extreme_finite_operands(backend, outward, back):
    model = instance([[0, outward], [back, 0]], vehicle_capacity=2**100, demands=[0, 2**100])
    assert prepare_static(model, backend=backend).evaluate([[1]]).cost == (
        Fraction.from_float(float(outward)) + Fraction.from_float(float(back))
    )


def test_empty_instance(backend):
    model = instance([[0]], vehicle_capacity=0, num_vehicles=0)
    result = prepare_static(model, backend=backend).evaluate([])
    assert result.cost == 0 and result.routes == ()


@pytest.mark.parametrize("routes", [[[True]], [[1.0]], [[0]], [[2]], [[], [1]], [], [[1], [1]]])
def test_malformed_routes(backend, routes):
    prepared = prepare_static(instance([[0, 0], [0, 0]]), backend=backend)
    with pytest.raises(ExactRouteError):
        prepared.evaluate(routes)


@pytest.mark.parametrize("attributes", [
    {"arc_costs": [[0, float("inf")], [0, 0]]},
    {"arc_costs": [[0, float("nan")], [0, 0]]},
    {"arc_costs": [[0, False], [0, 0]]},
    {"arc_costs": [[0, 10**400], [0, 0]]},
    {"arc_costs": [[0, -1], [0, 0]]},
    {"arc_costs": [[1, 0], [0, 0]]},
    {"arc_costs": [[0], [0, 0]]},
    {"vehicle_capacity": 2**53+1},
    {"demands": [0, -1]},
    {"demands": [1, 0]},
    {"depot": 1},
    {"num_vehicles": -1},
    {"time_windows": [(2, 1), (0, 0)]},
    {"service_times": [1, 0]},
    {"service_times": [0, float("nan")]},
])
def test_reject_malformed_instance(backend, attributes):
    model = instance([[0, 0], [0, 0]], windows=[(0, 0)]*2, services=[0, 0])
    for key, value in attributes.items():
        setattr(model, key, value)
    with pytest.raises(ValueError):
        prepare_static(model, backend=backend)


@pytest.mark.parametrize("field,value", [
    ("num_customers", True), ("num_customers", "1"), ("format_version", True),
    ("off_diagonal", "1.5"), ("off_diagonal", True), ("off_diagonal", float("nan")),
    ("off_diagonal", float("inf")), ("off_diagonal", 10**400),
])
def test_distance_sidecar_rejects_coercion_before_preparation(tmp_path, field, value):
    import json
    from mamut_routing_lib.distances import DistancesFormatError, load_instance_distances

    data = dict(format="mamut-distances", format_version=1, base_name="toy",
                benchmark_name="Poryos2026", metric="fastest", num_customers=1,
                values=[[0, 1.5], [2, 0]])
    if field == "off_diagonal":
        data["values"][0][1] = value
    else:
        data[field] = value
    path = tmp_path / "toy.distances-fastest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(DistancesFormatError):
        load_instance_distances(path)


def test_distance_sidecar_rejects_duplicate_fields(tmp_path):
    from mamut_routing_lib.distances import DistancesFormatError, load_instance_distances
    path = tmp_path / "toy.distances-fastest.json"
    path.write_text('{"format":"mamut-distances","format_version":1,"base_name":"toy",'
                    '"benchmark_name":"Poryos2026","metric":"fastest","num_customers":1,'
                    '"values":[[0,11],[2,0]],"values":[[0,1],[2,0]]}')
    with pytest.raises(DistancesFormatError, match="duplicate|Duplicate"):
        load_instance_distances(path)

"""Production TD exact entry point and independent arithmetic/feasibility checks."""

from fractions import Fraction
import math
import random

import pytest

from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_td import prepare_td
from mamut_routing_lib.exact_types import ExactRouteError, ExactRouteKnot
from mamut_routing_lib.td.models import BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW
from mamut_routing_lib.td.pwlf import NDCPWLF
from td_utils import make_toy_atfs, toy_instance_payload


@pytest.fixture
def backend(request):
    if request.param == 'native':
        pytest.importorskip('mamut_routing_lib._td_exact')
    return request.param


def example():
    instance = BenchmarkInstanceTDVRPTW.model_validate(toy_instance_payload(fleet_fixed_cost=1.25))
    instance.horizon = (0, 15)
    instance.vehicle_capacity = 4
    instance.time_windows = [(0, 15), (6, 7), (7, 9)]
    instance.service_times = [0, 1, 2]
    atfs = make_toy_atfs()
    atfs.horizon = (0, 15)
    atfs.arcs = {arc: NDCPWLF([0, 15], [1, 16]) for arc in atfs.arcs}
    atfs.arcs[0, 1] = NDCPWLF([0, 15], [1, 23.5])
    atfs.arcs[0, 2] = atfs.arcs[2, 0] = NDCPWLF([0, 15], [2, 17])
    return instance, atfs


@pytest.mark.parametrize(('backend', 'identity'), [('native', 'cpp-gmp-compose'), ('python', 'python-fraction-partition')], indirect=['backend'])
def test_worked_example_and_immutable_preparation(backend, identity):
    instance, atfs = example()
    duration = prepare_td(instance, atfs, backend=backend)
    with_charge = prepare_td(instance, atfs, backend=backend, objective_function=ObjectiveFunction.FLEET_COST_DURATION)
    instance.service_times[1] = 99
    atfs.arcs[0, 1].ys[0] = 20
    for check, expected in [(duration, Fraction(32, 3)), (with_charge, Fraction(79, 6))]:
        result = check.evaluate([[2], [1]])
        assert result.routes == ((1,), (2,))
        assert result.cost == expected
        assert result.backend == identity
        assert result.profile == 'td-binary64-exact-v1'
        assert [(r.duration, r.departure_time) for r in result.route_evaluations] == [
            (Fraction(14, 3), Fraction(10, 3)), (6, 5)
        ]
        for bad in ([[1, 2]], [[0, 1], [2]], [[1], [1]], [[1]], [[], [1], [2]]):
            with pytest.raises(ExactRouteError):
                check.evaluate(bad)


@pytest.mark.parametrize('backend', ['python', 'native'], indirect=True)
def test_curve_waiting_window_and_non_dyadic_optimum(backend):
    instance, atfs = example()
    checker = prepare_td(instance, atfs, backend=backend)
    knots = checker.route_curve([1])
    assert knots == (ExactRouteKnot(0, 8, 8), ExactRouteKnot(Fraction(10, 3), 8, 8), ExactRouteKnot(4, 9, 9))
    assert all(isinstance(q, Fraction) for k in knots for q in (k.departure, k.arrival, k.right_arrival))
    witness = checker.route_timing([1])
    assert min((k.arrival - k.departure, k.departure) for k in knots) == (witness.duration, witness.departure_time)


@pytest.mark.parametrize('backend', ['python', 'native'], indirect=True)
def test_curve_preserves_attained_value_at_jump_and_earliest_flat_minimum(backend):
    instance = BenchmarkInstanceTDVRP.model_validate(toy_instance_payload(with_time_windows=False))
    instance.horizon = (0, 15)
    instance.service_times = [0, 0, 0]
    atfs = make_toy_atfs()
    atfs.horizon = instance.horizon
    atfs.arcs = {(0, 1): NDCPWLF([0, 5, 5, 15], [0, 5, 6, 16]), (1, 0): NDCPWLF([0, 15], [0, 15])}
    checker = prepare_td(instance, atfs, backend=backend)
    assert checker.route_curve([1]) == (ExactRouteKnot(0, 0, 0), ExactRouteKnot(5, 5, 6), ExactRouteKnot(14, 15, 15))
    assert checker.route_timing([1]).departure_time == 0
    for invalid in ([], [0]):
        with pytest.raises(ExactRouteError):
            checker.route_curve(invalid)


@pytest.mark.parametrize('backend', ['python', 'native'], indirect=True)
def test_jump_plateau_closed_singleton_and_depot_deadline(backend):
    instance, atfs = example()
    instance.time_windows = [(5, 8), (0, 15), (0, 15)]
    instance.service_times = [0, 0, 0]
    atfs.arcs[0, 1] = NDCPWLF([0, 5, 15], [7, 7, 20])
    atfs.arcs[1, 0] = NDCPWLF([0, 7, 7, 15], [1, 8, 25, 33])
    atfs.arcs[0, 2] = atfs.arcs[2, 0] = NDCPWLF([0, 15], [0, 15])
    checker = prepare_td(instance, atfs, backend=backend)
    result = checker.evaluate([[2], [1]])
    assert (result.route_evaluations[0].duration, result.route_evaluations[0].departure_time) == (3, 5)
    assert checker.route_curve([1]) == (ExactRouteKnot(5, 8, 8),)
    instance.time_windows[0] = (5, 7)
    checker = prepare_td(instance, atfs, backend=backend)
    with pytest.raises(ExactRouteError):
        checker.evaluate([[1], [2]])
    with pytest.raises(ExactRouteError):
        checker.route_curve([1])


@pytest.mark.parametrize('backend', ['python', 'native'], indirect=True)
def test_subnormal_terminal_outside_horizon_and_sparse_inputs(backend):
    tiny = math.ulp(0.0)
    huge = float.fromhex('0x1.fffffffffffffp+1023')
    instance = BenchmarkInstanceTDVRP.model_validate(toy_instance_payload(with_time_windows=False))
    instance.horizon = (tiny, 2 * tiny)
    atfs = make_toy_atfs()
    atfs.horizon = instance.horizon
    atfs.arcs = {arc: NDCPWLF(list(atfs.horizon), list(atfs.horizon)) for arc in [(0, 1), (1, 2)]}
    atfs.arcs[2, 0] = NDCPWLF(list(atfs.horizon), [huge, huge])
    checker = prepare_td(instance, atfs, backend=backend)
    result = checker.evaluate([[1, 2]])
    assert result.cost == Fraction(huge) - 2 * Fraction(tiny)
    assert result.route_evaluations[0].departure_time == 2 * Fraction(tiny)
    with pytest.raises(ValueError, match='not materialized'):
        checker.evaluate([[2, 1]])


def test_bad_instance_or_backend():
    instance, atfs = example()
    with pytest.raises(ValueError, match='unknown TD exact backend'):
        prepare_td(instance, atfs, backend='other')
    with pytest.raises(ValueError):
        prepare_td(instance, atfs, backend='python', objective_function=ObjectiveFunction.MONO_COST)
    instance.fleet_fixed_cost = None
    with pytest.raises(ValueError, match='requires fleet_fixed_cost'):
        prepare_td(instance, atfs, backend='python', objective_function=ObjectiveFunction.FLEET_COST_DURATION)
    for change in ({'service_times': [1, 0, 0]}, {'depot': 1}, {'demands': [0, 1, 2**53+1]},
                   {'time_windows': [(0, 15), (math.nan, 15), (0, 15)]}, {'time_windows': [(0, 15), (9, 8), (0, 15)]}):
        with pytest.raises(ValueError):
            prepare_td(instance.model_copy(update=change), atfs, backend='python')
    atfs.arcs[1, 2].ys[-1] = math.inf  # Invalid even on an arc the submitted solution would not traverse.
    with pytest.raises(ValueError):
        prepare_td(instance, atfs, backend='python')


def _essential_knots(curve):
    """Discard only continuous interior points with equal incoming and outgoing slopes."""
    knots = list(curve[:1])
    for a, b, c in zip(curve, curve[1:], curve[2:]):
        left_slope = (b.arrival - a.right_arrival) / (b.departure - a.departure)
        right_slope = (c.arrival - b.right_arrival) / (c.departure - b.departure)
        if b.arrival != b.right_arrival or left_slope != right_slope:
            knots.append(b)
    if len(curve) > 1:
        knots.append(curve[-1])
    return knots


@pytest.mark.parametrize('seed', range(24))
def test_fraction_and_gmp_agree_on_jumps_and_waiting(seed):
    pytest.importorskip('mamut_routing_lib._td_exact')
    rng = random.Random(seed)
    instance, atfs = example()
    instance.vehicle_capacity = 10
    instance.time_windows = [(0, 15), (rng.randrange(5), 10), (rng.randrange(7), 13)]
    instance.service_times = [0, rng.choice([0, 0.5, 1]), rng.choice([0, 0.25])]
    for key in atfs.arcs:
        xs, ys = [], []
        for x in (0, 1, 4, 8, 15):
            for _ in range(rng.randint(1, 3)):
                xs.append(x)
                ys.append(max(x, ys[-1] if ys else 0) + rng.choice([0, 0, 0.5, 1]))
        atfs.arcs[key] = NDCPWLF(xs, ys)
    reference = prepare_td(instance, atfs, backend='python')
    native = prepare_td(instance, atfs)
    for routes in ([[1, 2]], [[2, 1]], [[1], [2]]):
        try:
            expected = reference.evaluate(routes)
        except ExactRouteError as error:
            with pytest.raises(ExactRouteError) as other:
                native.evaluate(routes)
            assert other.value.status == error.status
        else:
            result = native.evaluate(routes)
            assert result.route_evaluations == expected.route_evaluations
            assert result.cost == expected.cost
            for route, timing in zip(result.routes, result.route_evaluations):
                graph = native.route_curve(route)
                assert _essential_knots(graph) == _essential_knots(reference.route_curve(route))
                assert min((k.arrival - k.departure, k.departure) for k in graph) == (timing.duration, timing.departure_time)


@pytest.mark.parametrize(('location', 'value'), [
    (('format_version',), True), (('num_customers',), '2'), (('num_customers',), 2.0),
    (('horizon', 0), False), (('horizon', 1), math.inf), (('horizon', 1), 0),
    (('arcs', 0, 0), False), (('arcs', 0, 1), 1.75), (('arcs', 0, 1), '1'),
    (('arcs', 0, 2, 0), '0'), (('arcs', 0, 3, 0), math.nan), (('arcs', 0, 3, 0), True),
    (('arcs', 0, 3, 0), 10**400), (('arcs', 0, 2), '01'),
])
def test_direct_sidecar_loader_rejects_invalid_numeric_fields(tmp_path, location, value):
    import json
    from mamut_routing_lib.td.artifacts import ATFFormatError, atfs_to_canonical_json_bytes, load_instance_atfs

    payload = json.loads(atfs_to_canonical_json_bytes(make_toy_atfs()))
    parent = payload
    for item in location[:-1]:
        parent = parent[item]
    parent[location[-1]] = value
    sidecar = tmp_path / 'bad.atf.json'
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ATFFormatError):
        load_instance_atfs(sidecar)

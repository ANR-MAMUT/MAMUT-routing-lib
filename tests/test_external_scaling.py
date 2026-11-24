"""Integer scaling for external solvers: exact when the data allows, conservative otherwise."""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction

import pytest

from mamut_routing_lib.solvers.external.problem import (
    render_explicit_cvrplib,
    render_lkh_params,
    render_lkh_problem,
    render_matrix_format,
    render_x_format,
)
from mamut_routing_lib.solvers.external.scaling import (
    Limits,
    ScaleError,
    agrees_with_checker,
    decimal_places,
    format_float_matrix,
    max_decimals,
    plan_coordinate_scale,
    plan_explicit_scale,
    plan_float_matrix_scale,
    scale_explicit_data,
    star_bound,
    unscale_cost,
)


def test_decimal_places_reads_the_decimal_text() -> None:
    assert decimal_places(7) == 0 and decimal_places(7.0) == 0
    assert decimal_places(0.125) == 3 and decimal_places(1e-05) == 5
    assert decimal_places(265.88719412562915) is None


def test_max_decimals_is_a_round_trip_not_a_tolerance() -> None:
    assert max_decimals([[0.123, 1.5], [2, 3]]) == 3
    assert max_decimals([265.88719412562915]) is None
    assert max_decimals([0.30000000000000004]) is None


def test_the_round_trip_matches_the_checkers_reading_on_published_data() -> None:
    from mamut_routing_lib.checker import shortest_decimal_cost_value

    rng = random.Random(5)
    values = [round(rng.uniform(0, 2e5), 3) for _ in range(2000)]  # Mamut2026 / Poryos2026 style
    assert max_decimals(values) == 3
    assert all((shortest_decimal_cost_value(value) * 1000).denominator == 1 for value in values)
    assert agrees_with_checker([values[:1000], values[1000:]], 3) and not agrees_with_checker(values, 2)


def test_declared_decimals_are_checked_on_every_value() -> None:
    """A matrix declared at 3 decimals with a single 4-decimal value, placed where a sample would not look."""
    size = 100
    matrix = [[0.0 if i == j else round(10.0 + i + j / 8, 3) for j in range(size)] for i in range(size)]
    matrix[0][1] = 1.2345  # 10000 values, so a stride of 2 over the flattened matrix skips index 1
    assert agrees_with_checker(matrix, 3)  # the sample alone does not see it
    plan = plan_explicit_scale(matrix, decimals_hint=3, limits=Limits(max_arc_cost=10**7))
    assert not plan.exact and plan.factor == 10**4  # measured on fallback
    scaled_matrix, _, _ = scale_explicit_data(plan, matrix)
    assert scaled_matrix[0][1] == 12345


def test_scale_uses_shared_shortest_decimal_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter delegates interpretation of each sampled value to the named decimal helper."""
    from mamut_routing_lib import checker

    matrix = [[0, 12.345, 7.1], [12.344, 0, 3.003], [7.1, 3.0, 0]]
    assert plan_explicit_scale(matrix).exact
    # A checker that read floats as their binary values: non-dyadic decimals become full precision.
    monkeypatch.setattr(checker, "shortest_decimal_cost_value", lambda value: Fraction(value))
    assert decimal_places(0.125) == 3 and decimal_places(0.1) is None
    assert max_decimals(matrix) is None
    assert not plan_explicit_scale(matrix, limits=Limits(max_arc_cost=1e6)).exact
    # Declared decimals (collection instances) are held to the same reading.
    assert not plan_explicit_scale(matrix, decimals_hint=3, limits=Limits(max_arc_cost=1e6)).exact
    assert not plan_float_matrix_scale(matrix, decimals_hint=3, limits=Limits(max_arc_cost=5)).exact


def test_collection_style_matrix_uses_binary64_or_a_decimal_search_grid() -> None:
    matrix = [[0, 12.345, 7.1], [12.344, 0, 3.003], [7.1, 3.0, 0]]
    plan = plan_explicit_scale(matrix)
    assert plan.exact and plan.search_model == "binary64-exact-integer"
    exact_matrix, _, _ = scale_explicit_data(plan, matrix)
    assert all(arc * plan.cost_unit == Fraction(v) for row, original in zip(exact_matrix, matrix) for arc, v in zip(row, original))
    raw = exact_matrix[0][2] + exact_matrix[2][1] + exact_matrix[1][0]
    assert unscale_cost(raw, plan)[1] == Fraction(7.1) + Fraction(3) + Fraction(12.344)

    plan = plan_explicit_scale(matrix, limits=Limits(max_arc_cost=10**6))
    assert not plan.exact and plan.factor == 1000 and plan.search_model == "shortest-decimal-grid"
    scaled, windows, services = scale_explicit_data(plan, matrix)
    assert scaled == [[0, 12345, 7100], [12344, 0, 3003], [7100, 3000, 0]]
    assert windows is None and services is None
    assert star_bound(matrix) == pytest.approx(12.345 + 12.344 + 7.1 + 7.1)
    value, exact = unscale_cost(22445, plan)
    assert value == pytest.approx(22.445) and exact is None


def test_limits_force_an_approximate_coarser_scale() -> None:
    matrix = [[0, 12.345], [1.0, 0]]
    plan = plan_explicit_scale(matrix, limits=Limits(max_arc_cost=200))
    assert not plan.exact and plan.factor == 10 and plan.rounding == "nearest"
    assert scale_explicit_data(plan, matrix)[0] == [[0, 123], [10, 0]]
    with pytest.raises(ScaleError, match="exact"):
        plan_explicit_scale(matrix, limits=Limits(max_arc_cost=200), require_exact=True)
    with pytest.raises(ScaleError, match="fits"):
        plan_explicit_scale([[0, 1e9], [1, 0]], limits=Limits(max_arc_cost=1))


def test_largest_common_dyadic_includes_windows_and_services() -> None:
    matrix = [[0, 0.75], [1.5, 0]]
    windows = [(0.0, 9.0), (0.375, 6.0)]
    services = [0, 0.375]
    plan = plan_explicit_scale(matrix, time_windows=windows, service_times=services, require_exact=True)
    assert plan.cost_unit == Fraction(3, 8)
    assert plan.to_json()["cost_unit"] == {"numerator": "3", "denominator": "8"}
    assert scale_explicit_data(plan, matrix, time_windows=windows, service_times=services) == (
        [[0, 2], [4, 0]], [(0, 24), (1, 16)], [0, 1],
    )
    assert unscale_cost(6, plan) == (2.25, Fraction(9, 4))


def test_exact_bounds_include_nonmetric_routes_and_accumulated_time() -> None:
    matrix = [[0, 1, 1], [1, 0, 99], [1, 99, 0]]
    with pytest.raises(ScaleError, match="binary64"):
        plan_explicit_scale(matrix, limits=Limits(max_total_cost=100), require_exact=True)
    plan = plan_explicit_scale(matrix, limits=Limits(max_total_cost=100))
    assert not plan.exact and plan.total_cost_bound_scaled <= 100
    with pytest.raises(ScaleError, match="binary64"):
        plan_explicit_scale([[0, 1], [1, 0]], time_windows=[(0, 10), (0, 10)], service_times=[0, 20],
                            limits=Limits(max_time_value=31), require_exact=True)


def test_no_round_trip_through_float_for_export_or_report() -> None:
    matrix = [[0, 1.0, 0.1], [math.nextafter(1.0, 0.0), 0, 0.25], [1.0, 1.0, 0]]
    windows = [(0.0, 2.0), (0.0, 1.0), (0.0, 1.0)]
    services = [0.0, 0.1, 0.1]
    plan = plan_explicit_scale(matrix, time_windows=windows, service_times=services, require_exact=True)
    scaled, _, _ = scale_explicit_data(plan, matrix, time_windows=windows, service_times=services)
    # Individual binary64 operands often become even large integers. An odd
    # raw route sum above the binary64 consecutive integer range still matters.
    raw = scaled[0][2] + scaled[2][1] + scaled[1][0]
    assert raw > 2**53 and raw % 2 == 1
    assert unscale_cost(raw, plan)[1] == Fraction(0.1) + 1 + Fraction(matrix[1][0])
    # The generic CVRPLIB renderer also accepts exact integer inputs directly.
    rows = [[0, raw], [1, 0]]
    lines = render_explicit_cvrplib(matrix=rows, coordinates=[(0, 0), (1, 0)], demands=[0, 1],
                                   capacity=1, num_vehicles=None, time_windows=[(0, raw), (0, raw)],
                                   service_times=[0, raw]).splitlines()
    assert f"0 {raw}" in lines and f"2 0 {raw}" in lines and f"2 {raw}" in lines


def test_config_limits_keep_their_integer_boundary(tmp_path) -> None:
    from mamut_routing_lib.solvers.external.config import configured_descriptors

    config = tmp_path / "solvers.toml"
    config.write_text('[solvers.lkh-large]\nextends = "lkh3"\n[solvers.lkh-large.limits]\nmax_total_cost = 9007199254740995\n')
    limit = configured_descriptors(config)["lkh-large"].limits.max_total_cost
    assert type(limit) is int and limit == 2**53 + 3


def test_wide_raw_reports_and_strict_json_validation() -> None:
    from decimal import Decimal

    from mamut_routing_lib.solutions import SolutionFormatError, parse_lkh_mtsp_solution, parse_solution_text

    wide = 2**1600 + 1
    for source in (f"Route #1: 1\nCost {wide}.00", f'{{"routes":[[1]],"cost":{wide}.00}}'):
        assert parse_solution_text(source, exact_report=True) == ([[1]], wide)
        with pytest.raises(SolutionFormatError, match="finite binary64"):
            parse_solution_text(source)

    lkh = f"p, Cost: 0_{wide}\n1 2 1 (#1) Cost: {wide}"
    assert parse_lkh_mtsp_solution(lkh, exact_report=True) == ([[2]], 0, wide)
    assert parse_lkh_mtsp_solution(lkh.replace("0_", "1e-1000_"), exact_report=True)[1] != 0
    for source in ('{"routes":[[1]],"routes":[[2]]}', '{"routes":[[1.0]]}', '{"routes":[[]]}',
                   '{"routes":[[1]],"cost":true}', '{"routes":[[1]],"cost":NaN}', "Route #1: 1\nCost bad"):
        with pytest.raises(SolutionFormatError):
            parse_solution_text(source, exact_report=True)
    for token in ("1e1000000000", "1e-1000000000", "3.00000000000000000000001"):
        _, raw = parse_solution_text(f'{{"routes":[[1]],"cost":{token}}}', exact_report=True)
        assert isinstance(raw, Decimal) and raw == Decimal(token)
    unit_plan = plan_explicit_scale([[0, 1], [1, 0]])
    assert unscale_cost(Decimal("1e1000000000"), unit_plan) == (None, None)
    assert unscale_cost(Decimal("1e-1000000000"), unit_plan) == (0.0, None)
    tiny_plan = plan_explicit_scale([[0, math.ulp(0.0)], [math.ulp(0.0), 0]])
    assert unscale_cost(wide, tiny_plan) == (float(Fraction(wide, 2**1074)), Fraction(wide, 2**1074))


def test_subnormal_unit_and_display_overflow() -> None:
    from mamut_routing_lib.solvers.base import SolveMonitor

    tiny = math.ulp(0.0)
    plan = plan_explicit_scale([[0, tiny], [3 * tiny, 0]], require_exact=True)
    assert plan.cost_unit == Fraction(1, 2**1074)
    assert plan.factor == 2**1074
    json.dumps(plan.to_json(), allow_nan=False)
    assert unscale_cost(4, plan) == (4 * tiny, Fraction(4, 2**1074))
    monitor = SolveMonitor(None)
    monitor.rescale(1 / plan.cost_unit)
    monitor.observe(4.0)
    assert monitor.best_cost == 4 * tiny
    huge = math.nextafter(math.inf, 0.0)
    huge_plan = plan_explicit_scale([[0, huge], [huge, 0]], require_exact=True)
    assert unscale_cost(2, huge_plan) == (None, 2 * Fraction(huge))
    assert plan_explicit_scale([[0, 0], [0, 0]], require_exact=True).cost_unit == 1


def test_full_precision_vrptw_rounds_conservatively() -> None:
    matrix = [[0, 1.26, 2.31], [1.26, 0, 1.05], [2.31, 1.05, 0]]
    matrix = [[value + 1e-12 if value else 0.0 for value in row] for row in matrix]  # no exact decimals
    windows = [(0, 100), (2.5, 10.25), (1.0, 4.75)]
    services = [0, 0.5, 1.25]
    plan = plan_explicit_scale(matrix, time_windows=windows, service_times=services, limits=Limits(max_arc_cost=10_000))
    assert not plan.exact and plan.rounding == "conservative"
    scaled, scaled_windows, scaled_services = scale_explicit_data(
        plan, matrix, time_windows=windows, service_times=services
    )
    factor = plan.factor
    assert all(s >= m * factor for row_s, row_m in zip(scaled, matrix) for s, m in zip(row_s, row_m))
    assert all(r * factor <= ready for (r, _), (ready, _) in zip(windows, scaled_windows))
    assert all(d * factor >= due for (_, d), (_, due) in zip(windows, scaled_windows))
    assert all(s * factor <= scaled_s for s, scaled_s in zip(services, scaled_services))


def _feasible(order, matrix, windows, services) -> bool:
    time = 0.0
    previous = 0
    for node in order:
        time = max(time + matrix[previous][node], windows[node][0])
        if time > windows[node][1]:
            return False
        time += services[node]
        previous = node
    return time + matrix[previous][0] <= windows[0][1]


def test_conservative_rounding_never_admits_an_infeasible_schedule() -> None:
    """Property: a route feasible on the scaled integers is feasible on the published data."""
    rng = random.Random(7)
    for _ in range(200):
        n = 5
        points = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n + 1)]
        matrix = [[((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for b in points] for a in points]
        windows = [(0.0, 200.0)] + [
            (lo, lo + rng.uniform(1, 30)) for lo in (rng.uniform(0, 40) for _ in range(n))
        ]
        services = [0.0] + [rng.uniform(0, 3) for _ in range(n)]
        plan = plan_explicit_scale(matrix, time_windows=windows, service_times=services, limits=Limits(max_arc_cost=1e5))
        scaled, scaled_windows, scaled_services = scale_explicit_data(
            plan, matrix, time_windows=windows, service_times=services
        )
        order = list(range(1, n + 1))
        rng.shuffle(order)
        if _feasible(order, scaled, scaled_windows, scaled_services):
            assert _feasible(order, matrix, windows, services)


def test_coordinate_scale_picks_the_first_unit_that_fits() -> None:
    hgs = Limits(max_arc_cost=1e5, min_arc_cost=0.1)
    city = [(0.0, 0.0), (60_000.0, 0.0), (0.0, 40_000.0)]  # 72 km diagonal: metres fit
    big = [(0.0, 0.0), (100_000.0, 0.0), (0.0, 60_000.0)]  # 117 km: decametres
    assert plan_coordinate_scale(city, candidates=(1, 0.1, 0.01), limits=hgs).unit == "m"
    plan = plan_coordinate_scale(big, candidates=(1, 0.1, 0.01), limits=hgs)
    assert plan.unit == "dam" and plan.coordinate_decimals == 7 and not plan.exact
    filo = plan_coordinate_scale(city, candidates=(1000,), distance_model="nint", published_decimals=3)
    assert filo.unit == "mm" and filo.coordinate_decimals == 3
    assert filo.max_arc_error == pytest.approx(0.0005 + 0.0005)
    with pytest.raises(ScaleError):
        plan_coordinate_scale(big, candidates=(1,), limits=hgs)


def test_x_format_header_is_the_exact_filo2_sequence() -> None:
    lines = list(render_x_format(coordinates=[(0, 0), (1.5, 2)], demands=[0, 3], capacity=10, coordinate_decimals=3))
    assert lines[:7] == [
        "NAME : problem",
        "COMMENT : MAMUT-routing",
        "TYPE : CVRP",
        "DIMENSION : 2",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "CAPACITY : 10",
        "NODE_COORD_SECTION",
    ]
    assert lines[7:9] == ["1 0.000 0.000", "2 1.500 2.000"]
    assert lines[-4:] == ["DEPOT_SECTION", "1", "-1", "EOF"]


def test_lkh_problem_has_no_coordinates_and_vrptw_sections() -> None:
    lines = list(
        render_lkh_problem(
            problem_type="CVRPTW",
            matrix=[[0, 5], [6, 0]],
            demands=[0, 1],
            capacity=5,
            vehicles=2,
            time_windows=[(0, 100), (3, 9)],
            service_times=[0, 2],
        )
    )
    assert "NODE_COORD_SECTION" not in lines
    assert "VEHICLES : 2" in lines and "TYPE : CVRPTW" in lines
    assert lines[lines.index("EDGE_WEIGHT_SECTION") + 1] == "0 5"
    assert lines[lines.index("TIME_WINDOW_SECTION") + 2] == "2 3 9"
    with pytest.raises(ValueError):
        list(render_lkh_problem(problem_type="CVRP", matrix=[[0]], demands=[0], capacity=1, vehicles=1, time_windows=[(0, 1)]))
    params = list(render_lkh_params(problem="/w/problem.vrp", solution="/w/solution.sol", seed=3, time_limit_s=5))
    assert params[0] == "SPECIAL" and "MTSP_MIN_SIZE = 0" in params and "TIME_LIMIT = 5.0" in params


HGS_LIMITS = Limits(max_arc_cost=1e5, min_arc_cost=0.1, max_demand=1e5)


def test_float_matrix_goes_in_unchanged_when_it_fits() -> None:
    matrix = [[0, 120.5, 240.25], [121.5, 0, 130.75], [240.25, 130.0, 0]]  # asymmetric seconds
    plan = plan_float_matrix_scale(matrix, limits=HGS_LIMITS)
    assert plan.factor == 1.0 and plan.exact and plan.rounding == "float"
    assert list(format_float_matrix(matrix, plan)) == ["0 120.5 240.25", "121.5 0 130.75", "240.25 130 0"]


def test_float_matrix_writes_shifted_decimals_when_too_wide() -> None:
    matrix = [[0, 150_000.125], [200_000.0, 0]]  # metres across a 200 km city
    plan = plan_float_matrix_scale(matrix, limits=HGS_LIMITS)
    assert plan.factor == 0.1 and not plan.exact
    assert plan.family_units_per_solver_unit == 10
    assert list(format_float_matrix(matrix, plan)) == ["0.0 15000.0125", "20000.0 0.0"]
    with pytest.raises(ScaleError):
        plan_float_matrix_scale(matrix, limits=HGS_LIMITS, candidates=(1.0,))


def test_full_precision_doubles_are_exact_only_unscaled() -> None:
    sintef = [[0, 265.88719412562915], [265.88719412562915, 0]]
    assert plan_float_matrix_scale(sintef, limits=HGS_LIMITS).exact
    assert list(format_float_matrix(sintef, plan_float_matrix_scale(sintef)))[0] == "0 265.88719412562915"
    assert not plan_float_matrix_scale(sintef, limits=Limits(max_arc_cost=100)).exact


def test_matrix_format_is_what_the_patched_hgs_reader_expects() -> None:
    lines = list(
        render_matrix_format(
            matrix_rows=["0 5.5", "6 0"], dimension=2, coordinates=[(0, 0), (1.5, 2)], demands=[0, 3], capacity=10
        )
    )
    assert lines[:3] == ["NAME : problem", "COMMENT : MAMUT-routing", "TYPE : CVRP"]  # skipped by the reader
    assert lines[3:8] == [
        "DIMENSION : 2",  # before the matrix: the reader sizes it from DIMENSION
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "CAPACITY : 10",
        "EDGE_WEIGHT_SECTION",
    ]
    assert lines[8:11] == ["0 5.5", "6 0", "NODE_COORD_SECTION"]
    assert not any(line.startswith("VEHICLES") for line in lines)
    assert lines[-4:] == ["DEPOT_SECTION", "1", "-1", "EOF"]


def test_the_hgs_reader_patch_ships_with_the_lib() -> None:
    from pathlib import Path

    from mamut_routing_lib.solvers.external.catalog import patch_path

    text = Path(patch_path("hgs-cvrp-explicit-matrix.patch")).read_text()
    assert "--- a/Program/InstanceCVRPLIB.cpp" in text and "+++ b/Program/InstanceCVRPLIB.cpp" in text
    assert "EDGE_WEIGHT_SECTION" in text and "1a92795" in text

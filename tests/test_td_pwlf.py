from __future__ import annotations

import random

import pytest

from mamut_routing_lib.td import (
    NDCPWLF,
    PWLFError,
    apply_ready_time,
    make_service_theta,
    make_theta,
    restrict_domain,
)


class TestNDCPWLFBasics:
    def test_invariant_violations_raise(self):
        with pytest.raises(PWLFError):
            NDCPWLF([0.0, 1.0], [0.0])
        with pytest.raises(PWLFError):
            NDCPWLF([1.0, 0.0], [0.0, 1.0])
        with pytest.raises(PWLFError):
            NDCPWLF([0.0, 1.0], [1.0, 0.0])

    def test_identity(self):
        f = NDCPWLF.identity(2.0, 5.0)
        assert f.evaluate(2.0) == 2.0
        assert f.evaluate(3.5) == 3.5
        assert f.evaluate(5.0) == 5.0

    def test_evaluate_interpolates(self):
        f = NDCPWLF([0.0, 10.0], [0.0, 20.0])
        assert f.evaluate(5.0) == 10.0

    def test_evaluate_outside_domain_raises(self):
        f = NDCPWLF([0.0, 10.0], [0.0, 20.0])
        with pytest.raises(PWLFError):
            f.evaluate(-0.1)
        with pytest.raises(PWLFError):
            f.evaluate(10.1)

    def test_evaluate_at_step_returns_smallest_value(self):
        f = NDCPWLF([0.0, 5.0, 5.0, 10.0], [0.0, 5.0, 8.0, 13.0])
        assert f.evaluate(5.0) == 5.0

    def test_evaluate_on_plateau(self):
        f = NDCPWLF([0.0, 5.0, 10.0], [3.0, 3.0, 8.0])
        assert f.evaluate(2.0) == 3.0
        assert f.evaluate(5.0) == 3.0


class TestCompose:
    def test_compose_identity_is_noop(self):
        f = NDCPWLF([0.0, 50.0, 100.0], [30.0, 60.0, 110.0])
        identity = NDCPWLF.identity(0.0, 100.0)
        assert f.compose(identity) == f
        left = identity  # img(f) = [30, 110] extends beyond dom(identity)
        h = left.compose(f)
        # restriction: only departures with f(t) <= 100 survive
        assert h.xs[0] == 0.0
        assert h.ys == [f.evaluate(x) for x in h.xs]
        assert h.max_image == 100.0

    def test_compose_disjoint_domains_is_empty(self):
        f = NDCPWLF([200.0, 300.0], [200.0, 300.0])
        g = NDCPWLF([0.0, 10.0], [0.0, 20.0])
        assert f.compose(g).is_empty()

    def test_compose_single_point_overlap(self):
        f = NDCPWLF([20.0, 30.0], [40.0, 50.0])
        g = NDCPWLF([0.0, 10.0], [10.0, 20.0])
        h = f.compose(g)
        assert h.xs == [10.0]
        assert h.ys == [40.0]

    def test_compose_exact_values_simple(self):
        # f(x) = x + 10 over [0, 100]; g(x) = 2x over [0, 40]
        f = NDCPWLF([0.0, 100.0], [10.0, 110.0])
        g = NDCPWLF([0.0, 40.0], [0.0, 80.0])
        h = f.compose(g)
        assert h.evaluate(0.0) == 10.0
        assert h.evaluate(20.0) == 50.0
        assert h.evaluate(40.0) == 90.0

    def test_compose_plateau_in_g_gives_plateau(self):
        f = NDCPWLF([0.0, 100.0], [0.0, 100.0])
        g = NDCPWLF([0.0, 10.0, 20.0, 30.0], [5.0, 5.0, 5.0, 15.0])
        h = f.compose(g)
        assert h.evaluate(0.0) == 5.0
        assert h.evaluate(15.0) == 5.0
        assert h.evaluate(30.0) == 15.0

    def test_compose_random_matches_pointwise(self):
        rng = random.Random(42)
        for _ in range(50):
            f = _random_ndcpwlf(rng, rng.randint(2, 12))
            g = _random_ndcpwlf(rng, rng.randint(2, 12))
            h = f.compose(g)
            if h.is_empty():
                continue
            # invariants hold exactly
            assert all(h.xs[k] <= h.xs[k + 1] for k in range(len(h.xs) - 1))
            assert all(h.ys[k] <= h.ys[k + 1] for k in range(len(h.ys) - 1))
            # h agrees with f o g pointwise
            samples = list(h.xs) + [
                rng.uniform(h.min_domain, h.max_domain) for _ in range(20)
            ]
            for x in samples:
                gx = g.evaluate(x)
                if not (f.min_domain <= gx <= f.max_domain):
                    continue  # boundary rounding: g(x) fell epsilon-outside dom(f)
                assert h.evaluate(x) == pytest.approx(f.evaluate(gx), rel=1e-9, abs=1e-9)

    def test_compose_chain_duration_scenario(self):
        # Toy arc (1,2) ATF composed after constant arc (0,1) ATF, see td_utils.
        alpha_01 = NDCPWLF([0.0, 100.0], [10.0, 110.0])
        alpha_12 = NDCPWLF([0.0, 50.0, 100.0], [30.0, 60.0, 110.0])
        alpha_20 = NDCPWLF([0.0, 100.0], [10.0, 110.0])
        acc = alpha_01.compose(NDCPWLF.identity(0.0, 100.0))
        acc = alpha_12.compose(acc)
        acc = alpha_20.compose(acc)
        best = min(y - x for x, y in zip(acc.xs, acc.ys))
        assert best == 30.0


class TestMinShiftedImage:
    def test_earliest_argmin_is_returned(self):
        f = NDCPWLF([0.0, 40.0, 70.0], [46.0, 70.0, 100.0])
        duration, departure = f.min_shifted_image()
        assert duration == 30.0
        assert departure == 40.0

    def test_empty_raises(self):
        with pytest.raises(PWLFError):
            NDCPWLF.empty().min_shifted_image()


class TestMakeTheta:
    def test_no_wait_no_service_is_identity_like(self):
        theta = make_theta(0.0, 50.0, 0.0)
        assert theta.xs == [0.0, 50.0]
        assert theta.ys == [0.0, 50.0]

    def test_waiting_plateau(self):
        theta = make_theta(20.0, 50.0, 5.0)
        assert theta.evaluate(0.0) == 25.0
        assert theta.evaluate(20.0) == 25.0
        assert theta.evaluate(50.0) == 55.0

    def test_point_time_window(self):
        theta = make_theta(30.0, 30.0, 2.0)
        assert theta.evaluate(0.0) == 32.0
        assert theta.evaluate(30.0) == 32.0
        assert theta.max_domain == 30.0

    def test_invalid_window_raises(self):
        with pytest.raises(PWLFError):
            make_theta(10.0, 5.0, 0.0)


class TestMakeServiceTheta:
    def test_zero_upper_is_a_single_point(self):
        theta = make_service_theta(0.0, 7.0)
        assert theta.xs == [0.0]
        assert theta.ys == [7.0]

    def test_positive_upper_has_two_points(self):
        theta = make_service_theta(50.0, 7.0)
        assert theta.xs == [0.0, 50.0]
        assert theta.ys == [7.0, 57.0]

    def test_single_point_theta_composes_with_a_zero_accumulator(self):
        acc = NDCPWLF([0.0, 10.0], [0.0, 0.0])
        composed = make_service_theta(acc.max_image, 7.0).compose(acc)
        assert composed.evaluate(0.0) == 7.0
        assert composed.evaluate(10.0) == 7.0


class TestSlopeOneRule:
    def test_default_evaluate_keeps_the_ratio_formula(self):
        # The ATF materializers (and their sha256 pins) depend on this path.
        f = NDCPWLF([0.0, 49.0], [0.0, 49.0])
        assert f.evaluate(1.0) == 0.9999999999999999

    def test_slope_one_evaluate_is_exact(self):
        f = NDCPWLF([0.0, 22.0], [0.0, 22.0])
        assert f.evaluate(15.0) == 14.999999999999998
        assert f.evaluate(15.0, slope_one_exact=True) == 15.0

    def test_other_slopes_are_unaffected(self):
        f = NDCPWLF([0.0, 49.0], [0.0, 98.0])
        assert f.evaluate(1.0, slope_one_exact=True) == f.evaluate(1.0)

    def test_slope_one_compose_is_exact_both_ways(self):
        arc = NDCPWLF([0.0, 22.0], [100.0, 122.0])
        acc = NDCPWLF([0.0, 49.0], [0.0, 49.0])
        composed = arc.compose(acc, slope_one_exact=True)
        assert composed.xs == [0.0, 22.0]
        assert composed.ys == [100.0, 122.0]


class TestRestrictDomain:
    def test_breakpoints_inside_the_window_are_verbatim(self):
        f = NDCPWLF([0.0, 10.0, 20.0, 30.0], [5.0, 15.0, 40.0, 50.0])
        restricted = restrict_domain(f, 5.0, 25.0)
        assert restricted.xs == [5.0, 10.0, 20.0, 25.0]
        assert restricted.ys == [10.0, 15.0, 40.0, 45.0]

    def test_whole_domain_is_unchanged(self):
        f = NDCPWLF([0.0, 10.0, 10.0, 30.0], [5.0, 15.0, 18.0, 50.0])
        assert restrict_domain(f, 0.0, 30.0) == f
        assert restrict_domain(f, -5.0, 99.0) == f

    def test_steps_at_the_window_ends_keep_both_points(self):
        f = NDCPWLF([0.0, 10.0, 10.0, 20.0, 20.0, 30.0], [1.0, 11.0, 15.0, 25.0, 29.0, 39.0])
        restricted = restrict_domain(f, 10.0, 20.0)
        assert restricted.xs == [10.0, 10.0, 20.0, 20.0]
        assert restricted.ys == [11.0, 15.0, 25.0, 29.0]

    def test_single_departure_time(self):
        f = NDCPWLF([0.0, 10.0], [5.0, 15.0])
        restricted = restrict_domain(f, 4.0, 4.0)
        assert restricted.xs == [4.0]
        assert restricted.ys == [9.0]

    def test_disjoint_window_is_empty(self):
        f = NDCPWLF([0.0, 10.0], [5.0, 15.0])
        assert restrict_domain(f, 11.0, 20.0).is_empty()

    def test_replaces_identity_composition(self):
        # Same function as f ∘ identity; the composition only rounds the
        # abscissas it re-derives through the identity (by ulps), which the
        # verbatim restriction avoids.
        rng = random.Random(3)
        for _ in range(50):
            f = _random_ndcpwlf(rng, 8)
            high = rng.uniform(f.xs[0], f.xs[-1])
            restricted = restrict_domain(f, 0.0, high, slope_one_exact=False)
            composed = f.compose(NDCPWLF.identity(0.0, high))
            assert restricted.ys == composed.ys
            assert restricted.xs == pytest.approx(composed.xs, rel=1e-15, abs=1e-15)
            assert all(x in f.xs or x == high for x in restricted.xs)


class TestApplyReadyTime:
    def test_td_fold_v1_rounding_canaries_are_exact(self):
        # Measured on the pre-0.12 fold: θ composed like an arc rounds these.
        acc = NDCPWLF([0.0, 1.0], [13.0, 14.0])
        assert make_service_theta(43200.0, 0.0).compose(acc).ys[0] == 13.000000000000002
        assert apply_ready_time(acc, earliest=None, latest=None, service_time=0.0).ys[0] == 13.0
        acc = NDCPWLF([0.0, 1.0], [1009.0, 1010.0])
        assert make_theta(250.0, 12360.0, 5.0).compose(acc).ys[0] == 1014.0000000000001
        assert apply_ready_time(acc, earliest=250.0, latest=12360.0, service_time=5.0).ys[0] == 1014.0
        acc = NDCPWLF([0.0, 1.0], [11.0, 12.0])
        assert make_theta(0.0, 43200.0, 5.0).compose(acc).ys[0] == 15.999999999999998
        assert apply_ready_time(acc, earliest=0.0, latest=43200.0, service_time=5.0).ys[0] == 16.0

    def test_breakpoints_map_to_max_plus_service(self):
        acc = NDCPWLF([0.0, 10.0, 20.0], [3.0, 7.0, 30.0])
        result = apply_ready_time(acc, earliest=None, latest=None, service_time=2.0)
        assert result.xs == [0.0, 10.0, 20.0]
        assert result.ys == [5.0, 9.0, 32.0]

    def test_crossing_earliest_inserts_the_plateau_corner(self):
        acc = NDCPWLF([0.0, 10.0], [0.0, 10.0])
        result = apply_ready_time(acc, earliest=4.0, latest=None, service_time=1.0)
        assert result.xs == [0.0, 4.0, 10.0]
        assert result.ys == [5.0, 5.0, 11.0]

    def test_cut_at_latest_inserts_the_crossing(self):
        acc = NDCPWLF([0.0, 10.0], [0.0, 10.0])
        result = apply_ready_time(acc, earliest=None, latest=6.0, service_time=1.0)
        assert result.xs == [0.0, 6.0]
        assert result.ys == [1.0, 7.0]

    def test_one_piece_crossing_both_bounds_emits_earliest_first(self):
        acc = NDCPWLF([0.0, 10.0], [0.0, 10.0])
        result = apply_ready_time(acc, earliest=3.0, latest=8.0, service_time=0.5)
        assert result.xs == [0.0, 3.0, 8.0]
        assert result.ys == [3.5, 3.5, 8.5]

    def test_vertical_piece_crossing_latest_cuts_at_the_step(self):
        acc = NDCPWLF([0.0, 5.0, 5.0, 10.0], [1.0, 6.0, 20.0, 25.0])
        result = apply_ready_time(acc, earliest=None, latest=9.0, service_time=0.0)
        assert result.xs == [0.0, 5.0, 5.0]
        assert result.ys == [1.0, 6.0, 9.0]

    def test_plateau_at_latest_is_kept(self):
        acc = NDCPWLF([0.0, 4.0, 8.0, 10.0], [2.0, 6.0, 6.0, 9.0])
        result = apply_ready_time(acc, earliest=None, latest=6.0, service_time=0.0)
        assert result.xs == [0.0, 4.0, 8.0]
        assert result.ys == [2.0, 6.0, 6.0]

    def test_first_value_past_latest_is_infeasible(self):
        acc = NDCPWLF([0.0, 10.0], [7.0, 17.0])
        assert apply_ready_time(acc, earliest=0.0, latest=6.0, service_time=0.0).is_empty()

    def test_invalid_window_raises(self):
        acc = NDCPWLF([0.0, 10.0], [0.0, 10.0])
        with pytest.raises(PWLFError):
            apply_ready_time(acc, earliest=6.0, latest=5.0, service_time=0.0)

    def test_point_window(self):
        acc = NDCPWLF([0.0, 10.0], [0.0, 10.0])
        result = apply_ready_time(acc, earliest=5.0, latest=5.0, service_time=2.0)
        assert result.xs == [0.0, 5.0]
        assert result.ys == [7.0, 7.0]

    def test_matches_theta_composition_where_both_are_exact(self):
        # Integer data with power-of-two window widths: the two forms agree.
        rng = random.Random(11)
        for _ in range(200):
            xs = sorted(rng.sample(range(0, 64), 6))
            ys = sorted(rng.sample(range(0, 64), 6))
            acc = NDCPWLF([float(v) for v in xs], [float(v) for v in ys])
            earliest = float(rng.randint(0, 32))
            latest = earliest + 32.0
            service = float(rng.randint(0, 8))
            expected = make_theta(earliest, latest, service).compose(acc)
            result = apply_ready_time(acc, earliest=earliest, latest=latest, service_time=service)
            assert result == expected


def _random_ndcpwlf(rng: random.Random, num_points: int) -> NDCPWLF:
    xs = [rng.uniform(0.0, 5.0)]
    ys = [rng.uniform(0.0, 5.0)]
    for _ in range(num_points - 1):
        xs.append(xs[-1] + rng.uniform(0.01, 3.0))
        ys.append(ys[-1] + rng.uniform(0.0, 3.0))
    return NDCPWLF(xs, ys)

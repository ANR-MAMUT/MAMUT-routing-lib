from __future__ import annotations

import random

import pytest

from mamut_routing_lib.td import NDCPWLF, PWLFError, make_service_theta, make_theta


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


def _random_ndcpwlf(rng: random.Random, num_points: int) -> NDCPWLF:
    xs = [rng.uniform(0.0, 5.0)]
    ys = [rng.uniform(0.0, 5.0)]
    for _ in range(num_points - 1):
        xs.append(xs[-1] + rng.uniform(0.01, 3.0))
        ys.append(ys[-1] + rng.uniform(0.0, 3.0))
    return NDCPWLF(xs, ys)

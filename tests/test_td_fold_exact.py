"""Property test of the ``td-fold/2`` route fold against an exact rational reference.

Arcs are random integer step ATFs whose pieces have slope 0, slope 1 or are
vertical steps (the structure of Rifki2020's canonical ATFs); time windows and
service times are integers. On such data the contract promises an exact fold:
the checker's float result must equal, bit for bit, the reference fold run in
``Fraction`` arithmetic, and every duration and departure is an integer. The
pre-0.12 fold (``td-fold/1``, θ composed like an arc) is kept here as a canary:
it must disagree with the exact reference somewhere, otherwise this test would
not be exercising the rounding the contract removed.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mamut_routing_lib.td.pwlf import NDCPWLF, apply_ready_time, restrict_domain

HORIZON_END = 2000
NUM_ROUTES = 1500
SEED = 20260924


def _random_step_atf(rng: random.Random) -> tuple[list[int], list[int]]:
    """Integer ATF over ``[0, HORIZON_END]`` with slope-0, slope-1 and vertical pieces (FIFO, travel >= 1)."""
    x, y = 0, rng.randint(1, 60)
    xs, ys = [x], [y]
    while x < HORIZON_END:
        kind = rng.random()
        if kind < 0.2:
            y += rng.randint(1, 40)
            xs.append(x)
            ys.append(y)
            continue
        width = min(rng.randint(1, 120), HORIZON_END - x)
        if kind < 0.45 and y - x - 1 >= 1:
            x += min(width, y - x - 1)
        else:
            x += width
            y += width
        xs.append(x)
        ys.append(y)
    return xs, ys


def _vertex_map(conv, earliest, latest, service, acc_max):
    """θ as an explicit function, built from ``conv`` so the reference stays exact."""
    zero = conv(0)
    if earliest is not None:
        points = [(zero, earliest + service), (earliest, earliest + service), (latest, latest + service)]
    else:
        points = [(zero, service), (acc_max, acc_max + service)]
    xs: list = []
    ys: list = []
    for x, y in points:
        if xs and x == xs[-1] and y == ys[-1]:
            continue
        xs.append(x)
        ys.append(y)
    return NDCPWLF(xs, ys)


def _fold_v1(arcs, windows, services, depot_due, conv):
    """The pre-0.12 fold (every step composed), in the arithmetic given by ``conv``."""
    acc = NDCPWLF.identity(conv(0), conv(depot_due if depot_due is not None else HORIZON_END))
    for k, (xs, ys) in enumerate(arcs[:-1]):
        acc = NDCPWLF([conv(v) for v in xs], [conv(v) for v in ys]).compose(acc)
        if acc.is_empty():
            return None
        service = conv(services[k])
        if windows:
            theta = _vertex_map(conv, conv(windows[k][0]), conv(windows[k][1]), service, None)
        else:
            theta = _vertex_map(conv, None, None, service, acc.max_image)
        acc = theta.compose(acc)
        if acc.is_empty():
            return None
    xs, ys = arcs[-1]
    acc = NDCPWLF([conv(v) for v in xs], [conv(v) for v in ys]).compose(acc)
    if acc.is_empty():
        return None
    if depot_due is not None:
        acc = NDCPWLF.identity(conv(0), conv(depot_due)).compose(acc)
        if acc.is_empty():
            return None
    return acc.min_shifted_image()


def _fold_v2(arcs, windows, services, depot_due):
    """The checker fold (``td-fold/2``) in floats, as ``compute_route_ready_time_function`` runs it."""
    acc = None
    for k, (xs, ys) in enumerate(arcs[:-1]):
        f = NDCPWLF([float(v) for v in xs], [float(v) for v in ys])
        if acc is None:
            acc = restrict_domain(f, 0.0, float(depot_due if depot_due is not None else HORIZON_END))
        else:
            acc = f.compose(acc, slope_one_exact=True)
        if acc.is_empty():
            return None
        earliest, latest = (float(windows[k][0]), float(windows[k][1])) if windows else (None, None)
        acc = apply_ready_time(acc, earliest=earliest, latest=latest, service_time=float(services[k]))
        if acc.is_empty():
            return None
    f = NDCPWLF([float(v) for v in arcs[-1][0]], [float(v) for v in arcs[-1][1]])
    acc = f.compose(acc, slope_one_exact=True)
    if acc.is_empty():
        return None
    if depot_due is not None:
        acc = apply_ready_time(acc, earliest=None, latest=float(depot_due), service_time=0.0)
        if acc.is_empty():
            return None
    return acc.min_shifted_image()


def _random_routes():
    rng = random.Random(SEED)
    for _ in range(NUM_ROUTES):
        length = rng.randint(1, 4)
        arcs = [_random_step_atf(rng) for _ in range(length + 1)]
        windows: list[tuple[int, int]] = []
        with_windows = rng.random() < 0.5
        if with_windows:
            for _ in range(length):
                earliest = rng.randint(0, HORIZON_END // 2)
                windows.append((earliest, rng.randint(earliest, HORIZON_END + 200)))
        services = [rng.randint(0, 30) for _ in range(length)]
        depot_due = rng.randint(HORIZON_END // 2, HORIZON_END + 300) if with_windows else None
        yield arcs, windows, services, depot_due


def _same(result, reference) -> bool:
    if result is None or reference is None:
        return result is None and reference is None
    return Fraction(result[0]) == reference[0] and Fraction(result[1]) == reference[1]


def test_fold_v2_is_exact_on_integer_step_data():
    feasible = 0
    old_fold_disagreements = 0
    for arcs, windows, services, depot_due in _random_routes():
        reference = _fold_v1(arcs, windows, services, depot_due, Fraction)
        result = _fold_v2(arcs, windows, services, depot_due)
        assert _same(result, reference), (arcs, windows, services, depot_due, result, reference)
        if result is not None:
            feasible += 1
            duration, departure = result
            assert duration == int(duration) and departure == int(departure)
        if not _same(_fold_v1(arcs, windows, services, depot_due, float), reference):
            old_fold_disagreements += 1
    # The sample must be meaningful: mostly feasible routes, and the rounding
    # removed by td-fold/2 must actually occur under td-fold/1.
    assert feasible > NUM_ROUTES // 2
    assert old_fold_disagreements > 0

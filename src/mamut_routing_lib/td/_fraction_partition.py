"""Exact departure optimization by partitioning the initial departure interval."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction

from mamut_routing_lib.exact_types import ExactRouteKnot


@dataclass(frozen=True, slots=True)
class Stage:
    xs: tuple[Fraction, ...] = ()
    ys: tuple[Fraction, ...] = ()
    earliest: Fraction | None = None
    latest: Fraction | None = None
    service: Fraction = Fraction(0)

    def boundaries(self):
        if self.xs:
            return self.xs
        return tuple(value for value in (self.earliest, self.latest) if value is not None)

    def evaluate(self, time: Fraction) -> Fraction | None:
        if self.xs:
            if time < self.xs[0] or time > self.xs[-1]:
                return None
            right = bisect_left(self.xs, time)
            if self.xs[right] == time:
                return self.ys[right]  # The first ordinate is attained at a jump.
            left = right - 1
            weight = (time - self.xs[left]) / (self.xs[right] - self.xs[left])
            return self.ys[left] + weight * (self.ys[right] - self.ys[left])
        if self.latest is not None and time > self.latest:
            return None
        if self.earliest is not None:
            time = max(self.earliest, time)
        return time + self.service


def _forward(stages: list[Stage], departure: Fraction) -> Fraction | None:
    time = departure
    for stage in stages:
        time = stage.evaluate(time)
        if time is None:
            return None
    return time


def _partition(arcs, windows, services, low, high, depot_close) -> tuple[list[Stage], set[Fraction]]:
    if high < low:
        return [], set()
    stages = []
    for index, (xs, ys) in enumerate(arcs):
        stages.append(Stage(tuple(map(Fraction, xs)), tuple(map(Fraction, ys))))
        if index < len(services):
            window = windows[index]
            opens, closes = (None, None) if window is None else map(Fraction, window)
            stages.append(Stage(earliest=opens, latest=closes, service=Fraction(services[index])))
    if depot_close is not None:
        stages.append(Stage(latest=Fraction(depot_close)))

    # Partition the departure axis at each preimage of a new stage boundary.
    # On any open interval of the old partition the prefix is affine or infeasible;
    # its values at two interior points therefore determine every new cut.
    cuts = {Fraction(low), Fraction(high)}
    prefix = []
    for stage in stages:
        ordered = sorted(cuts)
        for left, right in zip(ordered, ordered[1:]):
            p, q = (2 * left + right) / 3, (left + 2 * right) / 3
            u, v = _forward(prefix, p), _forward(prefix, q)
            if u is None or v is None or u == v:
                continue
            slope = (v - u) / (q - p)
            for boundary in stage.boundaries():
                cut = p + (boundary - u) / slope
                if left < cut < right:
                    cuts.add(cut)
        prefix.append(stage)
    return stages, cuts


def solve(arcs, windows, services, low, high, depot_close) -> tuple[Fraction, Fraction] | None:
    """Return the minimum duration and its earliest attaining departure, or None."""
    stages, cuts = _partition(arcs, windows, services, low, high, depot_close)
    # A nonconstant affine duration has no interior minimum. The attained
    # lower value at each jump ensures a flat optimal interval includes its
    # left endpoint. Compare (duration, departure) to resolve all ties.
    candidates = ((arrival - departure, departure) for departure in cuts
                  if (arrival := _forward(stages, departure)) is not None)
    return min(candidates, default=None)


def curve(arcs, windows, services, low, high, depot_close) -> tuple[ExactRouteKnot, ...] | None:
    """Recover the complete feasible graph of the final arrival from its departure partition."""
    stages, cuts = _partition(arcs, windows, services, low, high, depot_close)
    points = [(x, y) for x in sorted(cuts) if (y := _forward(stages, x)) is not None]
    if not points:
        return None
    knots = []
    for (left, arrival), (right, next_arrival) in zip(points, points[1:]):
        # The open segment is affine and its value at the upper endpoint is
        # attained by left continuity. Extrapolate from the exact midpoint.
        mid_arrival = _forward(stages, (left + right) / 2)
        assert mid_arrival is not None  # FIFO and upper deadlines give an interval of feasible departures.
        knots.append(ExactRouteKnot(left, arrival, 2 * mid_arrival - next_arrival))
    last, arrival = points[-1]
    knots.append(ExactRouteKnot(last, arrival, arrival))
    return tuple(knots)

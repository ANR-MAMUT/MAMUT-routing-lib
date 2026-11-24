"""Integer scaling of instance data for external solvers.

For an explicit integer solver, express all materialized binary64 arc costs,
time-window endpoints and service times in their largest common dyadic unit.
Every emitted integer multiplied by this unit recovers its input exactly.
If the integers exceed the solver's bounds, use its decimal-grid search
model, or raise ScaleError when exact input was requested. The fallback
reads the operands' shortest decimal spellings and rounds arcs to nearest
for CVRP. For VRPTW it rounds travel, service and openings up and closings
down on this decimal reading. The benchmark checker prices returned routes
independently on the original binary64 data.

Coordinate solvers select from their supported coordinate scales. Floating
matrix solvers select a scale that their reader supports and accumulate in
floating point. Provenance identifies the search model used in each case.
"""

from __future__ import annotations

import itertools
import math
import numbers
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Iterable, Literal, Sequence

from mamut_routing_lib.exact_types import binary64_operand

__all__ = [
    "Limits",
    "ScaleError",
    "agrees_with_checker",
    "ScalePlan",
    "decimal_places",
    "max_decimals",
    "format_float_matrix",
    "plan_coordinate_scale",
    "plan_explicit_scale",
    "plan_float_matrix_scale",
    "scale_coordinates",
    "scale_explicit_data",
    "star_bound",
    "unscale_cost",
]

#: Largest number of decimals tried before data counts as full precision.
MAX_DECIMALS = 9
#: Smallest exponent tried in approximate mode (``s = 10**-3``).
MIN_EXPONENT = -3


try:  # numpy speeds up large matrices; the lib core does not require it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised without the pyvrp extra
    _np = None


class ScaleError(ValueError):
    """No scale factor fits the solver's numeric limits for this instance."""


@dataclass(frozen=True)
class Limits:
    """Numeric limits of a solver, in *scaled* units (``None`` = unbounded)."""

    max_arc_cost: float | None = None
    max_total_cost: float | None = None
    max_time_value: float | None = None
    max_demand: float | None = None
    min_arc_cost: float | None = None


@dataclass(frozen=True)
class ScalePlan:
    """How instance data was scaled for one solver run."""

    mode: Literal["explicit", "coordinates"]
    factor: float | int | None
    exact: bool
    rounding: Literal["exact", "nearest", "conservative", "solver", "float"]
    max_scaled_arc: int | float
    star_bound_scaled: int | float
    #: Per-arc rounding error bound in the input reading of the search model.
    max_arc_error: float = 0.0
    coordinate_decimals: int | None = None
    #: The coordinate unit implied by the factor for metre data (provenance only).
    unit: str | None = None

    search_model: str = ""
    # The value of one emitted integer in instance units. The numeric factor
    # above is for display and can be inexact or outside the binary64 range.
    cost_unit: Fraction | None = None
    total_cost_bound_scaled: int | None = None
    time_bound_scaled: int | None = None

    def to_json(self) -> dict[str, Any]:
        data = {key: value for key, value in asdict(self).items() if value is not None}
        if self.cost_unit is not None:
            data["cost_unit"] = {"numerator": str(self.cost_unit.numerator), "denominator": str(self.cost_unit.denominator)}
        return data


# --------------------------------------------------------------------------- #
# Reading decimals
# --------------------------------------------------------------------------- #


def _exact_value(value: int | float) -> Fraction:
    """Interpret a matrix operand on the adapter's shortest-decimal scale."""
    from mamut_routing_lib import checker

    return checker.shortest_decimal_cost_value(value)


def decimal_places(value: int | float, *, limit: int = MAX_DECIMALS) -> int | None:
    """Decimals in the shortest spelling (``0.125`` -> 3, ``7`` -> 0), ``None`` above ``limit``."""
    denominator = _exact_value(value).denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    places = max(twos, fives)
    return places if places <= limit else None


def max_decimals(values: Any, *, limit: int = MAX_DECIMALS) -> int | None:
    """Smallest ``k <= limit`` such that every value is exactly ``round(v * 10**k) / 10**k`` (``None`` if none).

    A round trip, not a tolerance: ``0.123`` has 3 decimals; a full-precision
    double such as ``265.88719412562915`` (Sintef2008) or float noise such as
    ``0.30000000000000004`` has none within the limit. The round trip is a fast
    estimate of the shortest-decimal reading. Confirm the answer against
    that reading with :func:`agrees_with_checker` and return ``None`` on disagreement.
    """
    flat = _flatten(values)
    if not flat:
        return 0
    found: int | None = None
    if _np is not None:
        array = _np.asarray(flat, dtype=_np.float64)
        for places in range(0, limit + 1):
            factor = 10.0**places
            if _np.array_equal(_np.round(array * factor) / factor, array):
                found = places
                break
    else:
        for places in range(0, limit + 1):
            factor = 10.0**places
            if all(math.isfinite(v * factor) and round(v * factor) / factor == v for v in flat):
                found = places
                break
    if found is None or not agrees_with_checker(flat, found):
        return None
    return found


def agrees_with_checker(values: Any, places: int, *, sample: int = 4096) -> bool:
    """Whether each sampled shortest decimal value is a multiple of ``10**-places``.

    ``max_decimals`` decides by a float round trip, and collections declare their
    decimals (``cvrplib.collection_cost_decimals``); an exact scale is only sound if
    its shortest-decimal reading agrees. ``sample`` values, evenly strided through
    ``values`` (a flat sequence or rows), are checked with that reading.
    """
    rows = values if isinstance(values, (list, tuple)) else ([] if values is None else list(values))
    if not rows:
        return True
    items = iter(rows) if isinstance(rows[0], numbers.Real) else itertools.chain.from_iterable(rows)
    total = len(rows) if isinstance(rows[0], numbers.Real) else sum(len(row) for row in rows)
    step = max(1, total // max(1, sample))
    scale = 10**places
    return all((_exact_value(value) * scale).denominator == 1 for value in itertools.islice(items, 0, None, step))


def _declared_or_measured(values: Any, declared: int | None) -> int | None:
    """The declared decimals when every value fits and sampled shortest decimals agree, else ``max_decimals``.

    The hint selects the starting resolution for the decimal search model. A failed
    round trip causes the data to be measured instead.
    """
    if declared is not None and 0 <= declared <= MAX_DECIMALS:
        factor = 10.0**declared
        if all(math.isfinite(v * factor) and round(v * factor) / factor == v for v in _flatten(values)) and agrees_with_checker(values, declared):
            return declared
    return max_decimals(values)


def _flatten(values: Any) -> list[float]:
    if values is None:
        return []
    out: list[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            out.extend(float(x) for x in item)
        else:
            out.append(float(item))
    return out


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #


def star_bound(arc_costs: Sequence[Sequence[float]], depot: int = 0) -> float:
    """``sum_i (a[depot][i] + a[i][depot])``: the cost of one route per customer."""
    return float(sum(float(arc_costs[depot][i]) + float(arc_costs[i][depot]) for i in range(len(arc_costs)) if i != depot))


def _max_arc(arc_costs: Sequence[Sequence[float]]) -> float:
    if _np is not None:
        return float(_np.max(_np.asarray(arc_costs, dtype=_np.float64)))
    return float(max(max(row) for row in arc_costs))


def _fits(limits: Limits, *, arc: float, total: float, time: float | None) -> bool:
    if limits.max_arc_cost is not None and arc > limits.max_arc_cost:
        return False
    if limits.max_total_cost is not None and total > limits.max_total_cost:
        return False
    if time is not None and limits.max_time_value is not None and time > limits.max_time_value:
        return False
    if limits.min_arc_cost is not None and arc < limits.min_arc_cost:
        return False
    return True


# --------------------------------------------------------------------------- #
# Explicit matrices
# --------------------------------------------------------------------------- #


def _common_dyadic_unit(values: Iterable[float]) -> Fraction:
    numerator, denominator = 0, 1
    for value in values:
        n, d = binary64_operand(value).as_integer_ratio()
        if d > denominator:
            numerator = math.gcd(numerator * (d // denominator), n)
            denominator = d
        elif numerator != 1:
            numerator = math.gcd(numerator, n * (denominator // d))
    # The all-zero data have no largest unit. One is the canonical choice.
    return Fraction(numerator or 1, denominator)


def _integer_weights(unit: Fraction, rounding: str) -> Callable[[float], int]:
    if rounding == "exact":
        def exact(value: float) -> int:
            n, d = binary64_operand(value).as_integer_ratio()
            quotient, remainder = divmod(n * unit.denominator, d * unit.numerator)
            if remainder:
                raise ScaleError("operand is not a multiple of the selected binary64 unit")
            return quotient

        return exact

    choose = {"nearest": round, "up": math.ceil, "down": math.floor}[rounding]
    return lambda value: choose(_exact_value(binary64_operand(value)) / unit)


def plan_explicit_scale(
    arc_costs: Sequence[Sequence[float]],
    *,
    depot: int = 0,
    time_windows: Sequence[Sequence[float]] | None = None,
    service_times: Sequence[float] | None = None,
    limits: Limits = Limits(),
    require_exact: bool = False,
    decimals_hint: int | None = None,
) -> ScalePlan:
    """Choose a common operand unit and check the solver's bounds in emitted integer units."""
    if (time_windows is None) != (service_times is None):
        raise ScaleError("time windows and service times must be supplied together")
    all_values = itertools.chain(
        itertools.chain.from_iterable(arc_costs),
        itertools.chain.from_iterable(time_windows or ()),
        service_times or (),
    )
    exact_unit = _common_dyadic_unit(all_values)
    row_maxima = [max(row) for row in arc_costs]
    customers = [i for i in range(len(arc_costs)) if i != depot]
    vrptw = time_windows is not None

    def candidate(unit: Fraction, exact: bool) -> ScalePlan | None:
        rounding = "exact" if exact else ("conservative" if vrptw else "nearest")
        weight = _integer_weights(unit, "up" if rounding == "conservative" else rounding)
        arc = weight(max(row_maxima))
        # Each customer has one outgoing arc, and each depot departure reaches
        # a distinct customer. This bound also covers nonmetric/asymmetric data.
        departures = sum(weight(arc_costs[depot][i]) for i in customers)
        total = departures + sum(weight(row_maxima[i]) for i in customers)
        time_bound = None
        if time_windows is not None:
            close = weight if exact else _integer_weights(unit, "down")
            # A route can depart by the depot deadline and wait at any opening.
            travel = min(total, (len(customers) + 1) * arc)
            start = max(close(time_windows[depot][1]), max(weight(ready) for ready, _ in time_windows))
            elapsed = start + travel + sum(map(weight, service_times or ()))
            time_bound = max(elapsed, max(close(due) for _, due in time_windows))
        if not _fits(limits, arc=arc, total=total, time=time_bound):
            return None
        reciprocal = 1 / unit
        try:
            factor = reciprocal.numerator if reciprocal.denominator == 1 else float(reciprocal)
        except OverflowError:
            factor = None
        return ScalePlan(
            mode="explicit", factor=factor, exact=exact, rounding=rounding,
            search_model="binary64-exact-integer" if exact else "shortest-decimal-grid",
            cost_unit=unit, max_scaled_arc=arc,
            star_bound_scaled=departures + sum(weight(arc_costs[i][depot]) for i in customers),
            total_cost_bound_scaled=total, time_bound_scaled=time_bound,
            max_arc_error=0.0 if exact else float(unit) * (1.0 if vrptw else 0.5),
        )

    plan = candidate(exact_unit, True)
    if plan is not None:
        return plan
    if require_exact:
        raise ScaleError(f"no exact binary64 integer scale fits the solver's limits (unit {exact_unit})")
    decimals = _declared_or_measured(arc_costs, decimals_hint)
    if decimals is not None and time_windows is not None:
        windows = max_decimals(time_windows)
        services = max_decimals(service_times or ())
        decimals = None if windows is None or services is None else max(decimals, windows, services)
    top = MAX_DECIMALS if decimals is None else min(decimals, MAX_DECIMALS)
    for exponent in range(top, MIN_EXPONENT - 1, -1):
        plan = candidate(Fraction(10) ** -exponent, False)
        if plan is not None:
            return plan
    raise ScaleError(f"no integer scale between 10^{top} and 10^{MIN_EXPONENT} fits the solver's limits")


def scale_explicit_data(
    plan: ScalePlan,
    arc_costs: Sequence[Sequence[float]],
    *,
    time_windows: Sequence[Sequence[float]] | None = None,
    service_times: Sequence[float] | None = None,
) -> tuple[list[list[int]], list[tuple[int, int]] | None, list[int] | None]:
    """Apply the selected rational unit to every operand sent to the integer solver."""
    if plan.cost_unit is None:
        raise ScaleError("integer export needs an explicit integer scale plan")
    weight = _integer_weights(plan.cost_unit, "up" if plan.rounding == "conservative" else plan.rounding)
    matrix = [[weight(value) for value in row] for row in arc_costs]
    if time_windows is None:
        return matrix, None, None
    close = _integer_weights(plan.cost_unit, "down") if plan.rounding == "conservative" else weight
    windows = [(weight(ready), close(due)) for ready, due in time_windows]
    for vertex, (ready, due) in enumerate(windows):
        if ready > due:
            raise ScaleError(f"integer unit {plan.cost_unit} empties the time window at vertex {vertex}: [{ready}, {due}]")
    return matrix, windows, [weight(value) for value in (service_times or ())]


def unscale_cost(raw: int | float | Decimal | None, plan: ScalePlan) -> tuple[float | None, Fraction | None]:
    """Return a display value and, for exact integer exports, the lossless unscaled report."""
    if raw is None:
        return None, None
    if isinstance(raw, Decimal):
        # Even the extreme binary64 units shift the display by at most 324
        # decimal places. Never expand a compact exponent into a huge Fraction.
        if not raw.is_finite() or raw.adjusted() > 700:
            return None, None
        if raw.adjusted() < -700:
            return -0.0 if raw.is_signed() else 0.0, None
        if raw.as_tuple().exponent < -2000:
            # A malformed fractional report can contain arbitrary precision.
            # Its written text remains available in the run provenance.
            return None, None
        unit = plan.cost_unit
        if unit is None:
            assert plan.factor is not None
            unit = 1 / Fraction(plan.factor)
        try:
            return float(Fraction(raw) * unit), None
        except OverflowError:
            return None, None
    if plan.cost_unit is None:
        assert plan.factor is not None
        try:
            display = float(raw) / plan.factor
        except OverflowError:
            return None, None
        return (display if math.isfinite(display) else None), None
    report = Fraction(raw)
    value = report * plan.cost_unit
    try:
        display = float(value)
    except OverflowError:
        display = None
    return display, value if plan.exact and report.denominator == 1 else None


# --------------------------------------------------------------------------- #
# Explicit matrices for solvers that compute in floating point (HGS-CVRP)
# --------------------------------------------------------------------------- #


def plan_float_matrix_scale(
    arc_costs: Sequence[Sequence[float]],
    *,
    depot: int = 0,
    candidates: Sequence[float] = (1.0, 0.1, 0.01, 0.001),
    limits: Limits = Limits(),
    decimals_hint: int | None = None,
) -> ScalePlan:
    """The first factor of ``candidates`` whose largest arc and star bound fit the limits.

    At factor one the floating reader recovers the published binary64 values.
    A shifted decimal spelling can round to a different binary64 significand.
    """
    largest_arc = _max_arc(arc_costs)
    star = star_bound(arc_costs, depot)
    for factor in candidates:
        factor = float(factor)
        if not _fits(limits, arc=largest_arc * factor, total=star * factor, time=None):
            continue
        exact = factor == 1.0
        return ScalePlan(
            mode="explicit",
            factor=factor,
            exact=exact,
            rounding="float",
            search_model="floating-matrix",
            max_scaled_arc=largest_arc * factor,
            star_bound_scaled=star * factor,
            max_arc_error=0.0 if exact else 1e-15 * largest_arc,
        )
    raise ScaleError(
        f"no matrix scale among {list(candidates)} fits the solver's limits "
        f"(largest arc {largest_arc:g}, star bound {star:g})"
    )


def _plain_number(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() and abs(value) < 2**53 else repr(value)


def format_float_matrix(arc_costs: Sequence[Sequence[float]], plan: ScalePlan) -> Iterable[str]:
    """Matrix rows as decimal text for ``plan``: the published values at factor 1, exact decimal
    multiples at ``10**-k`` (``Decimal`` arithmetic on each value's shortest text)."""
    from decimal import Decimal

    factor = float(plan.factor)
    if factor == 1.0:
        for row in arc_costs:
            yield " ".join(_plain_number(value) for value in row)
        return
    shift = round(math.log10(factor))
    if not math.isclose(10.0**shift, factor):
        for row in arc_costs:
            yield " ".join(repr(float(value) * factor) for value in row)
        return
    for row in arc_costs:
        yield " ".join(format(Decimal(_plain_number(value)).scaleb(shift), "f") for value in row)


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #

_UNITS = {1000.0: "mm", 100.0: "cm", 10.0: "dm", 1.0: "m", 0.1: "dam", 0.01: "hm", 0.001: "km"}


def plan_coordinate_scale(
    coordinates: Sequence[Sequence[float]],
    *,
    depot: int = 0,
    candidates: Sequence[float] = (1.0,),
    limits: Limits = Limits(),
    margin: float = 0.9,
    distance_model: Literal["float", "nint"] = "float",
    published_decimals: int | None = None,
) -> ScalePlan:
    """Choose the coordinate factor for a solver that computes Euclidean distances itself."""
    xs = [float(point[0]) for point in coordinates]
    ys = [float(point[1]) for point in coordinates]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    dx, dy = float(coordinates[depot][0]), float(coordinates[depot][1])
    star = sum(2.0 * math.hypot(x - dx, y - dy) for index, (x, y) in enumerate(zip(xs, ys)) if index != depot)
    for factor in candidates:
        factor = float(factor)
        arc = diagonal * factor
        arc_limit = None if limits.max_arc_cost is None else limits.max_arc_cost * margin
        if arc_limit is not None and arc > arc_limit:
            continue
        if limits.min_arc_cost is not None and arc < limits.min_arc_cost:
            continue
        if limits.max_total_cost is not None and star * factor > limits.max_total_cost:
            continue
        decimals = max(0, 6 - int(round(math.log10(factor))))
        published = 0.0 if published_decimals is None else 0.5 * 10.0 ** (-published_decimals)
        error = published + (0.5 / factor if distance_model == "nint" else 0.0)
        return ScalePlan(
            mode="coordinates",
            factor=factor,
            exact=False,
            rounding="solver",
            search_model="coordinate-distances",
            max_scaled_arc=arc,
            star_bound_scaled=star * factor,
            max_arc_error=error,
            coordinate_decimals=decimals,
            unit=_UNITS.get(factor),
        )
    raise ScaleError(
        f"no coordinate scale among {list(candidates)} fits the solver's limits "
        f"(bounding-box diagonal {diagonal:g}, star bound {star:g})"
    )


def scale_coordinates(coordinates: Sequence[Sequence[float]], plan: ScalePlan) -> list[tuple[float, float]]:
    return [(float(x) * plan.factor, float(y) * plan.factor) for x, y in coordinates]

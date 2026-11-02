"""Integer scaling of instance data for external solvers.

Most literature solvers work in integers, or round distances to integers, and
have numeric limits (HGS-CVRP refuses distances above 1e5, AILS-II keeps its
acceptance threshold in an ``int``, LKH-3 multiplies costs by ``PRECISION``
inside 32-bit integers). This module picks a scale factor ``s`` per solver and
instance and says whether the scaled problem is *exactly* the published one.

Explicit matrices (:func:`plan_explicit_scale`):

- **Exact**: ``s = 10**K`` where ``K`` is the largest number of decimals among
  arc costs, time windows and service times (read as the decimals their JSON
  text spells, like the checker's ``exact_cost_value``). Mamut2026 and
  Poryos2026 matrices have 3 decimals, so ``s = 1000``; Dimacs2021 is already
  integral (``s = 1``). Used when the scaled largest arc, star bound and
  largest due date fit the solver's limits. The solver's reported cost then
  unscales exactly (``Fraction(R, s)``).
- **Approximate**: otherwise (full-precision Sintef2008 doubles, or limits too
  tight), the largest ``s = 10**k`` that fits. CVRP rounds to nearest. VRPTW
  rounds *conservatively*: travel times, service times and ready times up, due
  dates down, so any schedule the solver finds feasible is feasible for the
  checker on the published data.

Coordinates (:func:`plan_coordinate_scale`, for solvers that compute Euclidean
distances themselves): the first candidate factor from the solver's list whose
bounding-box diagonal and star bound fit its limits. FILO2 rounds distances to
integers, hence millimetres (``s = 1000``) for metre coordinates; HGS-CVRP gets
metres, or decametres when the diagonal exceeds 90 km. Always approximate.

The scaled star bound ``sum(a[0][i] + a[i][0])`` (one route per customer) bounds
the cost of any reasonable incumbent, which is what solver accumulators hold.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Literal, Sequence

__all__ = [
    "Limits",
    "ScaleError",
    "ScalePlan",
    "decimal_places",
    "max_decimals",
    "plan_coordinate_scale",
    "plan_explicit_scale",
    "scale_coordinates",
    "scale_explicit_data",
    "star_bound",
    "unscale_cost",
]

#: Largest number of decimals tried before data counts as full precision.
MAX_DECIMALS = 9
#: Smallest exponent tried in approximate mode (``s = 10**-3``).
MIN_EXPONENT = -3
#: A scaled value within this many float ulps of an integer is that integer (``0.1 * 10`` is
#: ``1.0000000000000002``); anything further is rounded in the requested direction.
_ULP_TOLERANCE = 8 * 2.220446049250313e-16

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
    factor: float
    exact: bool
    rounding: Literal["exact", "nearest", "conservative", "solver"]
    max_scaled_arc: float
    star_bound_scaled: float
    #: Worst rounding error per arc, in instance units (``0`` when exact).
    max_arc_error: float = 0.0
    coordinate_decimals: int | None = None
    #: The coordinate unit implied by the factor for metre data (provenance only).
    unit: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


# --------------------------------------------------------------------------- #
# Reading decimals
# --------------------------------------------------------------------------- #


def decimal_places(value: int | float, *, limit: int = MAX_DECIMALS) -> int | None:
    """Decimals the value's shortest text spells (``0.125`` -> 3, ``7`` -> 0), ``None`` above ``limit``."""
    if isinstance(value, bool):
        raise TypeError("not a number")
    if isinstance(value, int) or float(value).is_integer():
        return 0
    denominator = Fraction(repr(float(value))).denominator
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
    ``0.30000000000000004`` has none within the limit, like the checker's reading
    of the decimal text.
    """
    flat = _flatten(values)
    if not flat:
        return 0
    if _np is not None:
        array = _np.asarray(flat, dtype=_np.float64)
        for places in range(0, limit + 1):
            factor = 10.0**places
            if _np.array_equal(_np.round(array * factor) / factor, array):
                return places
        return None
    for places in range(0, limit + 1):
        factor = 10.0**places
        if all(round(v * factor) / factor == v for v in flat):
            return places
    return None


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
    """Choose the scale for an explicit integer matrix (see the module docstring)."""
    largest_arc = _max_arc(arc_costs)
    star = star_bound(arc_costs, depot)
    horizon = max((float(window[1]) for window in time_windows), default=None) if time_windows else None
    if decimals_hint is not None:
        decimals: int | None = decimals_hint
    else:
        decimals = max_decimals(arc_costs)
    if decimals is not None and time_windows is not None:
        windows = max_decimals(time_windows)
        services = max_decimals(service_times or [])
        decimals = None if windows is None or services is None else max(decimals, windows, services)

    if decimals is not None:
        factor = 10.0**decimals
        if _fits(limits, arc=largest_arc * factor, total=star * factor, time=None if horizon is None else horizon * factor):
            return ScalePlan(
                mode="explicit",
                factor=factor,
                exact=True,
                rounding="exact",
                max_scaled_arc=largest_arc * factor,
                star_bound_scaled=star * factor,
            )
    if require_exact:
        reason = (
            "its data has more than 9 decimals" if decimals is None else
            f"its exact scale 10^{decimals} exceeds the solver's limits"
        )
        raise ScaleError(f"no exact integer scale: {reason}")
    top = MAX_DECIMALS if decimals is None else min(decimals, MAX_DECIMALS)
    vrptw = time_windows is not None
    for exponent in range(top, MIN_EXPONENT - 1, -1):
        factor = 10.0**exponent
        if _fits(limits, arc=largest_arc * factor, total=star * factor, time=None if horizon is None else horizon * factor):
            return ScalePlan(
                mode="explicit",
                factor=factor,
                exact=False,
                rounding="conservative" if vrptw else "nearest",
                max_scaled_arc=largest_arc * factor,
                star_bound_scaled=star * factor,
                max_arc_error=(1.0 if vrptw else 0.5) / factor,
            )
    raise ScaleError(
        f"no scale between 10^{top} and 10^{MIN_EXPONENT} fits the solver's limits "
        f"(largest arc {largest_arc:g}, star bound {star:g})"
    )


def _round_matrix(arc_costs: Sequence[Sequence[float]], factor: float, mode: str) -> list[list[int]]:
    if _np is not None:
        array = _np.asarray(arc_costs, dtype=_np.float64) * factor
        if mode == "conservative":
            # Up, but a value already integral within float noise stays put (0.1*10 is 1.0000000000000002).
            near = _np.abs(array - _np.round(array)) <= _ULP_TOLERANCE * _np.maximum(1.0, _np.abs(array))
            rounded = _np.where(near, _np.round(array), _np.ceil(array))
        else:
            rounded = _np.round(array)
        return rounded.astype(_np.int64).tolist()
    return [[_round_value(float(value) * factor, "up" if mode == "conservative" else "nearest") for value in row] for row in arc_costs]


def _round_value(value: float, direction: str) -> int:
    nearest = round(value)
    if abs(value - nearest) <= _ULP_TOLERANCE * max(1.0, abs(value)):
        return int(nearest)
    if direction == "up":
        return int(math.ceil(value))
    if direction == "down":
        return int(math.floor(value))
    return int(nearest)


def scale_explicit_data(
    plan: ScalePlan,
    arc_costs: Sequence[Sequence[float]],
    *,
    time_windows: Sequence[Sequence[float]] | None = None,
    service_times: Sequence[float] | None = None,
) -> tuple[list[list[int]], list[tuple[int, int]] | None, list[int] | None]:
    """The integer matrix, time windows and service times for ``plan``."""
    factor = plan.factor
    conservative = plan.rounding == "conservative"
    matrix = _round_matrix(arc_costs, factor, "conservative" if conservative else "nearest")
    windows = None
    services = None
    if time_windows is not None:
        windows = [
            (
                _round_value(float(ready) * factor, "up" if conservative else "nearest"),
                _round_value(float(due) * factor, "down" if conservative else "nearest"),
            )
            for ready, due in time_windows
        ]
        services = [_round_value(float(value) * factor, "up" if conservative else "nearest") for value in (service_times or [])]
    return matrix, windows, services


def unscale_cost(raw: int | float | None, plan: ScalePlan) -> tuple[float | None, Fraction | None]:
    """(instance-unit cost, exact Fraction when the plan is exact and the raw value integral)."""
    if raw is None:
        return None, None
    exact = None
    if plan.exact and float(raw).is_integer():
        exact = Fraction(int(round(float(raw)))) / Fraction(plan.factor).limit_denominator(10**12)
    return float(raw) / plan.factor, exact


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

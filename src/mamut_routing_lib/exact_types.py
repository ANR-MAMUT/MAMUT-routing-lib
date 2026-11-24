"""Lossless outputs and route checks shared by the two exact profiles."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isfinite
from numbers import Integral
from typing import Literal

from mamut_routing_lib.enums import ObjectiveFunction, SolutionCheckStatus

STATIC_PROFILE = "binary64-exact-v1"
TD_PROFILE = "td-binary64-exact-v1"
ExactBackend = Literal["native", "python"]


def binary64_operand(value: int | float) -> float:
    if type(value) not in (int, float):
        raise ValueError("expected a numeric binary64 input")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("operand exceeds the binary64 input range") from exc
    if not isfinite(result) or result < 0:
        raise ValueError("expected a finite nonnegative binary64 operand")
    return result


def nonnegative_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def exact_resource(value: int, name: str) -> int:
    value = nonnegative_integer(value, name)
    if binary64_operand(value) != value:
        raise ValueError(f"{name} must be exactly representable in binary64")
    return value


class ExactRouteError(ValueError):
    """The submitted routes violate a constraint of the selected profile."""

    def __init__(self, status: SolutionCheckStatus, message: str):
        super().__init__(message)
        self.status = status


def _fraction_record(value: Fraction) -> dict[str, str]:
    # Decimal(int) converts exactly and does not depend on CPython's global
    # limit for formatting large integers as decimal strings.
    return {"numerator": str(Decimal(value.numerator)), "denominator": str(Decimal(value.denominator))}


@dataclass(frozen=True)
class ExactRouteEvaluation:
    duration: Fraction
    departure_time: Fraction

    def as_dict(self) -> dict:
        return {"duration": _fraction_record(self.duration), "departure_time": _fraction_record(self.departure_time)}


@dataclass(frozen=True, slots=True)
class ExactRouteKnot:
    """A departure, its attained depot arrival and the arrival limit immediately to its right."""

    departure: Fraction
    arrival: Fraction
    right_arrival: Fraction


@dataclass(frozen=True)
class ExactEvaluation:
    """A feasible submission. TD witnesses follow canonical stored route order."""

    profile: str
    backend: str
    objective: ObjectiveFunction
    routes: tuple[tuple[int, ...], ...]
    cost: Fraction
    route_evaluations: tuple[ExactRouteEvaluation, ...] = ()

    @property
    def display_cost(self) -> float | None:
        try:
            return float(self.cost)
        except OverflowError:
            return None

    @property
    def objective_key(self) -> tuple:
        if self.objective == ObjectiveFunction.HIERARCHICAL_VEHICLE_COST:
            return (len(self.routes), self.cost)
        return (self.cost,)

    def as_dict(self) -> dict:
        """Return JSON-compatible evaluation data, before attaching manifest or BKS provenance."""
        display = self.display_cost
        data = {
            "profile": self.profile, "backend": self.backend, "objective": self.objective.value,
            "routes": [list(route) for route in self.routes], "num_routes": len(self.routes),
            "authoritative_cost": _fraction_record(self.cost),
            "cost": display, "display_overflow": display is None,
        }
        if self.profile == TD_PROFILE:
            data["route_evaluations"] = [evaluation.as_dict() for evaluation in self.route_evaluations]
        return data


def checked_stops(route, num_customers: int) -> tuple[int, ...]:
    """Normalize a route's customer identifiers before any backend uses them as indices."""
    if not isinstance(route, Sequence) or isinstance(route, (bytes, str)):
        raise ExactRouteError(SolutionCheckStatus.INVALID_CUSTOMER_INDEX, "Expected a sequence of customer identifiers.")
    normalized = []
    for stop in route:
        if isinstance(stop, bool) or not isinstance(stop, Integral) or not 1 <= stop <= num_customers:
            raise ExactRouteError(SolutionCheckStatus.INVALID_CUSTOMER_INDEX, f"Invalid customer index {stop!r}.")
        normalized.append(int(stop))
    return tuple(normalized)


def canonical_checked_routes(routes, *, num_customers, demands, capacity, fleet_limit) -> tuple[tuple[int, ...], ...]:
    """Check the submitted enumeration before sorting its nonempty customer sequences."""
    status = SolutionCheckStatus
    if not isinstance(routes, Sequence) or isinstance(routes, (bytes, str)):
        raise ExactRouteError(status.INVALID_CUSTOMER_INDEX, "Expected a sequence of routes.")
    seen = set()
    checked = []
    for route in routes:
        if not isinstance(route, Sequence) or isinstance(route, (bytes, str)) or not route:
            raise ExactRouteError(status.INVALID_CUSTOMER_INDEX, "Each route must be a nonempty customer sequence.")
        load = 0
        stops = checked_stops(route, num_customers)
        for customer in stops:
            if customer in seen:
                raise ExactRouteError(status.CUSTOMER_SERVED_MULTIPLE_TIMES, f"Customer {customer} occurs twice.")
            seen.add(customer)
            load += demands[customer]
        if load > capacity:
            raise ExactRouteError(status.VEHICLE_CAPACITY_EXCEEDED, "Route demand exceeds vehicle capacity.")
        checked.append(stops)
    if len(seen) != num_customers:
        raise ExactRouteError(status.NOT_ALL_CUSTOMERS_SERVED, "Some customers are missing.")
    if fleet_limit is not None and len(checked) > fleet_limit:
        raise ExactRouteError(status.TOO_MANY_VEHICLES_USED, "The solution exceeds the fleet limit.")
    return tuple(sorted(checked))

"""The solver's reported objective and its binary64 display in family cost units."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
import re


_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")
_COST_OBJECTIVES = {"total-arc-cost", "total-route-duration"}
_VALUE_KEYS = {"objective", "encoding", "value"}
_UNIT_KEY = "family_units_per_solver_unit"


def make_solver_report(objective, value, *, encoding, unit=None, routing_cost_component=None) -> dict:
    """Build a report. The optional routing component uses the same unit as the objective."""
    report = {"objective": objective, "encoding": encoding, "value": value}
    if unit is not None:
        # Decimal(int) keeps the standard fraction encoding independent of Python's integer text limit.
        report[_UNIT_KEY] = {"numerator": str(Decimal(unit.numerator)), "denominator": str(Decimal(unit.denominator))}
    if routing_cost_component is not None:
        report["routing_cost_component"] = routing_cost_component
    return report


def _reported_value(record: dict) -> Decimal | float:
    objective = record.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("solver_report requires a named objective")
    value = record.get("value")
    encoding = record.get("encoding")
    if encoding == "binary64":
        if type(value) in (int, float):
            try:
                number = float(value)
                if math.isfinite(number) and number == value:
                    return number
            except OverflowError:
                pass
        raise ValueError("solver_report binary64 value must be a finite JSON number representable in binary64")
    syntax = {"integer": _INTEGER, "decimal-text": _DECIMAL}.get(encoding) if isinstance(encoding, str) else None
    if syntax and isinstance(value, str) and syntax.fullmatch(value):
        try:
            return Decimal(value)
        except InvalidOperation:
            pass
    raise ValueError("solver_report requires an integer token, a decimal-text token or a binary64 number")


def _read_unit(record: dict) -> Fraction:
    if not isinstance(record, dict) or set(record) != {"numerator", "denominator"}:
        raise ValueError(f"solver_report.{_UNIT_KEY} requires a positive rational")
    for value in record.values():
        if not isinstance(value, str) or not _POSITIVE_INTEGER.fullmatch(value):
            raise ValueError(f"solver_report.{_UNIT_KEY} requires positive integer strings")
    numerator = int(Decimal(record["numerator"]))
    denominator = int(Decimal(record["denominator"]))
    if math.gcd(numerator, denominator) != 1:
        raise ValueError(f"solver_report.{_UNIT_KEY} must be in lowest terms")
    return Fraction(numerator, denominator)


def solver_cost_display(report: dict | None) -> float | None:
    """Validate the report and round the comparable component times its unit once, ties to even."""
    if report is None:
        return None
    if not isinstance(report, dict) or report.keys() - (_VALUE_KEYS | {_UNIT_KEY, "routing_cost_component"}):
        raise ValueError("invalid solver_report fields")
    value = _reported_value(report)
    objective = report["objective"]
    if "routing_cost_component" in report:
        component = report["routing_cost_component"]
        if not isinstance(component, dict) or set(component) != _VALUE_KEYS:
            raise ValueError("solver_report.routing_cost_component requires objective, encoding and value")
        value = _reported_value(component)
        objective = component["objective"]
        if objective not in _COST_OBJECTIVES:
            raise ValueError("solver_report.routing_cost_component requires a routing cost objective")
    if _UNIT_KEY not in report:
        return None
    unit = _read_unit(report[_UNIT_KEY])
    if objective not in _COST_OBJECTIVES:
        return None
    if not value:
        return 0.0
    if isinstance(value, Decimal):
        # Bound the decimal order of the product before expanding a compact exponent.
        order = value.adjusted() + len(report[_UNIT_KEY]["numerator"]) - len(report[_UNIT_KEY]["denominator"])
        if order > 310:
            return None
        if order < -326:
            return -0.0 if value.is_signed() else 0.0
    try:
        return float(Fraction(value) * unit)
    except OverflowError:
        return None


def solver_report_metadata(report: dict | None) -> dict:
    """Return the optional JSON fields shared by run outputs and BKS metadata."""
    if report is None:
        return {}
    display = solver_cost_display(report)
    if display is None:
        return {"solver_report": report}
    return {"solver_report": report, "solver_cost": display}


def checked_solver_metadata(metadata: dict) -> dict:
    """Check a supplied display against the report. Legacy records without a report remain readable."""
    if "solver_report" not in metadata:
        return metadata
    if metadata["solver_report"] is None:
        raise ValueError("omit solver_report when the solver supplies no report")
    fields = solver_report_metadata(metadata["solver_report"])
    if "solver_cost" in metadata:
        display = metadata["solver_cost"]
        if type(display) not in (int, float) or display != fields.get("solver_cost"):
            raise ValueError("solver_cost differs from the solver report's projection in family units")
    return {**metadata, **fields}

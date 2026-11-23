from fractions import Fraction
import json

import pytest

from mamut_routing_lib import STATIC_PROFILE, TD_PROFILE, evaluate_exact, prepare_exact
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_types import ExactEvaluation
from td_utils import write_toy_instance_files


def test_public_dispatch_and_lossless_json(tmp_path):
    static = tmp_path / "static.vrp.json"
    static.write_text(json.dumps({
        "instance_name": "toy", "instance_origin": "Solomon1987", "benchmark_name": "Poryos2026",
        "coordinates": [[0, 0], [1, 0]], "num_customers": 1, "demands": [0, 1],
        "vehicle_capacity": 1, "arc_costs": [[0, 0.1], [0.2, 0]],
    }))
    result = evaluate_exact(static, [[1]], profile=STATIC_PROFILE, backend="python")
    assert result.cost == Fraction.from_float(0.1) + Fraction.from_float(0.2)
    assert "route_evaluations" not in result.as_dict()
    with pytest.raises(ValueError, match="profile"):
        prepare_exact(static, profile=TD_PROFILE, backend="python")

    td = write_toy_instance_files(tmp_path, gzip_sidecar=True)
    prepared = prepare_exact(td, backend="python", profile=TD_PROFILE)
    result = prepared.evaluate([[2], [1]])
    assert result.routes == ((1,), (2,))
    assert result.cost == sum(w.duration for w in result.route_evaluations) == 45
    assert result.as_dict()["route_evaluations"][0]["departure_time"] == {"numerator": "0", "denominator": "1"}


def test_integer_string_encoding_beyond_cpython_format_limit():
    cost = Fraction(10**5000 + 1, 3)
    result = ExactEvaluation(TD_PROFILE, "python-fraction-partition", ObjectiveFunction.DURATION, (), cost)
    data = result.as_dict()
    assert data["authoritative_cost"] == {"numerator": "1" + "0" * 4999 + "1", "denominator": "3"}
    assert data["cost"] is None and data["display_overflow"] is True

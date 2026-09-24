"""The published td-fold/2 test vectors (``fixtures/td/fold-v2-vectors.json``) match this lib.

The vectors are the hand-off to reimplementations of the checker fold
(KAYROS): each micro vector records the td-fold/2 result of one operation and,
where it illustrates the change, the pre-0.12 (td-fold/1) result. The route
vectors need the benchmark data and are checked by the MAMUT-routing test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamut_routing_lib.td import TD_CHECKER_CONTRACT
from mamut_routing_lib.td.pwlf import NDCPWLF, apply_ready_time, make_service_theta, make_theta, restrict_domain

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "td" / "fold-v2-vectors.json").read_text())
HORIZON_END = 43200.0


def _fn(payload: dict) -> NDCPWLF:
    return NDCPWLF(payload["xs"], payload["ys"])


def _as_dict(f: NDCPWLF) -> dict:
    return {"xs": list(f.xs), "ys": list(f.ys)}


def _td_fold_2(vector: dict) -> NDCPWLF:
    if vector["op"] == "apply_ready_time":
        return apply_ready_time(
            _fn(vector["acc"]),
            earliest=vector["earliest"],
            latest=vector["latest"],
            service_time=vector["service_time"],
        )
    if vector["op"] == "restrict_domain":
        return restrict_domain(_fn(vector["f"]), vector["low"], vector["high"])
    return _fn(vector["f"]).compose(_fn(vector["g"]), slope_one_exact=True)


def _td_fold_1(vector: dict) -> NDCPWLF:
    if vector["op"] == "apply_ready_time":
        if vector["earliest"] is None:
            theta = make_service_theta(HORIZON_END, vector["service_time"])
        else:
            theta = make_theta(vector["earliest"], vector["latest"], vector["service_time"])
        return theta.compose(_fn(vector["acc"]))
    if vector["op"] == "restrict_domain":
        return _fn(vector["f"]).compose(NDCPWLF.identity(vector["low"], vector["high"]))
    return _fn(vector["f"]).compose(_fn(vector["g"]))


def test_vectors_carry_the_current_contract() -> None:
    assert VECTORS["contract"] == TD_CHECKER_CONTRACT


@pytest.mark.parametrize("vector", VECTORS["micro"], ids=[vector["name"] for vector in VECTORS["micro"]])
def test_micro_vector(vector: dict) -> None:
    assert _as_dict(_td_fold_2(vector)) == vector["expected"]
    if vector.get("td_fold_1") is not None:
        assert _as_dict(_td_fold_1(vector)) == vector["td_fold_1"]


def test_the_vectors_show_the_contract_change() -> None:
    differing = [v for v in VECTORS["micro"] if v.get("td_fold_1") not in (None, v["expected"])]
    assert len(differing) >= 5

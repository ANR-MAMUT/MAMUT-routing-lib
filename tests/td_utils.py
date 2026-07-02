"""Shared helpers for the TD test modules: a tiny hand-checkable toy instance."""

from __future__ import annotations

from pathlib import Path

from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.td import (
    InstanceATFs,
    NDCPWLF,
    compute_atf_sha256,
    save_instance_atfs,
)

TOY_HORIZON = (0.0, 100.0)


def constant_travel_atf(travel_time: float) -> NDCPWLF:
    return NDCPWLF(
        [TOY_HORIZON[0], TOY_HORIZON[1]],
        [TOY_HORIZON[0] + travel_time, TOY_HORIZON[1] + travel_time],
    )


def make_toy_atfs() -> InstanceATFs:
    """Complete graph over depot 0 and customers 1, 2.

    Arc (1, 2) is genuinely time-dependent: departing at t <= 50 costs
    30 - 0.4 t (congestion easing linearly), departing later costs 10.
    Its ATF breakpoints: (0, 30), (50, 60), (100, 110).
    """
    arcs = {
        (0, 1): constant_travel_atf(10.0),
        (0, 2): constant_travel_atf(15.0),
        (1, 0): constant_travel_atf(10.0),
        (1, 2): NDCPWLF([0.0, 50.0, 100.0], [30.0, 60.0, 110.0]),
        (2, 0): constant_travel_atf(10.0),
        (2, 1): constant_travel_atf(15.0),
    }
    return InstanceATFs(
        instance_name="TOY1",
        benchmark_name="Dabia2013",
        horizon=TOY_HORIZON,
        num_customers=2,
        arcs=arcs,
        generator={"name": "test-fixture"},
    )


def toy_instance_payload(*, with_time_windows: bool = True) -> dict:
    payload = {
        "instance_name": "TOY1",
        "instance_origin": "Solomon1987",
        "benchmark_name": "Dabia2013",
        "num_customers": 2,
        "num_vehicles": 2,
        "vehicle_capacity": 10,
        "coordinates": [[0, 0], [1, 0], [1, 1]],
        "demands": [0, 4, 4],
        "service_times": [0, 0, 0],
        "depot": 0,
        "horizon": [0, 100],
        "td": {"model": "atf-ndcpwlf", "atf_path": "TOY1.atf.json"},
        "metadata": {},
    }
    if with_time_windows:
        payload["time_windows"] = [[0, 100], [0, 100], [0, 100]]
    return payload


def write_toy_instance_files(
    directory: Path,
    *,
    with_time_windows: bool = True,
    gzip_sidecar: bool = False,
) -> Path:
    """Write TOY1.vrp.json + sidecar into ``directory``; return the instance path."""
    atfs = make_toy_atfs()
    sidecar_name = "TOY1.atf.json.gz" if gzip_sidecar else "TOY1.atf.json"
    save_instance_atfs(atfs, directory / sidecar_name)

    payload = toy_instance_payload(with_time_windows=with_time_windows)
    payload["td"]["atf_path"] = sidecar_name
    payload["td"]["atf_sha256"] = compute_atf_sha256(atfs)
    instance_path = directory / "TOY1.vrp.json"
    save_json_to_file(payload, instance_path)
    return instance_path

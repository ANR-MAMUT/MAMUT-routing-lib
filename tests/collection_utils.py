"""A toy family-first collection (Poryos2026-shaped) next to a historic family, for tests."""

from __future__ import annotations

from pathlib import Path

from mamut_routing_lib.distances import (
    InstanceDistances,
    compute_distances_sha256,
    save_instance_distances,
)
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.sidecars import CollectionMarker, save_collection_marker

BASE = "poryos-toyville-n2-hyb"
CITY = "toyville"


def static_payload(*, metric: str, arc_costs_source: dict, with_tw: bool = False) -> dict:
    payload = {
        "instance_name": BASE,
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Poryos2026",
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 10,
        "coordinates": [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
        "demands": [0, 4, 4],
        "depot": 0,
        "metric_variant": metric,
        "arc_costs_source": arc_costs_source,
        "metadata": {"base_instance_name": BASE},
    }
    if with_tw:
        payload["service_times"] = [0, 10, 10]
        payload["time_windows"] = [[0, 86400], [100, 5000], [200, 6000]]
    return payload


def td_payload(subinstance: str, *, with_tw: bool) -> dict:
    payload = {
        "instance_name": f"{BASE}-{subinstance}",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": "Poryos2026",
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 10,
        "coordinates": [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
        "demands": [0, 4, 4],
        "service_times": [0, 10, 10],
        "depot": 0,
        "horizon": [0.0, 86400.0],
        "td": {
            "model": "road-graph",
            "graph": {"path": f"sidecars/{CITY}/n=2/{BASE}/{BASE}.road.json.gz"},
            "traffic": {"path": f"sidecars/{CITY}/n=2/{BASE}/{BASE}.traffic-bpr-heavy.json.gz"},
            "sample_step": 60.0,
            "simplify_tolerance": 1.0,
        },
        "metadata": {"base_instance_name": BASE, "subinstance": subinstance},
    }
    if with_tw:
        payload["time_windows"] = [[0, 86400], [100, 50000], [200, 60000]]
    return payload


def write_toy_collection(tmp_path: Path) -> Path:
    """Write ``<tmp_path>/benchmarks`` with the Poryos2026 toy collection (6 instances incl. a Sintef2008 one)."""
    root = tmp_path / "benchmarks"

    # Family-first collection.
    collection = root / "Poryos2026"
    save_collection_marker(CollectionMarker(family="Poryos2026"), collection)
    distances = InstanceDistances(
        base_name=BASE,
        benchmark_name="Poryos2026",
        metric="fastest",
        num_customers=2,
        values=[[0.0, 120.5, 240.25], [120.5, 0.0, 130.75], [240.25, 130.75, 0.0]],
    )
    distances_rel = f"sidecars/{CITY}/n=2/{BASE}/{BASE}.distances-fastest.json.gz"
    save_instance_distances(distances, collection / distances_rel)
    distances_sha = compute_distances_sha256(distances)

    save_json_to_file(
        static_payload(metric="euclidean", arc_costs_source={"model": "euclidean", "decimals": 3}),
        collection / "CVRP" / "euclidean" / CITY / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    save_json_to_file(
        static_payload(
            metric="fastest",
            arc_costs_source={
                "model": "distances-sidecar",
                "distances": {"path": distances_rel, "sha256": distances_sha},
            },
        ),
        collection / "CVRP" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    save_json_to_file(
        static_payload(
            metric="fastest",
            arc_costs_source={
                "model": "distances-sidecar",
                "distances": {"path": distances_rel, "sha256": distances_sha},
            },
            with_tw=True,
        ),
        collection / "VRPTW" / "fastest" / CITY / "n=2" / BASE / f"{BASE}.vrp.json",
    )
    save_json_to_file(
        td_payload("bpr-heavy", with_tw=False),
        collection / "TDVRP" / CITY / "n=2" / BASE / "bpr-heavy" / f"{BASE}-bpr-heavy.vrp.json",
    )
    save_json_to_file(
        td_payload("bpr-heavy", with_tw=True),
        collection / "TDVRPTW" / CITY / "n=2" / BASE / "bpr-heavy" / f"{BASE}-bpr-heavy.vrp.json",
    )

    # Historic 4-part layout family next to the collection (regression).
    historic = {
        "instance_name": "toy-h1",
        "instance_origin": "Solomon1987",
        "benchmark_name": "Sintef2008",
        "num_customers": 2,
        "num_vehicles": 2,
        "vehicle_capacity": 10,
        "coordinates": [[0, 0], [1, 1], [2, 2]],
        "demands": [0, 4, 4],
        "depot": 0,
        "arc_costs": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "service_times": [0, 10, 10],
        "time_windows": [[0, 100], [0, 100], [0, 100]],
        "metadata": {},
    }
    save_json_to_file(historic, root / "VRPTW" / "Sintef2008" / "n=2" / "toy-h1.vrp.json")
    return root

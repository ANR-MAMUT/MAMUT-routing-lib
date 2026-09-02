from __future__ import annotations

import re
from pathlib import Path

import pytest

from mamut_routing_lib.cvrplib import (
    ExportResult,
    UnsupportedInstanceError,
    VrpExportOptions,
    coordinate_formatter,
    euclidean_arc_costs,
    export_filename,
    export_instance_file,
    format_vrp_comment,
    instance_to_vrp_text,
    render_cvrplib,
    render_solomon,
    vector_formatter,
)
from mamut_routing_lib.artifacts import load_benchmark_instance
from mamut_routing_lib.distances import InstanceDistances, save_instance_distances
from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.models import BenchmarkInstance, BenchmarkInstanceCVRPCollection, BenchmarkInstanceVRPTWCollection
from mamut_routing_lib.sidecars import CollectionMarker, save_collection_marker

from td_utils import write_toy_instance_files


def _collection_payload(*, metric: str, family: str = "Mamut2026", vrptw: bool = False) -> dict:
    payload = {
        "instance_name": "mamut-testville-n2-k1-poi",
        "instance_origin": "OsmCvrpGen",
        "benchmark_name": family,
        "num_customers": 2,
        "num_vehicles": None,
        "vehicle_capacity": 7,
        "coordinates": [[0.0, 0.0], [3.0, 4.0], [-1.5, 2.25]],
        "demands": [0, 3, 4],
        "depot": 0,
        "reference_lla": {"lat": 45.0, "lon": 5.0, "alt": 0.0},
        "metric_variant": metric,
        "arc_costs_source": {"model": "euclidean", "decimals": 3},
        "metadata": {"problem_type": "VRPTW" if vrptw else "CVRP", "city": "testville", "num_vehicles_lb": 1},
    }
    if vrptw:
        payload["service_times"] = [0, 60, 90]
        payload["time_windows"] = [[0, 3600], [100, 2000], [0, 3000]]
        payload["metadata"]["tw_set"] = {"name": "spread", "td_paired": False}
    return payload


def _write_collection(tmp_path: Path, payload: dict, *, sidecar_values: list[list[float]] | None = None) -> Path:
    root = tmp_path / "Mamut2026"
    save_collection_marker(CollectionMarker(family=payload["benchmark_name"]), root)
    base = payload["instance_name"]
    instance_dir = root / "CVRP" / payload["metric_variant"] / "testville" / "n=2" / base
    if sidecar_values is not None:
        metric = payload["metric_variant"]
        sidecar_rel = f"sidecars/testville/n=2/{base}/{base}.distances-{metric}.json.gz"
        distances = InstanceDistances(
            base_name=base,
            benchmark_name=payload["benchmark_name"],
            metric=metric,
            num_customers=2,
            values=sidecar_values,
        )
        save_instance_distances(distances, root / sidecar_rel)
        from mamut_routing_lib.distances import compute_distances_sha256

        payload["arc_costs_source"] = {
            "model": "distances-sidecar",
            "distances": {"path": sidecar_rel, "sha256": compute_distances_sha256(distances)},
        }
    instance_path = instance_dir / f"{base}.vrp.json"
    save_json_to_file(payload, instance_path)
    return instance_path


def _parse_sections(text: str) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current: str | None = None
    for line in text.splitlines():
        if re.fullmatch(r"[A-Z_]+_SECTION", line):
            current = line
            sections[current] = []
        elif line in ("EOF", "") or ":" in line and current is None:
            continue
        elif current is not None:
            sections[current].append(line.split())
    return sections


# --------------------------------------------------------------------------- #
# formatting rules
# --------------------------------------------------------------------------- #


def test_vector_formatter_prints_integers_when_all_integral() -> None:
    fmt = vector_formatter([0, 5.0, 6])
    assert [fmt(v) for v in (0, 5.0, 6)] == ["0", "5", "6"]


def test_vector_formatter_keeps_round_trip_floats_with_dot_zero() -> None:
    fmt = vector_formatter([0.0, 89.89438247187641, 5.0])
    assert [fmt(v) for v in (0.0, 89.89438247187641, 5.0)] == ["0.0", "89.89438247187641", "5.0"]


def test_vector_formatter_fixed_decimals_for_collection_costs() -> None:
    fmt = vector_formatter([0.0, 2731.469], decimals=3)
    assert [fmt(v) for v in (0.0, 2731.469, 12.5)] == ["0.000", "2731.469", "12.500"]


def test_coordinate_formatter_six_decimals_unless_all_integral() -> None:
    assert coordinate_formatter([[100, 100], [20, 59]])(100) == "100"
    assert coordinate_formatter([[-500.874011, 70.822412]])(-500.874011) == "-500.874011"
    assert coordinate_formatter([[1.0, 2.5]])(1.0) == "1.000000"


def test_euclidean_arc_costs_rounds_and_zeroes_diagonal() -> None:
    matrix = euclidean_arc_costs([[0, 0], [3, 4], [-1.5, 2.25]])
    assert matrix[0] == [0.0, 5.0, round((1.5**2 + 2.25**2) ** 0.5, 3)]
    assert matrix[1][1] == 0.0 and matrix[2][1] == matrix[1][2]


# --------------------------------------------------------------------------- #
# renderers / instance conversion
# --------------------------------------------------------------------------- #


def test_toy_cvrp_renders_explicit_full_matrix_golden(toy_cvrp_instance) -> None:
    # The toy coordinates are integral floats: the value-driven rule prints them as integers.
    text = instance_to_vrp_text(toy_cvrp_instance)
    assert text == (
        "NAME : poryos-n2-testcvrp\n"
        "COMMENT : Poryos2026 poryos-n2-testcvrp; authors: Florian Rascoussier (0nyr) and Adrien Pichon (Anzury); "
        "converted from MAMUT-routing .vrp.json\n"
        "TYPE : CVRP\n"
        "DIMENSION : 3\n"
        "EDGE_WEIGHT_TYPE : EXPLICIT\n"
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
        "CAPACITY : 10\n"
        "EDGE_WEIGHT_SECTION\n"
        "0 5 6\n"
        "5 0 3\n"
        "6 3 0\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 1 1\n"
        "3 2 2\n"
        "DEMAND_SECTION\n"
        "1 0\n"
        "2 3\n"
        "3 4\n"
        "DEPOT_SECTION\n"
        "1\n"
        "-1\n"
        "EOF\n"
    )


def test_toy_vrptw_renders_cvrptw_sections_and_fixed_fleet(toy_vrptw_instance) -> None:
    instance = toy_vrptw_instance.model_copy(update={"num_vehicles": 2, "depot": 0})
    text = instance_to_vrp_text(instance, options=VrpExportOptions(comment="custom comment"))
    lines = text.splitlines()
    assert lines[:8] == [
        "NAME : poryos-n2-testvrptw",
        "COMMENT : custom comment",
        "TYPE : CVRPTW",
        "DIMENSION : 3",
        "VEHICLES : 2",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "CAPACITY : 10",
    ]
    sections = _parse_sections(text)
    assert sections["TIME_WINDOW_SECTION"] == [["1", "0", "100"], ["2", "0", "50"], ["3", "0", "50"]]
    assert sections["SERVICE_TIME_SECTION"] == [["1", "0"], ["2", "2"], ["3", "2"]]
    assert sections["DEPOT_SECTION"] == [["1"], ["-1"]]
    assert text.endswith("EOF\n")


def test_depot_index_is_one_based_in_depot_section() -> None:
    text = render_cvrplib(
        name="x",
        comment="",
        coordinates=[[0, 0], [1, 1]],
        demands=[2, 0],
        capacity=5,
        depot=1,
        arc_costs=[[0, 1], [1, 0]],
    )
    assert "COMMENT" not in text
    assert text.split("DEPOT_SECTION\n")[1] == "2\n-1\nEOF\n"


def test_render_cvrplib_validates_shapes() -> None:
    with pytest.raises(ValueError, match="DIMENSION"):
        render_cvrplib(name="x", comment="", coordinates=[[0, 0], [1, 1]], demands=[0], capacity=1, arc_costs=[[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="arc_costs"):
        render_cvrplib(name="x", comment="", coordinates=[[0, 0], [1, 1]], demands=[0, 1], capacity=1)


def test_historical_float_matrix_keeps_full_precision(toy_vrptw_instance) -> None:
    instance = toy_vrptw_instance.model_copy(
        update={"arc_costs": [[0.0, 5.5, 6.25], [5.5, 0.0, 3.0], [6.25, 3.0, 0.0]], "coordinates": [(0, 0), (1, 1), (2, 2)]}
    )
    sections = _parse_sections(instance_to_vrp_text(instance))
    assert sections["EDGE_WEIGHT_SECTION"] == [["0.0", "5.5", "6.25"], ["5.5", "0.0", "3.0"], ["6.25", "3.0", "0.0"]]
    assert sections["NODE_COORD_SECTION"] == [["1", "0", "0"], ["2", "1", "1"], ["3", "2", "2"]]


def test_collection_euclidean_source_renders_three_decimals_and_family_comment(tmp_path: Path) -> None:
    instance_path = _write_collection(tmp_path, _collection_payload(metric="euclidean"))
    instance = load_benchmark_instance(instance_path)
    assert isinstance(instance, BenchmarkInstanceCVRPCollection)

    text = instance_to_vrp_text(instance, instance_path=instance_path)
    assert text.splitlines()[1] == (
        "COMMENT : Mamut2026 euclidean metric; city testville; No of trucks: 1 (lower bound, fleet not fixed); "
        "3-decimal seconds/meters; ENU ref in mamut-testville-n2-k1-poi.vrp.json"
    )
    sections = _parse_sections(text)
    assert sections["EDGE_WEIGHT_SECTION"][0] == ["0.000", "5.000", "2.704"]
    assert sections["EDGE_WEIGHT_SECTION"][1][1] == "0.000"
    assert sections["NODE_COORD_SECTION"][2] == ["3", "-1.500000", "2.250000"]


def test_poryos_comment_has_no_fleet_clause_and_names_the_tw_set(tmp_path: Path) -> None:
    payload = _collection_payload(metric="fastest", family="Poryos2026", vrptw=True)
    payload["instance_name"] = "poryos-testville-n2-poi-tw-spread"
    instance = BenchmarkInstanceVRPTWCollection(**payload)
    assert format_vrp_comment(instance) == (
        "Poryos2026 fastest metric; city testville; 3-decimal seconds/meters; "
        "ENU ref in poryos-testville-n2-poi-tw-spread.vrp.json; time windows set spread"
    )
    assert format_vrp_comment(instance, edge_weight_type="EUC_2D").endswith(
        "; EUC_2D: costs are TSPLIB nint distances, not the published 3-decimal costs"
    )


def test_collection_sidecar_source_hydrates_the_pinned_matrix(tmp_path: Path) -> None:
    values = [[0.0, 10.5, 20.25], [11.0, 0.0, 7.125], [19.0, 8.0, 0.0]]
    instance_path = _write_collection(tmp_path, _collection_payload(metric="shortest"), sidecar_values=values)
    instance = load_benchmark_instance(instance_path)

    text = instance_to_vrp_text(instance, instance_path=instance_path)
    sections = _parse_sections(text)
    assert sections["EDGE_WEIGHT_SECTION"] == [
        ["0.000", "10.500", "20.250"],
        ["11.000", "0.000", "7.125"],
        ["19.000", "8.000", "0.000"],
    ]
    # A caller holding the matrix bypasses the sidecar but keeps the collection precision.
    assert instance_to_vrp_text(instance, values) == text
    with pytest.raises(ValueError, match="instance_path"):
        instance_to_vrp_text(instance)


def test_euc_2d_drops_the_matrix_for_euclidean_instances_only(tmp_path: Path) -> None:
    euclid_path = _write_collection(tmp_path / "e", _collection_payload(metric="euclidean"))
    text = instance_to_vrp_text(load_benchmark_instance(euclid_path), options=VrpExportOptions(edge_weight_type="EUC_2D"))
    lines = text.splitlines()
    assert "EDGE_WEIGHT_TYPE : EUC_2D" in lines
    assert "EDGE_WEIGHT_FORMAT : FULL_MATRIX" not in lines
    assert "EDGE_WEIGHT_SECTION" not in lines
    assert lines.index("EDGE_WEIGHT_TYPE : EUC_2D") + 1 == lines.index("CAPACITY : 7")

    shortest_path = _write_collection(tmp_path / "s", _collection_payload(metric="shortest"), sidecar_values=[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    with pytest.raises(UnsupportedInstanceError, match="euclidean"):
        instance_to_vrp_text(load_benchmark_instance(shortest_path), instance_path=shortest_path, options=VrpExportOptions(edge_weight_type="EUC_2D"))


def test_solomon_golden_and_rejections(toy_cvrp_instance) -> None:
    instance = BenchmarkInstance(
        instance_name="C101",
        instance_origin="Solomon1987",
        benchmark_name="Sintef2008",
        num_customers=2,
        num_vehicles=25,
        vehicle_capacity=200,
        coordinates=[(40, 50), (45, 68), (45, 70)],
        demands=[0, 10, 30],
        service_times=[0, 90, 90],
        time_windows=[(0, 1236), (912, 967), (825, 870)],
        depot=0,
        arc_costs=[[0, 1, 2], [1, 0, 3], [2, 3, 0]],
        metadata={"metric_variant": "euclidean", "authors": "Marius M. Solomon"},
    )
    text = instance_to_vrp_text(instance, options=VrpExportOptions(format="solomon"))
    assert text == (
        "C101\n"
        "\n"
        "VEHICLE\n"
        "NUMBER     CAPACITY\n"
        "    25       200\n"
        "\n"
        "CUSTOMER\n"
        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME\n"
        "\n"
        "    0         40         50          0          0       1236          0\n"
        "    1         45         68         10        912        967         90\n"
        "    2         45         70         30        825        870         90\n"
    )
    assert export_filename("C101.vrp.json", VrpExportOptions(format="solomon")) == "C101.txt"

    with pytest.raises(UnsupportedInstanceError, match="VRPTW-only"):
        instance_to_vrp_text(toy_cvrp_instance, options=VrpExportOptions(format="solomon"))
    fastest = instance.model_copy(update={"metadata": {"metric_variant": "fastest"}})
    with pytest.raises(UnsupportedInstanceError, match="euclidean"):
        instance_to_vrp_text(fastest, options=VrpExportOptions(format="solomon"))
    with pytest.raises(ValueError, match="depot"):
        render_solomon(name="x", capacity=1, num_vehicles=None, coordinates=[[0, 0], [1, 1]], demands=[0, 1], time_windows=[[0, 1], [0, 1]], service_times=[0, 0], depot=1)


def test_time_dependent_instances_are_unsupported(tmp_path: Path) -> None:
    instance_path = write_toy_instance_files(tmp_path)
    instance = load_benchmark_instance(instance_path)
    with pytest.raises(UnsupportedInstanceError, match="time-dependent"):
        instance_to_vrp_text(instance, instance_path=instance_path)
    result = export_instance_file(instance_path)
    assert result.status == "unsupported"
    assert result.problem_type.value == "TDVRPTW"
    assert not result.output_path.exists()


def test_export_instance_file_writes_sibling_and_respects_existing(tmp_path: Path, toy_cvrp_instance) -> None:
    source = tmp_path / "poryos-n2-testcvrp.vrp.json"
    save_json_to_file(toy_cvrp_instance.model_dump(mode="json"), source)

    result = export_instance_file(source)
    assert isinstance(result, ExportResult)
    assert result.status == "written"
    assert result.output_path == tmp_path / "poryos-n2-testcvrp.vrp"
    assert result.output_path.read_text(encoding="utf-8") == instance_to_vrp_text(toy_cvrp_instance)
    assert result.num_customers == 2 and result.problem_type.value == "CVRP"

    result.output_path.write_text("stale", encoding="utf-8")
    assert export_instance_file(source).status == "exists"
    assert result.output_path.read_text(encoding="utf-8") == "stale"
    assert export_instance_file(source, overwrite=True).status == "written"
    assert result.output_path.read_text(encoding="utf-8").startswith("NAME : poryos-n2-testcvrp\n")

    explicit = export_instance_file(source, tmp_path / "out" / "custom.vrp", options=VrpExportOptions(edge_weight_type="EXPLICIT"))
    assert explicit.output_path == tmp_path / "out" / "custom.vrp" and explicit.output_path.is_file()


def test_export_options_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="edge_weight_type"):
        VrpExportOptions(edge_weight_type="EUC_3D")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="format"):
        VrpExportOptions(format="tsplib")  # type: ignore[arg-type]

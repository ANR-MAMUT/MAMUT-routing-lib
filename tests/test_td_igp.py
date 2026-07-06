"""Tests for the igp-profile td model: categories sidecar, materialization, loader."""

from __future__ import annotations

import random

import pytest

from mamut_routing_lib.json_utils import save_json_to_file
from mamut_routing_lib.td import (
    ATFFormatError,
    IGPFormatError,
    InstanceCategories,
    TDIGPProfileRef,
    atfs_to_canonical_json_bytes,
    build_arc_atf,
    compute_atf_sha256,
    compute_categories_sha256,
    compute_route_duration,
    euclidean_distance,
    ichoua_travel_time,
    load_instance_categories,
    load_td_instance,
    materialize_instance_atfs,
    save_instance_categories,
    td_instance_from_payload,
)

# IGP 2003 Table 1, scenario 2 (a = 2).
S2_SPEEDS = [[0.33, 0.67, 0.33], [0.67, 1.33, 0.67], [1.33, 2.67, 1.33]]
HORIZON = (0.0, 300.0)
PERIODS = [[0.0, 100.0], [100.0, 200.0], [200.0, 300.0]]
COORDINATES = [[0, 0], [40, 0], [40, 30], [0, 30]]  # depot + 3 customers


def make_categories(num_customers: int = 3) -> InstanceCategories:
    n = num_customers + 1
    rng = random.Random(42)
    rows = [["0"] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = str(rng.getrandbits(32) % 3)
            rows[i][j] = c
            rows[j][i] = c
    return InstanceCategories(
        base_name="Lera-TOY",
        benchmark_name="Lera2026",
        num_customers=num_customers,
        num_categories=3,
        categories=["".join(row) for row in rows],
        generator={"name": "test-fixture", "seed": 42},
    )


def igp_instance_payload(*, with_time_windows: bool = True) -> dict:
    payload = {
        "instance_name": "Lera-TOY-S2",
        "instance_origin": "GehHom1999",
        "benchmark_name": "Lera2026",
        "num_customers": 3,
        "num_vehicles": 2,
        "vehicle_capacity": 10,
        "coordinates": COORDINATES,
        "demands": [0, 4, 4, 2],
        "service_times": [0, 10, 10, 10],
        "depot": 0,
        "horizon": [0.0, 300.0],
        "td": {
            "model": "igp-profile",
            "time_periods": PERIODS,
            "speeds": S2_SPEEDS,
            "categories_path": "Lera-TOY.igp.json",
        },
        "metadata": {},
    }
    if with_time_windows:
        payload["time_windows"] = [[0, 300], [0, 300], [0, 300], [0, 300]]
    return payload


def write_igp_instance_files(directory, *, gzip_sidecar: bool = False, with_time_windows: bool = True):
    categories = make_categories()
    sidecar_name = "Lera-TOY.igp.json.gz" if gzip_sidecar else "Lera-TOY.igp.json"
    save_instance_categories(categories, directory / sidecar_name)

    payload = igp_instance_payload(with_time_windows=with_time_windows)
    payload["td"]["categories_path"] = sidecar_name
    payload["td"]["categories_sha256"] = compute_categories_sha256(categories)
    instance = td_instance_from_payload(payload)
    payload["td"]["atf_sha256"] = compute_atf_sha256(materialize_instance_atfs(instance, categories))
    instance_path = directory / "Lera-TOY-S2.vrp.json"
    save_json_to_file(payload, instance_path)
    return instance_path


class TestIchouaEngine:
    def test_travel_time_crosses_boundaries(self):
        # Distance 40 at speed 0.33 from t=0: crosses into period 2 (speed 0.67).
        # Period 1 covers 0.33 * 100 = 33 distance; remaining 7 at 0.67.
        tt = ichoua_travel_time(PERIODS, S2_SPEEDS[0], 40.0, 0.0)
        assert tt == pytest.approx(100.0 + 7.0 / 0.67, abs=1e-12)

    def test_atf_matches_direct_loop_everywhere(self):
        for speeds in S2_SPEEDS:
            atf = build_arc_atf(PERIODS, speeds, 40.0, HORIZON)
            for k in range(400):
                t0 = HORIZON[0] + k * (HORIZON[1] - HORIZON[0]) / 399
                direct = t0 + ichoua_travel_time(PERIODS, speeds, 40.0, t0)
                assert atf.evaluate(t0) == pytest.approx(direct, abs=1e-9)

    def test_atf_breakpoints_span_horizon(self):
        atf = build_arc_atf(PERIODS, S2_SPEEDS[1], 25.0, HORIZON)
        assert atf.xs[0] == HORIZON[0]
        assert atf.xs[-1] == HORIZON[1]
        assert all(y >= x for x, y in zip(atf.xs, atf.ys))


class TestCategoriesSidecar:
    def test_roundtrip_plain_and_gzip(self, tmp_path):
        categories = make_categories()
        for name in ("Lera-TOY.igp.json", "Lera-TOY.igp.json.gz"):
            save_instance_categories(categories, tmp_path / name)
            loaded = load_instance_categories(tmp_path / name)
            assert loaded == categories
            assert compute_categories_sha256(loaded) == compute_categories_sha256(categories)

    def test_symmetry_violation_rejected(self, tmp_path):
        categories = make_categories()
        rows = [list(row) for row in categories.categories]
        rows[1][2] = str((int(rows[1][2]) + 1) % 3)
        categories.categories = ["".join(row) for row in rows]
        with pytest.raises(IGPFormatError, match="symmetric"):
            save_instance_categories(categories, tmp_path / "bad.igp.json")

    def test_diagonal_must_be_zero(self, tmp_path):
        categories = make_categories()
        rows = [list(row) for row in categories.categories]
        rows[2][2] = "1"
        categories.categories = ["".join(row) for row in rows]
        with pytest.raises(IGPFormatError, match="diagonal"):
            save_instance_categories(categories, tmp_path / "bad.igp.json")

    def test_category_out_of_range_rejected(self, tmp_path):
        categories = make_categories()
        rows = [list(row) for row in categories.categories]
        rows[0][1] = "7"
        rows[1][0] = "7"
        categories.categories = ["".join(row) for row in rows]
        with pytest.raises(IGPFormatError, match="out of range"):
            save_instance_categories(categories, tmp_path / "bad.igp.json")


class TestMaterialization:
    def test_deterministic_bytes_and_sha(self):
        categories = make_categories()
        instance = td_instance_from_payload(igp_instance_payload())
        first = materialize_instance_atfs(instance, categories)
        second = materialize_instance_atfs(instance, categories)
        assert atfs_to_canonical_json_bytes(first) == atfs_to_canonical_json_bytes(second)
        assert compute_atf_sha256(first) == compute_atf_sha256(second)

    def test_complete_graph_and_cross_validation(self):
        categories = make_categories()
        instance = td_instance_from_payload(igp_instance_payload())
        atfs = materialize_instance_atfs(instance, categories)
        assert len(atfs.arcs) == 4 * 3
        rng = random.Random(7)
        for (i, j), atf in atfs.arcs.items():
            distance = euclidean_distance(COORDINATES[i], COORDINATES[j])
            speeds = S2_SPEEDS[categories.category(i, j)]
            for _ in range(25):
                t0 = rng.uniform(*HORIZON)
                direct = t0 + ichoua_travel_time(PERIODS, speeds, distance, t0)
                assert atf.evaluate(t0) == pytest.approx(direct, abs=1e-9)

    def test_generator_is_the_fixed_materializer_constant(self):
        categories = make_categories()
        instance = td_instance_from_payload(igp_instance_payload())
        atfs = materialize_instance_atfs(instance, categories)
        assert atfs.generator == {"name": "igp-profile-materializer", "version": 1}


class TestLoader:
    @pytest.mark.parametrize("gzip_sidecar", [False, True])
    def test_load_end_to_end(self, tmp_path, gzip_sidecar):
        instance_path = write_igp_instance_files(tmp_path, gzip_sidecar=gzip_sidecar)
        loaded = load_td_instance(instance_path)
        assert loaded.atf_path is None
        assert loaded.categories_path is not None
        assert len(loaded.atfs.arcs) == 12
        # The checker consumes materialized ATFs like any sidecar-backed instance.
        evaluation = compute_route_duration(loaded.instance, loaded.atfs, [1, 2, 3])
        assert evaluation.feasible

    def test_tdvrp_twin_loads(self, tmp_path):
        instance_path = write_igp_instance_files(tmp_path, with_time_windows=False)
        loaded = load_td_instance(instance_path)
        assert not hasattr(loaded.instance, "time_windows")
        assert len(loaded.atfs.arcs) == 12

    def test_wrong_categories_sha_rejected(self, tmp_path):
        instance_path = write_igp_instance_files(tmp_path)
        payload = igp_instance_payload()
        payload["td"]["categories_sha256"] = "0" * 64
        save_json_to_file(payload, instance_path)
        with pytest.raises(ATFFormatError, match="categories sha256 mismatch"):
            load_td_instance(instance_path)

    def test_wrong_atf_sha_rejected(self, tmp_path):
        instance_path = write_igp_instance_files(tmp_path)
        payload = igp_instance_payload()
        categories = make_categories()
        payload["td"]["categories_sha256"] = compute_categories_sha256(categories)
        payload["td"]["atf_sha256"] = "0" * 64
        save_json_to_file(payload, instance_path)
        with pytest.raises(ATFFormatError, match="materialized ATF sha256 mismatch"):
            load_td_instance(instance_path)
        loaded = load_td_instance(instance_path, verify_sha256=False)
        assert len(loaded.atfs.arcs) == 12


class TestModelValidation:
    def test_non_positive_speed_rejected(self):
        payload = igp_instance_payload()
        payload["td"]["speeds"] = [[0.33, 0.0, 0.33]] + S2_SPEEDS[1:]
        with pytest.raises(ValueError, match="strictly positive"):
            td_instance_from_payload(payload)

    def test_non_contiguous_periods_rejected(self):
        payload = igp_instance_payload()
        payload["td"]["time_periods"] = [[0.0, 100.0], [110.0, 300.0]]
        with pytest.raises(ValueError, match="contiguous"):
            td_instance_from_payload(payload)

    def test_periods_must_span_horizon(self):
        payload = igp_instance_payload()
        payload["td"]["time_periods"] = [[0.0, 100.0], [100.0, 200.0], [200.0, 290.0]]
        payload["td"]["speeds"] = S2_SPEEDS
        with pytest.raises(ValueError, match="span exactly the horizon"):
            td_instance_from_payload(payload)

    def test_ragged_speeds_rejected(self):
        payload = igp_instance_payload()
        payload["td"]["speeds"] = [[0.33, 0.67], [0.67, 1.33, 0.67], [1.33, 2.67, 1.33]]
        with pytest.raises(ValueError, match="one per time period"):
            td_instance_from_payload(payload)

    def test_igp_ref_discriminated(self):
        instance = td_instance_from_payload(igp_instance_payload())
        assert isinstance(instance.td, TDIGPProfileRef)
        assert instance.td.num_categories() == 3

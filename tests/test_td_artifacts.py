from __future__ import annotations

import gzip
import json

import pytest

from mamut_routing_lib.artifacts import discover_benchmark_instances, load_benchmark_instance
from mamut_routing_lib.enums import ProblemType
from mamut_routing_lib.td import (
    ATFFormatError,
    BenchmarkInstanceTDVRP,
    BenchmarkInstanceTDVRPTW,
    atfs_to_canonical_json_bytes,
    compute_atf_sha256,
    load_instance_atfs,
    load_td_instance,
    save_instance_atfs,
)
from td_utils import make_toy_atfs, write_toy_instance_files


class TestSidecarRoundTrip:
    def test_plain_round_trip(self, tmp_path):
        atfs = make_toy_atfs()
        path = tmp_path / "TOY1.atf.json"
        save_instance_atfs(atfs, path)
        loaded = load_instance_atfs(path)
        assert loaded.arcs == atfs.arcs
        assert loaded.horizon == atfs.horizon
        assert loaded.generator == atfs.generator

    def test_gzip_round_trip_same_sha(self, tmp_path):
        atfs = make_toy_atfs()
        plain = tmp_path / "TOY1.atf.json"
        packed = tmp_path / "TOY1.atf.json.gz"
        save_instance_atfs(atfs, plain)
        save_instance_atfs(atfs, packed)
        assert compute_atf_sha256(load_instance_atfs(plain)) == compute_atf_sha256(
            load_instance_atfs(packed)
        )
        # gzip bytes are deterministic (mtime=0)
        again = tmp_path / "TOY1-bis.atf.json.gz"
        save_instance_atfs(atfs, again)
        assert packed.read_bytes() == again.read_bytes()

    def test_canonical_bytes_stable_after_parse(self, tmp_path):
        atfs = make_toy_atfs()
        path = tmp_path / "TOY1.atf.json"
        save_instance_atfs(atfs, path)
        assert atfs_to_canonical_json_bytes(load_instance_atfs(path)) == path.read_bytes()

    def test_bad_suffix_rejected(self, tmp_path):
        with pytest.raises(ATFFormatError):
            save_instance_atfs(make_toy_atfs(), tmp_path / "TOY1.json")


class TestSidecarValidation:
    def _payload(self):
        return json.loads(atfs_to_canonical_json_bytes(make_toy_atfs()).decode("utf-8"))

    def _write(self, tmp_path, payload):
        path = tmp_path / "BAD.atf.json"
        path.write_text(json.dumps(payload))
        return path

    def test_unsorted_arcs_rejected(self, tmp_path):
        payload = self._payload()
        payload["arcs"][0], payload["arcs"][1] = payload["arcs"][1], payload["arcs"][0]
        with pytest.raises(ATFFormatError, match="sorted"):
            load_instance_atfs(self._write(tmp_path, payload))

    def test_incomplete_graph_rejected(self, tmp_path):
        payload = self._payload()
        payload["arcs"].pop()
        with pytest.raises(ATFFormatError, match="complete"):
            load_instance_atfs(self._write(tmp_path, payload))

    def test_non_fifo_rejected(self, tmp_path):
        payload = self._payload()
        payload["arcs"][0][3] = sorted(payload["arcs"][0][3], reverse=True)
        with pytest.raises(ATFFormatError):
            load_instance_atfs(self._write(tmp_path, payload))

    def test_negative_travel_time_rejected(self, tmp_path):
        payload = self._payload()
        xs = payload["arcs"][0][2]
        payload["arcs"][0][3] = [x - 1.0 for x in xs]
        with pytest.raises(ATFFormatError, match="negative travel time"):
            load_instance_atfs(self._write(tmp_path, payload))

    def test_domain_must_span_horizon(self, tmp_path):
        payload = self._payload()
        payload["arcs"][0][2] = [1.0] + payload["arcs"][0][2][1:]
        with pytest.raises(ATFFormatError, match="horizon"):
            load_instance_atfs(self._write(tmp_path, payload))


class TestInstanceLoading:
    def test_load_td_instance_tdvrptw(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path)
        loaded = load_td_instance(instance_path)
        assert isinstance(loaded.instance, BenchmarkInstanceTDVRPTW)
        assert loaded.atfs.num_customers == 2
        assert loaded.instance.td.atf_sha256 == compute_atf_sha256(loaded.atfs)

    def test_load_td_instance_tdvrp(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path, with_time_windows=False)
        loaded = load_td_instance(instance_path)
        assert isinstance(loaded.instance, BenchmarkInstanceTDVRP)

    def test_load_td_instance_gzip_sidecar(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path, gzip_sidecar=True)
        loaded = load_td_instance(instance_path)
        assert loaded.atf_path.name.endswith(".atf.json.gz")
        assert isinstance(loaded.instance, BenchmarkInstanceTDVRPTW)

    def test_sha_mismatch_rejected(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path)
        payload = json.loads(instance_path.read_text())
        payload["td"]["atf_sha256"] = "0" * 64
        instance_path.write_text(json.dumps(payload))
        with pytest.raises(ATFFormatError, match="sha256 mismatch"):
            load_td_instance(instance_path)

    def test_tampered_gzip_sidecar_detected(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path, gzip_sidecar=True)
        sidecar = tmp_path / "TOY1.atf.json.gz"
        payload = json.loads(gzip.decompress(sidecar.read_bytes()).decode("utf-8"))
        payload["arcs"][0][3] = [y + 1.0 for y in payload["arcs"][0][3]]
        sidecar.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8"), mtime=0))
        with pytest.raises(ATFFormatError, match="sha256 mismatch"):
            load_td_instance(instance_path)


class TestDiscoveryIntegration:
    def test_discover_and_load_tdvrptw_layout(self, tmp_path):
        family_dir = tmp_path / "TDVRPTW" / "Dabia2013" / "n=2"
        family_dir.mkdir(parents=True)
        write_toy_instance_files(family_dir)
        discovered = discover_benchmark_instances(tmp_path)
        assert len(discovered) == 1
        item = discovered[0]
        assert item.problem_type == ProblemType.TDVRPTW
        assert item.benchmark_name == "Dabia2013"
        assert item.num_customers == 2
        assert item.instance_id == "tdvrptw-dabia2013-n2-TOY1"
        instance = item.load()
        assert isinstance(instance, BenchmarkInstanceTDVRPTW)

    def test_generic_loader_routes_td_payloads(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path)
        instance = load_benchmark_instance(instance_path)
        assert isinstance(instance, BenchmarkInstanceTDVRPTW)


class TestReferenceLLA:
    """TD instances may carry the optional reference_lla geodetic anchor
    (local ENU meters to lat/lon, as in the CVRP/VRPTW workbench families)."""

    def test_reference_lla_accepted_and_exposed(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path)
        payload = json.loads(instance_path.read_text())
        payload["reference_lla"] = {"lat": 45.7578147, "lon": 4.83511885, "alt": 0.0}
        instance_path.write_text(json.dumps(payload))
        instance = load_benchmark_instance(instance_path)
        assert instance.reference_lla is not None
        assert instance.reference_lla.lat == 45.7578147
        assert instance.reference_lla.lon == 4.83511885

    def test_reference_lla_defaults_to_none(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path)
        instance = load_benchmark_instance(instance_path)
        assert instance.reference_lla is None

    def test_malformed_reference_lla_rejected(self, tmp_path):
        instance_path = write_toy_instance_files(tmp_path)
        payload = json.loads(instance_path.read_text())
        payload["reference_lla"] = {"lat": 45.0, "lon": 4.8, "altitude": 12.0}
        instance_path.write_text(json.dumps(payload))
        with pytest.raises(Exception, match="reference_lla"):
            load_benchmark_instance(instance_path)

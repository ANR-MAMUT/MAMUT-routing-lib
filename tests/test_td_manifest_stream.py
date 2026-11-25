import gzip
import hashlib
import json

import pytest

from mamut_routing_lib.manifests import atf_stream_sha256
from mamut_routing_lib.td.artifacts import load_td_instance
from mamut_routing_lib.td.manifest_stream import stream_td_materialization
from td_utils import write_toy_instance_files


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("reformat", [False, True])
def test_complete_digest_uses_every_arc_and_retains_selected_arcs(tmp_path, compressed, reformat):
    path = write_toy_instance_files(tmp_path, gzip_sidecar=compressed)
    full = load_td_instance(path, verify_sha256=True).atfs
    payload = json.loads(path.read_bytes())
    sidecar = tmp_path / payload["td"]["atf_path"]
    opener = gzip.open if compressed else open
    with opener(sidecar, "rb") as stream:
        data = stream.read()
    if reformat:
        data = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":")).encode()
        with opener(sidecar, "wb") as stream:
            stream.write(data)
    selected = {(2, 0)}
    result = stream_td_materialization(path, tmp_path, selected)
    rows = ((i, j, arc.xs, arc.ys) for (i, j), arc in sorted(full.arcs.items()))
    assert result.atf_stream_sha256 == atf_stream_sha256(full.num_vertices(), full.horizon, rows)
    assert result.canonical_sha256 == payload["td"]["atf_sha256"]
    assert result.sidecars[0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert result.operands.generator == full.generator
    assert result.num_arcs == 6 and result.operands.arcs.keys() == selected


def test_detects_tamper_outside_requested_arcs(tmp_path):
    path = write_toy_instance_files(tmp_path)
    sidecar = tmp_path / "TOY1.atf.json"
    data = json.loads(sidecar.read_bytes())
    unselected = data["arcs"][0]
    unselected[3][-1] += 1
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="expected"):
        stream_td_materialization(path, tmp_path, {(2, 0)})


def test_stored_sidecar_can_be_recovered_from_verified_cache(tmp_path):
    path = write_toy_instance_files(tmp_path, gzip_sidecar=True)
    selected = {(0, 1), (1, 0)}
    before = stream_td_materialization(path, tmp_path, selected)
    folder = tmp_path / "cache"
    folder.mkdir()
    sidecar = tmp_path / "TOY1.atf.json.gz"
    sidecar.rename(folder / sidecar.name)
    replay = stream_td_materialization(path, tmp_path, selected, atf_cache_root=folder)
    assert replay.atf_stream_sha256 == before.atf_stream_sha256
    assert replay.sidecars == []
    assert replay.operands.arcs == before.operands.arcs


def test_empty_canonical_graph_has_a_digest(tmp_path):
    path = write_toy_instance_files(tmp_path)
    desc = json.loads(path.read_bytes())
    from mamut_routing_lib.td.artifacts import InstanceATFs, compute_atf_sha256, save_instance_atfs
    atfs = InstanceATFs("TOY1", "Dabia2013", (0.0, 100.0), 0, {}, {"source": "empty"})
    save_instance_atfs(atfs, tmp_path / "TOY1.atf.json")
    desc["num_customers"] = 0
    for field in ("coordinates", "demands", "service_times", "time_windows"):
        desc[field] = desc[field][:1]
    desc["td"]["atf_sha256"] = compute_atf_sha256(atfs)
    path.write_text(json.dumps(desc))
    checked = stream_td_materialization(path, tmp_path)
    assert checked.num_arcs == checked.num_knots == 0
    assert checked.atf_stream_sha256 == atf_stream_sha256(1, atfs.horizon, [])


def test_generated_integer_knots_hash_like_the_full_sidecar():
    from dataclasses import replace

    from mamut_routing_lib.td import NDCPWLF, compute_atf_sha256
    from mamut_routing_lib.td.manifest_stream import _generated_records
    from td_utils import make_toy_atfs

    atfs = make_toy_atfs()
    arc = atfs.arcs[1, 2]
    atfs.arcs[1, 2] = NDCPWLF(list(map(int, arc.xs)), list(map(int, arc.ys)))
    digest = hashlib.sha256()
    blocks = _generated_records(
        replace(atfs, arcs={}), lambda keys: {key: atfs.arcs[key] for key in keys}, digest, block_arcs=2,
    )
    assert sum(1 for _ in blocks) == len(atfs.arcs)
    assert digest.hexdigest() == compute_atf_sha256(atfs)

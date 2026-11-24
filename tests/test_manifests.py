import json
from copy import deepcopy

import pytest
from typer.testing import CliRunner

from mamut_routing_lib import load_benchmark_instance, load_bks, manifest_path_for, verify_manifest, write_bks_file
from mamut_routing_lib.cli import app
from mamut_routing_lib.manifests import SPECIFICATION, atf_stream_sha256, file_sha256, manifest_bytes, matrix_sha256
from mamut_routing_lib.objectives import base_objective
from mamut_routing_lib.solutions import check_routes, preview_bks_update, submit_bks
from mamut_routing_lib.td import load_td_instance
from td_utils import write_toy_instance_files
from test_models import make_valid_historical_instance_kwargs


def fixture(root, td):
    if td:
        path = write_toy_instance_files(root, gzip_sidecar=True)
        data = load_td_instance(path)
        atfs = data.atfs
        semantic = atf_stream_sha256(atfs.num_vertices(), atfs.horizon,
                                     ((i, j, f.xs, f.ys) for (i, j), f in sorted(atfs.arcs.items())))
        sidecars = [dict(path=data.atf_path.name, scope="decompressed", sha256=file_sha256(data.atf_path, "decompressed"))]
        material = dict(kind="stored-atf", interpretation="binary64", atf_stream_sha256=semantic)
    else:
        path = root / "C101.vrp.json"
        payload = make_valid_historical_instance_kwargs()
        path.write_text(json.dumps(payload))
        sidecars = []
        material = dict(kind="embedded-matrix", interpretation="binary64", matrix_sha256=matrix_sha256(payload["arc_costs"]))
    model = load_benchmark_instance(path)
    manifest = dict(specification=SPECIFICATION, benchmark_revision="r1", instance_name=model.instance_name,
                    family=model.benchmark_name.value, problem="TDVRPTW" if td else "VRPTW", objective=base_objective(model).value,
                    units=dict(cost="fixture-unit", time="fixture-unit"), default_profile="td-binary64-exact-v1" if td else "binary64-exact-v1",
                    features=["single-depot"], instance=dict(path=path.name, sha256=file_sha256(path)),
                    sidecars=sidecars, materialization=material)
    manifest_path_for(path).write_bytes(manifest_bytes(manifest))
    return path, model, manifest


@pytest.mark.parametrize("td", [False, True])
def test_verify_edition_and_bks_outside_git(tmp_path, td):
    path, model, manifest = fixture(tmp_path, td)
    manifest_path = manifest_path_for(path)
    manifest_sha = file_sha256(manifest_path)
    assert manifest_path.name == path.name.removesuffix(".vrp.json") + ".manifest.json"
    assert verify_manifest(path, expected_sha256=manifest_sha).benchmark_revision == "r1"
    submitted = submit_bks(path, [[1, 2]], authors="Fixture", backend="python")
    bks = load_bks(submitted.path)
    assert bks.metadata["standard"]["manifest_sha256"] == manifest_sha
    assert bks.metadata["standard"]["benchmark_revision"] == "r1"
    assert "exact_evaluation" not in bks.metadata
    assert submit_bks(path, bks.routes, authors="Fixture", marked_input=bks, backend="python").tie
    verdict = check_routes(model, path, bks.routes, backend="python", marked_input=bks)
    assert verdict.is_valid()
    runner = CliRunner()
    assert runner.invoke(app, ["verify-manifest", str(path), "--sha256", manifest_sha]).exit_code == 0
    before = submitted.path.read_bytes()
    wrong = deepcopy(bks)
    wrong.metadata["standard"]["benchmark_revision"] = "r2"
    preview = preview_bks_update(path, wrong.routes, backend="python", marked_input=wrong)
    assert preview["action"] == "invalid"
    assert submitted.path.read_bytes() == before
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(path, expected_sha256="0" * 64)
    payload = json.loads(path.read_bytes())
    payload["time_windows"][-1][1] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(path)
    assert manifest_bytes(manifest) == manifest_path.read_bytes()


def test_canonical_materialization_vectors():
    assert matrix_sha256([[0, .1, .3], [.1, 0, .2], [.3, .2, -0.]]) == "ab8a8d7f8797135d5f7a232191b579112bd067db48991ccff0897d4af01f3ca5"
    arcs = [(0, 1, [-0., 0., 1., 1., 2., 2.], [.5, .75, 1.25, 1.75, 2.25, 2.5]),
            (1, 0, [0., 2.], [5e-324, 2.])]
    assert atf_stream_sha256(2, [-0., 2.], arcs) == "7c78f7702dfae08e9345736456114d92b5bd07a082c50fb76264dad25fd5dfd2"
    assert atf_stream_sha256(1, [0, 4], []) == "392744bcc86d0b1221769100eaf2a62b7b8a0dc1d7849d43e223c096dfdfa9f4"
    with pytest.raises(ValueError, match="Missing"):
        atf_stream_sha256(2, [0, 2], arcs[:1])


def test_manifest_rejects_omitted_artifact_and_forged_stream(tmp_path):
    path, _, doc = fixture(tmp_path, True)
    manifest = manifest_path_for(path)
    wrong = deepcopy(doc)
    wrong["sidecars"] = []
    manifest.write_bytes(manifest_bytes(wrong))
    with pytest.raises(ValueError, match="omits"):
        verify_manifest(path)
    doc["materialization"]["atf_stream_sha256"] = "0" * 64
    manifest.write_bytes(manifest_bytes(doc))
    with pytest.raises(ValueError, match="complete materialization"):
        verify_manifest(path)


def test_certified_improvement_keeps_edition_and_archives_targeted_claim(tmp_path):
    path, _, _ = fixture(tmp_path, False)
    submitted = submit_bks(path, [[1], [2]], authors="Original team", backend="python")
    incumbent = load_bks(submitted.path)
    incumbent.metadata["optimality"] = {
        "status": "solver-reported-optimal", "solver": "Historical solver", "target_profile": "legacy-reading",
    }
    write_bks_file(incumbent, submitted.path)
    original = submitted.path.read_bytes()
    original_sha = file_sha256(submitted.path)
    improved = submit_bks(path, [[1, 2]], authors="New team", backend="python")
    assert improved.action == "replaced"
    certified = load_bks(improved.path)
    assert certified.metadata["standard"]["manifest_sha256"] == incumbent.metadata["standard"]["manifest_sha256"]
    assert "optimality" not in certified.metadata
    assert (tmp_path / ".bks-history" / (original_sha + ".json")).read_bytes() == original


def test_td_digest_only_and_route_specific_views(tmp_path):
    path, _, _ = fixture(tmp_path, True)
    empty = verify_manifest(path, routes=[])
    partial = verify_manifest(path, routes=[[1, 2]])
    full = verify_manifest(path)
    assert not empty.checkable.atfs.arcs
    assert set(partial.checkable.atfs.arcs) == {(0, 1), (1, 2), (2, 0)}
    assert len(full.checkable.atfs.arcs) == 6
    assert empty.result_identity() == partial.result_identity() == full.result_identity()


@pytest.mark.parametrize("change", ["name", "family", "capacity", "matrix"])
def test_static_checker_binds_supplied_model_to_descriptor(tmp_path, change):
    from mamut_routing_lib import BenchmarkName, BenchmarkSolution, check_solution

    path, published, _ = fixture(tmp_path, False)
    other = published.model_copy(deep=True)
    if change == "name":
        other.instance_name = "other"
    elif change == "family":
        other.benchmark_name = BenchmarkName.DIMACS_2021
    elif change == "capacity":
        other.vehicle_capacity = 0
    else:
        other.arc_costs[0][1] += 1
    solution = BenchmarkSolution(instance_name=other.instance_name, routes=[[1, 2]])
    with pytest.raises(ValueError, match="differs from"):
        check_solution(other, solution, instance_path=path, backend="python")
    manifest_path_for(path).unlink()
    with pytest.raises(ValueError, match="differs from"):
        check_solution(other, solution, instance_path=path, backend="python")


def test_td_in_memory_check_and_certified_edition_binding(tmp_path, monkeypatch):
    from mamut_routing_lib import BenchmarkSolution
    from mamut_routing_lib import manifests
    from mamut_routing_lib.td import NDCPWLF, check_td_solution

    path, _, _ = fixture(tmp_path, True)
    loaded = load_td_instance(path)
    routes = [[1, 2]]
    certified = load_bks(submit_bks(path, routes, authors="Fixture", backend="python").path)
    ordinary = BenchmarkSolution(instance_name=loaded.instance.instance_name, routes=routes)
    calls = []
    stream = manifests.stream_td_materialization

    def counted(*args, **kwargs):
        calls.append(args[0])
        return stream(*args, **kwargs)

    monkeypatch.setattr(manifests, "stream_td_materialization", counted)
    assert check_td_solution(loaded, ordinary, backend="python").is_valid()
    assert calls == []
    assert check_td_solution(loaded, certified, backend="python").is_valid()
    assert len(calls) == 1
    arc = loaded.atfs.arcs[0, 1]
    loaded.atfs.arcs[0, 1] = NDCPWLF(arc.xs, [y + 1 for y in arc.ys])
    with pytest.raises(ValueError, match="materialized input differs"):
        check_td_solution(loaded, certified, backend="python")

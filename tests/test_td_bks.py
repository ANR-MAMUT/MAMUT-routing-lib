"""Tests for the TD-aware BKS helpers (Duration objective, TD checker refereed)."""

from __future__ import annotations

import pytest

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks, write_bks_file
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, SolverReportedOptimality
from mamut_routing_lib.td import (
    annotate_td_bks_optimality,
    check_td_solution,
    create_td_bks_from_solution,
    load_td_instance,
    save_td_solution_as_bks_if_improved,
)

from td_utils import write_toy_instance_files

AUTHORS = "Test Author"


@pytest.fixture(params=[True, False], ids=["tdvrptw", "tdvrp"])
def toy_loaded(request, tmp_path):
    instance_path = write_toy_instance_files(tmp_path, with_time_windows=request.param)
    return load_td_instance(instance_path)


def _solution(loaded, routes):
    return BenchmarkSolution(instance_name=loaded.instance.instance_name, routes=routes)


def test_create_td_bks_prices_with_td_checker(toy_loaded):
    solution = _solution(toy_loaded, [[1], [2]])
    bks = create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)
    reference = check_td_solution(toy_loaded, solution)
    assert bks.objective_function == ObjectiveFunction.DURATION
    assert bks.cost == reference.routing_cost
    assert bks.metadata["authors"] == AUTHORS
    assert bks.metadata["validated_cost"] == reference.routing_cost
    assert bks.metadata["validated_num_routes"] == 2


def test_create_td_bks_accepts_instance_path(toy_loaded):
    solution = _solution(toy_loaded, [[1], [2]])
    bks = create_td_bks_from_solution(toy_loaded.instance_path, solution, authors=AUTHORS)
    assert bks.cost == check_td_solution(toy_loaded, solution).routing_cost


def test_create_td_bks_rejects_invalid_solution(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2], [2]])  # customer 2 served twice
    with pytest.raises(ValueError, match="invalid TD solution"):
        create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)


def test_create_td_bks_rejects_cost_mismatch(toy_loaded):
    reference = check_td_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]))
    solution = BenchmarkSolution(
        instance_name=toy_loaded.instance.instance_name,
        routes=[[1], [2]],
        cost=reference.routing_cost + 1.0,
    )
    # check_td_solution itself enforces the exact cost match.
    with pytest.raises(ValueError, match="does not match computed Duration"):
        create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)


def test_create_td_bks_accepts_exact_cost(toy_loaded):
    reference = check_td_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]))
    solution = BenchmarkSolution(
        instance_name=toy_loaded.instance.instance_name,
        routes=[[1], [2]],
        cost=reference.routing_cost,
    )
    bks = create_td_bks_from_solution(toy_loaded, solution, authors=AUTHORS)
    assert bks.cost == reference.routing_cost


def test_create_td_bks_requires_authors(toy_loaded):
    with pytest.raises(ValueError, match="authors"):
        create_td_bks_from_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]), authors="  ")


def test_save_creates_then_keeps_then_replaces(toy_loaded):
    bks_path = get_bks_path_for_instance(toy_loaded.instance_path, ObjectiveFunction.DURATION)
    assert not bks_path.exists()

    # The single-route solution [1, 2] is strictly cheaper than {[1], [2]} on
    # the toy instance (one depot round trip saved).
    worse = _solution(toy_loaded, [[1], [2]])
    better = _solution(toy_loaded, [[1, 2]])
    worse_cost = check_td_solution(toy_loaded, worse).routing_cost
    better_cost = check_td_solution(toy_loaded, better).routing_cost
    assert better_cost < worse_cost

    created = save_td_solution_as_bks_if_improved(toy_loaded, worse, authors=AUTHORS)
    assert created.action == "created"
    assert created.path == bks_path
    assert load_bks(bks_path).cost == worse_cost

    kept = save_td_solution_as_bks_if_improved(toy_loaded, worse, authors=AUTHORS)
    assert kept.action == "kept_existing"
    assert load_bks(bks_path).cost == worse_cost

    replaced = save_td_solution_as_bks_if_improved(toy_loaded, better, authors=AUTHORS)
    assert replaced.action == "replaced"
    assert replaced.previous_path == bks_path
    stored = load_bks(bks_path)
    assert stored.cost == better_cost
    assert stored.routes == [[1, 2]]
    assert stored.objective_function == ObjectiveFunction.DURATION


def test_save_carries_extra_metadata(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    result = save_td_solution_as_bks_if_improved(
        toy_loaded,
        solution,
        authors=AUTHORS,
        metadata={"solver": "kayros-0.0.1.dev0", "seed": 42},
    )
    assert result.action == "created"
    stored = load_bks(result.path)
    assert stored.metadata["solver"] == "kayros-0.0.1.dev0"
    assert stored.metadata["seed"] == 42
    assert stored.metadata["authors"] == AUTHORS


def _optimality(**overrides):
    """The legacy stamp shape, which the library still reads but no longer writes."""
    payload = {
        "proven": True,
        "prover": "test exact solver 1.0",
        "certificate": "optimal under test arithmetic",
        "checker": "td-fold/2",
        "date": "2026-07-07",
    }
    payload.update(overrides)
    return payload


def _claim(**overrides):
    payload = {"status": "solver-reported-optimal", "target_profile": "td-binary64-exact-v1",
               "solver": "test exact solver 1.0"}
    payload.update(overrides)
    return payload


def _store_legacy(loaded, solution):
    """A historical record without an exact evaluation."""
    price = check_td_solution(loaded, solution).routing_cost
    bks = BenchmarkBKS(instance_name=solution.instance_name, objective_function=ObjectiveFunction.DURATION,
                       routes=solution.routes, cost=price, metadata={'authors': AUTHORS})
    write_bks_file(bks, get_bks_path_for_instance(loaded.instance_path, ObjectiveFunction.DURATION))


def _store_exact(loaded, solution):
    return save_td_solution_as_bks_if_improved(loaded, solution, authors=AUTHORS, backend='python').path


def test_annotate_records_a_solver_claim_with_bounds_and_evidence(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    path = _store_exact(toy_loaded, solution)
    before = load_bks(path)
    claim = _claim(reported_primal=before.cost, reported_dual=before.cost, campaign="test run",
                   evidence={"path": "evidence/run.log", "sha256": "a" * 64, "scope": "file"})
    assert annotate_td_bks_optimality(toy_loaded, claim, backend='python') == path
    stored = load_bks(path)
    assert stored.routes == before.routes and stored.cost == before.cost
    assert stored.metadata["authors"] == AUTHORS
    assert stored.metadata["optimality"] == claim


def test_annotate_optimality_accepts_model_instance(toy_loaded):
    path = _store_exact(toy_loaded, _solution(toy_loaded, [[1, 2]]))
    annotate_td_bks_optimality(toy_loaded, SolverReportedOptimality(**_claim()), backend='python')
    assert load_bks(path).metadata["optimality"] == _claim()


def test_annotate_optimality_requires_existing_bks(toy_loaded):
    with pytest.raises(FileNotFoundError, match="No stored Duration BKS"):
        annotate_td_bks_optimality(toy_loaded, _claim())


@pytest.mark.parametrize("claim", [
    _optimality(),
    _optimality(proven_optimum=1.0, dual_bound=1.0),
    {**_claim(), "proven": True},
    {**_claim(), "prover": "test exact solver 1.0"},
    _claim(status="proven-optimal"),
    _claim(target_profile="td-fold/2"),
    _claim(solver=""),
    _claim(reported_dual=float("inf")),
    _claim(evidence={"path": "/abs/run.log", "sha256": "a" * 64}),
    _claim(evidence={"path": "run.log", "sha256": "not-a-digest"}),
])
def test_annotate_refuses_legacy_hybrid_and_malformed_claims(toy_loaded, claim):
    path = _store_exact(toy_loaded, _solution(toy_loaded, [[1, 2]]))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        annotate_td_bks_optimality(toy_loaded, claim, backend='python')
    assert path.read_bytes() == before


def test_annotate_refuses_an_incumbent_without_exact_evaluation(toy_loaded):
    _store_legacy(toy_loaded, _solution(toy_loaded, [[1, 2]]))
    with pytest.raises(ValueError, match="requires an incumbent evaluated"):
        annotate_td_bks_optimality(toy_loaded, _claim(), backend='python')


def test_historical_td_claim_reads_with_its_provenance_but_cannot_be_written(toy_loaded, tmp_path):
    from mamut_routing_lib.artifacts import save_json_to_file

    claim = _claim(target_profile="td-fold/1", arithmetic="historical arithmetic", certificate="record",
                   date="2026-08-06", wall_time_s=5, time_limit_s=60)
    bks = create_td_bks_from_solution(toy_loaded, _solution(toy_loaded, [[1, 2]]), authors=AUTHORS, backend="python")
    raw = bks.model_dump(mode="json")
    raw["metadata"]["optimality"] = claim
    path = tmp_path / "historical.bks.Duration.json"
    save_json_to_file(raw, path)
    loaded = load_bks(path)
    assert loaded.metadata["optimality"] == claim
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_bks_file(loaded, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("claim", [{"status": "solver-reported-optimal"}, _claim(target_profile="td-fold/1"),
                                    _claim(proven=True), _claim(reported_dual=float("inf"))])
def test_writer_checks_mutated_claim_before_replacing_valid_file(toy_loaded, tmp_path, claim):
    path = tmp_path / "test.bks.Duration.json"
    bks = create_td_bks_from_solution(toy_loaded, _solution(toy_loaded, [[1, 2]]), authors=AUTHORS, backend="python")
    bks.metadata["optimality"] = _claim()
    write_bks_file(bks, path)
    assert load_bks(path).metadata["optimality"] == _claim()
    original = path.read_bytes()
    bks.metadata["optimality"] = claim
    with pytest.raises(ValueError):
        write_bks_file(bks, path)
    assert path.read_bytes() == original


def test_write_bks_file_refuses_a_legacy_claim_but_old_files_still_load(toy_loaded, tmp_path):
    from mamut_routing_lib.artifacts import save_json_to_file

    solution = _solution(toy_loaded, [[1, 2]])
    legacy = BenchmarkBKS(instance_name=solution.instance_name, objective_function=ObjectiveFunction.DURATION,
                          routes=solution.routes, metadata={'authors': AUTHORS, 'optimality': _optimality()})
    target = tmp_path / "legacy.bks.Duration.json"
    with pytest.raises(ValueError, match="legacy optimality claim"):
        write_bks_file(legacy, target)
    assert not target.exists()
    save_json_to_file(legacy.model_dump(mode="json"), target)
    assert load_bks(target).metadata["optimality"]["proven"] is True


def test_stamped_metadata_survives_save_if_improved(toy_loaded):
    # A later (non-improving) save attempt must not clobber the claim.
    solution = _solution(toy_loaded, [[1, 2]])
    _store_exact(toy_loaded, solution)
    annotate_td_bks_optimality(toy_loaded, _claim(), backend='python')
    kept = save_td_solution_as_bks_if_improved(toy_loaded, solution, authors=AUTHORS, backend='python')
    assert kept.action == "kept_existing"
    assert load_bks(kept.path).metadata["optimality"] == _claim()


def test_exact_candidate_and_legacy_proof_display_collision(tmp_path):
    from hashlib import sha256
    from mamut_routing_lib.td.pwlf import NDCPWLF

    path = write_toy_instance_files(tmp_path, with_time_windows=False)
    loaded = load_td_instance(path)
    horizon = (0, 2)
    huge = 2**55
    loaded.instance.horizon = loaded.atfs.horizon = horizon
    loaded.atfs.arcs = {arc: NDCPWLF([0, 2], [0, 2]) for arc in loaded.atfs.arcs}
    loaded.atfs.arcs[1, 2] = NDCPWLF([0, 2], [1, 3])
    loaded.atfs.arcs[1, 0] = loaded.atfs.arcs[2, 0] = NDCPWLF([0, 2], [huge, huge])
    store = get_bks_path_for_instance(path, ObjectiveFunction.DURATION)
    incumbent = BenchmarkBKS(instance_name=loaded.instance.instance_name, routes=[[1, 2]], cost=huge,
                             objective_function=ObjectiveFunction.DURATION, metadata={'authors': AUTHORS, 'optimality': _optimality()})
    # Old files with a legacy stamp still load, so craft one without the store writer.
    from mamut_routing_lib.artifacts import save_json_to_file
    save_json_to_file(incumbent.model_dump(mode='json'), store)
    original = store.read_bytes()
    def submit(routes):
        return save_td_solution_as_bks_if_improved(loaded, _solution(loaded, routes), authors=AUTHORS, backend='python')
    assert submit([[1, 2]]).tie and store.read_bytes() == original
    assert submit([[2, 1]]).action == 'replaced'
    stored = load_bks(store)
    assert stored.cost == incumbent.cost == float(huge)
    assert stored.metadata['exact_evaluation']['authoritative_cost'] == {'numerator': str(huge-2), 'denominator': '1'}
    assert 'optimality' not in stored.metadata and 'standard' not in stored.metadata
    assert (store.parent / '.bks-history' / (sha256(original).hexdigest()+'.json')).read_bytes() == original
    assert submit([[1, 2]]).action == 'kept_existing'
    stored.metadata['exact_evaluation']['route_evaluations'][0]['duration']['numerator'] = '0'
    write_bks_file(stored, store)
    with pytest.raises(ValueError, match='route_evaluations'):
        submit([[2, 1]])


def test_marked_td_stamp_requires_explicit_solver_target(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    result = save_td_solution_as_bks_if_improved(toy_loaded, solution, authors=AUTHORS, backend='python')
    with pytest.raises(ValueError):
        annotate_td_bks_optimality(toy_loaded, _optimality(), backend='python')
    stamp = {'status': 'solver-reported-optimal', 'target_profile': 'td-binary64-exact-v1', 'solver': 'test solver'}
    annotate_td_bks_optimality(toy_loaded, stamp, backend='python')
    assert load_bks(result.path).metadata['optimality'] == stamp


@pytest.mark.parametrize('marker', ['standard', 'exact_evaluation'])
def test_legacy_reprice_refuses_exact_record_before_loading_artifacts(toy_loaded, marker):
    from mamut_routing_lib.td.reprice import reprice_td_bks

    bks = create_td_bks_from_solution(toy_loaded, _solution(toy_loaded, [[1], [2]]), authors=AUTHORS, backend='python')
    bks.metadata[marker] = bks.metadata.pop('exact_evaluation')
    path = get_bks_path_for_instance(toy_loaded.instance_path, bks.objective_function)
    write_bks_file(bks, path)
    original = path.read_bytes()
    for dry_run in (True, False):
        report = reprice_td_bks(path, dry_run=dry_run)
        assert report.action == 'error' and 'exact-profile' in report.detail
        assert path.read_bytes() == original

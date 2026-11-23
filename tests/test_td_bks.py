"""Tests for the TD-aware BKS helpers (Duration objective, TD checker refereed)."""

from __future__ import annotations

import pytest

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks, write_bks_file
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, OptimalityMetadata
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
    payload = {
        "proven": True,
        "prover": "test exact solver 1.0",
        "certificate": "optimal under test arithmetic",
        "checker": "td-fold/2",
        "date": "2026-07-07",
    }
    payload.update(overrides)
    return payload


def _store_legacy(loaded, solution):
    """Historical layouts and rounding are still readable and annotatable under their target."""
    price = check_td_solution(loaded, solution).routing_cost
    bks = BenchmarkBKS(instance_name=solution.instance_name, objective_function=ObjectiveFunction.DURATION,
                       routes=solution.routes, cost=price, metadata={'authors': AUTHORS})
    write_bks_file(bks, get_bks_path_for_instance(loaded.instance_path, ObjectiveFunction.DURATION))


def test_annotate_optimality_stamps_and_round_trips(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    cost = check_td_solution(toy_loaded, solution).routing_cost
    _store_legacy(toy_loaded, solution)

    stamp = _optimality(proven_optimum=cost, dual_bound=cost, wall_time_s=1.5, time_limit_s=600.0)
    bks_path = annotate_td_bks_optimality(toy_loaded, stamp)

    stored = load_bks(bks_path)
    assert stored.cost == cost  # untouched
    assert stored.routes == [[1, 2]]  # untouched
    assert stored.metadata["authors"] == AUTHORS  # finder credit untouched
    optimality = stored.metadata["optimality"]
    assert optimality["target_profile"] == "td-fold/2"
    assert optimality["proven"] is True
    assert optimality["prover"] == "test exact solver 1.0"
    assert optimality["proven_optimum"] == cost
    assert "note" not in optimality  # None fields are dropped on write


def test_annotate_optimality_accepts_model_instance(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    _store_legacy(toy_loaded, solution)
    bks_path = annotate_td_bks_optimality(toy_loaded, OptimalityMetadata(**_optimality()))
    assert load_bks(bks_path).metadata["optimality"]["proven"] is True


def test_annotate_optimality_requires_existing_bks(toy_loaded):
    with pytest.raises(FileNotFoundError, match="No stored Duration BKS"):
        annotate_td_bks_optimality(toy_loaded, _optimality())


def test_annotate_optimality_rejects_cost_mismatch(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    cost = check_td_solution(toy_loaded, solution).routing_cost
    _store_legacy(toy_loaded, solution)
    with pytest.raises(ValueError, match="does not match the stored"):
        annotate_td_bks_optimality(toy_loaded, _optimality(proven_optimum=cost + 1.0))


def test_annotate_optimality_dust_needs_note(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    cost = check_td_solution(toy_loaded, solution).routing_cost
    _store_legacy(toy_loaded, solution)

    dust_optimum = cost - 1e-9
    with pytest.raises(ValueError, match="note"):
        annotate_td_bks_optimality(toy_loaded, _optimality(proven_optimum=dust_optimum))

    bks_path = annotate_td_bks_optimality(
        toy_loaded,
        _optimality(proven_optimum=dust_optimum, note="stored cost differs by float summation dust"),
    )
    assert load_bks(bks_path).metadata["optimality"]["proven_optimum"] == dust_optimum


def test_annotate_optimality_rejects_malformed_stamp(toy_loaded):
    solution = _solution(toy_loaded, [[1, 2]])
    _store_legacy(toy_loaded, solution)
    with pytest.raises(ValueError):
        annotate_td_bks_optimality(toy_loaded, _optimality(proven=False))
    with pytest.raises(ValueError):
        annotate_td_bks_optimality(toy_loaded, _optimality(unexpected_key="x"))


def test_stamped_metadata_survives_save_if_improved(toy_loaded):
    # A later (non-improving) save attempt must not clobber the stamp.
    solution = _solution(toy_loaded, [[1, 2]])
    _store_legacy(toy_loaded, solution)
    annotate_td_bks_optimality(toy_loaded, _optimality())
    kept = save_td_solution_as_bks_if_improved(toy_loaded, solution, authors=AUTHORS)
    assert kept.action == "kept_existing"
    assert load_bks(kept.path).metadata["optimality"]["proven"] is True


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
    write_bks_file(incumbent, store)
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

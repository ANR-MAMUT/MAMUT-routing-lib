"""Create and compare TD incumbents under td-binary64-exact-v1."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks, write_bks_file
from mamut_routing_lib.bks import BKSUpdateAction, BKSUpdateResult
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.objectives import base_objective
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, SolverReportedOptimality
from mamut_routing_lib.td.artifacts import LoadedTDInstance, load_td_benchmark_instance, load_td_instance
from mamut_routing_lib.td.checker import TD_OBJECTIVES


def _as_loaded(instance: LoadedTDInstance | str | Path) -> LoadedTDInstance:
    return instance if isinstance(instance, LoadedTDInstance) else load_td_instance(instance)


def _td_objective(loaded, objective_function: ObjectiveFunction | None) -> ObjectiveFunction:
    if objective_function is None:
        instance = loaded.instance if isinstance(loaded, LoadedTDInstance) else load_td_benchmark_instance(loaded)
        return base_objective(instance)
    if objective_function not in TD_OBJECTIVES:
        raise ValueError(f'TD BKS store only supports {[o.value for o in TD_OBJECTIVES]}, got {objective_function!r}')
    return ObjectiveFunction(objective_function)


def _checker(loaded, objective_function, backend):
    # td.__init__ also exposes the BKS helpers while the exact TD module imports td models.
    from mamut_routing_lib.exact_td import prepare_td
    return prepare_td(loaded.instance, loaded.atfs, objective_function=objective_function, backend=backend)


def _candidate(loaded, checker, solution, authors, metadata, manifest=None):
    from mamut_routing_lib.exact_store import assert_candidate_metadata, evaluation_metadata
    from mamut_routing_lib.exact_types import ExactRouteError

    if not isinstance(authors, str) or not authors.strip():
        raise ValueError('authors must be a non-empty string')
    if solution.instance_name != loaded.instance.instance_name:
        raise ValueError('candidate identifies a different instance')
    if getattr(solution, 'objective_function', checker.objective) != checker.objective:
        raise ValueError('candidate declares a different objective')
    try:
        evaluation = checker.evaluate(solution.routes)
    except ExactRouteError as error:
        raise ValueError(f'Cannot create BKS from invalid TD solution: {error}') from error
    if solution.cost is not None and solution.cost != evaluation.display_cost:
        raise ValueError(f'Provided cost {solution.cost!r} does not match computed {checker.objective.value} '
                         f'{evaluation.display_cost!r}.')
    merged = dict(metadata or {})
    assert_candidate_metadata(solution, merged, evaluation, manifest)
    if 'optimality' in merged:
        claim = SolverReportedOptimality.model_validate(merged['optimality'])
        if claim.target_profile != evaluation.profile:
            raise ValueError('claim target_profile differs from the evaluated TD profile')
    merged.update(authors=authors, validated_cost=evaluation.display_cost, validated_num_routes=len(evaluation.routes),
                  **evaluation_metadata(evaluation, manifest))
    merged.setdefault('date', datetime.now(UTC).replace(tzinfo=None).isoformat(timespec='seconds'))
    bks = BenchmarkBKS(instance_name=loaded.instance.instance_name, objective_function=checker.objective,
                       routes=[list(route) for route in evaluation.routes], cost=evaluation.display_cost, metadata=merged)
    return bks, evaluation


def _published_inputs(loaded, objective, routes, *, allow_in_memory=False):
    from mamut_routing_lib.manifests import adjacent_manifest
    path = loaded.instance_path if isinstance(loaded, LoadedTDInstance) else loaded
    manifest = adjacent_manifest(path, objective, routes=routes,
                                 supplied_checkable=loaded if isinstance(loaded, LoadedTDInstance) else None)
    if manifest is not None:
        return manifest.checkable, manifest
    if isinstance(loaded, LoadedTDInstance) and not allow_in_memory:
        from mamut_routing_lib.td._preload import require_td_preload
        require_td_preload(loaded, load_td_benchmark_instance(path), Path(path))
    return _as_loaded(loaded), None


def create_td_bks_from_solution(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    objective_function: ObjectiveFunction | None = None,
    backend: str = 'native',
) -> BenchmarkBKS:
    """Price a candidate, check any declared cost and bind exact witnesses to canonical routes."""
    objective_function = _td_objective(loaded, objective_function)
    loaded, manifest = _published_inputs(loaded, objective_function, solution.routes, allow_in_memory=True)
    checker = _checker(loaded, objective_function, backend)
    return _candidate(loaded, checker, solution, authors, metadata, manifest)[0]


def save_td_solution_as_bks_if_improved(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    objective_function: ObjectiveFunction | None = None,
    backend: str = 'native',
) -> BKSUpdateResult:
    """Re-evaluate both route sets and store only a strict exact improvement.

    A legacy incumbent keeps its original bytes on ties and worse proposals. Replacing an
    incumbent with a claim first archives its original record. Writers serialize per instance.
    """
    from mamut_routing_lib.exact_store import preserve_claim_record, verify_stored

    instance_path = loaded.instance_path if isinstance(loaded, LoadedTDInstance) else loaded
    objective_function = _td_objective(loaded, objective_function)
    path = get_bks_path_for_instance(instance_path, objective_function)
    incumbent_bks = load_bks(path) if path.exists() else None
    all_routes = [*solution.routes, *incumbent_bks.routes] if incumbent_bks is not None else solution.routes
    loaded, manifest = _published_inputs(loaded, objective_function, all_routes)
    checker = _checker(loaded, objective_function, backend)
    candidate_bks, candidate_evaluation = _candidate(loaded, checker, solution, authors, metadata, manifest)
    previous = None
    tie = False
    action: BKSUpdateAction = 'created'
    if incumbent_bks is not None:
        previous = path
        if incumbent_bks.instance_name != candidate_bks.instance_name or incumbent_bks.objective_function != checker.objective:
            raise ValueError(f'Stored BKS identifies a different instance or objective at {path}')
        try:
            incumbent_evaluation = checker.evaluate(incumbent_bks.routes)
            verify_stored(incumbent_bks, incumbent_evaluation, manifest)
        except ValueError as error:
            raise ValueError(f'Stored BKS is invalid at {path}: {error}') from error
        candidate_objective_key = candidate_evaluation.objective_key
        incumbent_objective_key = incumbent_evaluation.objective_key
        tie = candidate_objective_key == incumbent_objective_key
        if candidate_objective_key < incumbent_objective_key:
            preserve_claim_record(path, incumbent_bks)
            action = 'replaced'
        else:
            action = 'kept_existing'
    if action != 'kept_existing':
        write_bks_file(candidate_bks, path)
    return BKSUpdateResult(action=action, path=path, previous_path=previous, tie=tie,
                           candidate_cost=candidate_evaluation.display_cost, candidate_num_routes=len(candidate_evaluation.routes))


def annotate_td_bks_optimality(
    loaded: LoadedTDInstance | str | Path,
    optimality: SolverReportedOptimality | dict[str, Any],
    objective_function: ObjectiveFunction | None = None,
    *,
    backend: str = 'native',
) -> Path:
    """Attach a solver claim of optimality to the existing incumbent under the declared objective.

    The claim must be a ``SolverReportedOptimality`` whose ``target_profile`` is the
    TD profile, and the incumbent must carry an exact evaluation that still verifies.
    The legacy ``proven: true`` stamp is refused. Reported bounds do not certify the incumbent.
    """
    from mamut_routing_lib.exact_store import preserve_claim_record, verify_stored
    from mamut_routing_lib.exact_types import TD_PROFILE

    objective_function = _td_objective(loaded, objective_function)
    instance_path = loaded.instance_path if isinstance(loaded, LoadedTDInstance) else loaded
    path = get_bks_path_for_instance(instance_path, objective_function)
    if not path.exists():
        raise FileNotFoundError(f'No stored {objective_function.value} BKS to annotate at {path}')
    bks = load_bks(path)
    loaded, manifest = _published_inputs(loaded, objective_function, bks.routes)
    if bks.instance_name != loaded.instance.instance_name or bks.objective_function != objective_function:
        raise ValueError(f'Stored BKS identifies a different instance or objective at {path}')
    payload = optimality if isinstance(optimality, dict) else optimality.model_dump(mode='json', exclude_none=True)
    claim = SolverReportedOptimality.model_validate(payload)
    if claim.target_profile != TD_PROFILE:
        raise ValueError('solver claim must target the profile of the stored TD result')
    if 'exact_evaluation' not in bks.metadata and 'standard' not in bks.metadata:
        raise ValueError(f'A solver claim requires an incumbent evaluated under {TD_PROFILE} at {path}')
    checker = _checker(loaded, objective_function, backend)
    verify_stored(bks, checker.evaluate(bks.routes), manifest)
    record = claim.model_dump(mode='json', exclude_none=True)
    if record != bks.metadata.get('optimality'):
        stamped = BenchmarkBKS(**{**bks.model_dump(), 'metadata': {**bks.metadata, 'optimality': record}})
        preserve_claim_record(path, bks)
        write_bks_file(stamped, path)
    return path

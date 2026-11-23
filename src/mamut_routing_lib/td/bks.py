"""Create and compare TD incumbents under td-binary64-exact-v1."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks, write_bks_file
from mamut_routing_lib.bks import BKSUpdateResult
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, OptimalityMetadata, SolverReportedOptimality
from mamut_routing_lib.td.artifacts import LoadedTDInstance, load_td_instance
from mamut_routing_lib.td.checker import TD_OBJECTIVES


def _as_loaded(instance: LoadedTDInstance | str | Path) -> LoadedTDInstance:
    return instance if isinstance(instance, LoadedTDInstance) else load_td_instance(instance)


def _validate_td_objective(objective_function: ObjectiveFunction) -> ObjectiveFunction:
    if objective_function not in TD_OBJECTIVES:
        raise ValueError(f'TD BKS store only supports {[o.value for o in TD_OBJECTIVES]}, got {objective_function!r}')
    return ObjectiveFunction(objective_function)


def _checker(loaded, objective_function, backend):
    # td.__init__ also exposes the BKS helpers while the exact TD module imports td models.
    from mamut_routing_lib.exact_td import prepare_td
    return prepare_td(loaded.instance, loaded.atfs, objective_function=_validate_td_objective(objective_function), backend=backend)


def _candidate(loaded, checker, solution, authors, metadata):
    from mamut_routing_lib.exact_store import standard_metadata, verify_stored
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
    if 'standard' in merged or 'standard' in solution.metadata:
        raise ValueError('cannot write metadata.standard without verifying its family manifest')
    if 'exact_evaluation' in solution.metadata:
        if 'exact_evaluation' in merged and merged['exact_evaluation'] != solution.metadata['exact_evaluation']:
            raise ValueError('conflicting supplied exact evaluation records')
        merged['exact_evaluation'] = solution.metadata['exact_evaluation']
    if 'exact_evaluation' in merged:
        source = BenchmarkBKS(instance_name=solution.instance_name, routes=solution.routes, cost=solution.cost,
                              objective_function=checker.objective, metadata={**merged, 'authors': authors})
        verify_stored(source, evaluation)
    if 'optimality' in merged:
        claim = SolverReportedOptimality.model_validate(merged['optimality'])
        if claim.target_profile != evaluation.profile:
            raise ValueError('claim target_profile differs from the evaluated TD profile')
    merged.update(authors=authors, validated_cost=evaluation.display_cost, validated_num_routes=len(evaluation.routes),
                  exact_evaluation=standard_metadata(evaluation))
    merged.setdefault('date', datetime.now(UTC).replace(tzinfo=None).isoformat(timespec='seconds'))
    bks = BenchmarkBKS(instance_name=loaded.instance.instance_name, objective_function=checker.objective,
                       routes=[list(route) for route in evaluation.routes], cost=evaluation.display_cost, metadata=merged)
    return bks, evaluation


def create_td_bks_from_solution(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
    backend: str = 'native',
) -> BenchmarkBKS:
    """Price a candidate, check any declared cost and bind exact witnesses to canonical routes."""
    loaded = _as_loaded(loaded)
    checker = _checker(loaded, objective_function, backend)
    return _candidate(loaded, checker, solution, authors, metadata)[0]


def save_td_solution_as_bks_if_improved(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
    backend: str = 'native',
) -> BKSUpdateResult:
    """Re-evaluate both route sets and store only a strict exact improvement.

    A legacy incumbent keeps its original bytes on ties and worse proposals. Replacing an
    incumbent with a claim first archives its original record. Writers serialize per instance.
    """
    from mamut_routing_lib.exact_store import preserve_claim_record, verify_stored

    loaded = _as_loaded(loaded)
    checker = _checker(loaded, objective_function, backend)
    candidate_bks, candidate_evaluation = _candidate(loaded, checker, solution, authors, metadata)
    path = get_bks_path_for_instance(loaded.instance_path, checker.objective)
    previous = None
    tie = False
    action = 'created'
    if path.exists():
        previous = path
        incumbent_bks = load_bks(path)
        if incumbent_bks.instance_name != candidate_bks.instance_name or incumbent_bks.objective_function != checker.objective:
            raise ValueError(f'Stored BKS identifies a different instance or objective at {path}')
        try:
            incumbent_evaluation = checker.evaluate(incumbent_bks.routes)
            verify_stored(incumbent_bks, incumbent_evaluation)
        except ValueError as error:
            raise ValueError(f'Stored BKS is invalid at {path}: {error}') from error
        candidate_objective_key = candidate_evaluation.objective_key
        incumbent_objective_key = incumbent_evaluation.objective_key
        tie = candidate_objective_key == incumbent_objective_key
        if candidate_objective_key < incumbent_objective_key:
            if 'standard' in incumbent_bks.metadata:
                raise ValueError('cannot replace a metadata.standard incumbent until its family manifest can be verified')
            preserve_claim_record(path, incumbent_bks)
            action = 'replaced'
        else:
            action = 'kept_existing'
    if action != 'kept_existing':
        write_bks_file(candidate_bks, path)
    return BKSUpdateResult(action=action, path=path, previous_path=previous, tie=tie,
                           candidate_cost=candidate_evaluation.display_cost, candidate_num_routes=len(candidate_evaluation.routes))


# Historical annotations compare solver-reported numbers with a rounded legacy cost.
OPTIMALITY_COST_TOLERANCE = 1e-6


def annotate_td_bks_optimality(
    loaded: LoadedTDInstance | str | Path,
    optimality: OptimalityMetadata | SolverReportedOptimality | dict[str, Any],
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
    *,
    backend: str = 'native',
) -> Path:
    """Attach a targeted solver claim to the existing incumbent under the declared objective.

    Legacy stamps require the original checker or arithmetic and an unmarked record. The
    historical numeric consistency guard applies only to those rounded records. A new-profile
    claim identifies the solver and its target. Its reported bounds do not certify the incumbent.
    """
    from mamut_routing_lib.exact_store import preserve_claim_record, verify_stored
    from mamut_routing_lib.exact_types import TD_PROFILE

    loaded = _as_loaded(loaded)
    objective_function = _validate_td_objective(objective_function)
    path = get_bks_path_for_instance(loaded.instance_path, objective_function)
    if not path.exists():
        raise FileNotFoundError(f'No stored {objective_function.value} BKS to annotate at {path}')
    bks = load_bks(path)
    if bks.instance_name != loaded.instance.instance_name or bks.objective_function != objective_function:
        raise ValueError(f'Stored BKS identifies a different instance or objective at {path}')
    payload = optimality if isinstance(optimality, dict) else optimality.model_dump(mode='json', exclude_none=True)
    if 'standard' in bks.metadata:
        raise ValueError('cannot rewrite metadata.standard until its family manifest can be verified')
    if 'exact_evaluation' in bks.metadata:
        claim = SolverReportedOptimality.model_validate(payload)
        if claim.target_profile != TD_PROFILE:
            raise ValueError('solver claim must target the profile of the stored TD result')
        checker = _checker(loaded, objective_function, backend)
        verify_stored(bks, checker.evaluate(bks.routes))
    else:
        claim = OptimalityMetadata.model_validate(payload)
        target = claim.target_profile
        if target is None and claim.checker in ('td-fold/1', 'td-fold/2'):
            target = claim.checker
        if not target or target == TD_PROFILE:
            raise ValueError('legacy annotation must identify its original historical target_profile')
        claim = claim.model_copy(update={'target_profile': target})
        if claim.proven_optimum is not None:
            if bks.cost is None:
                raise ValueError('cannot compare a reported historical optimum with a missing stored cost')
            deviation = abs(claim.proven_optimum - bks.cost)
            if deviation > OPTIMALITY_COST_TOLERANCE:
                raise ValueError(f'proven_optimum {claim.proven_optimum!r} does not match the stored cost at {path}')
            if deviation != 0 and not claim.note:
                raise ValueError('a note is required to explain a historical solver/checker cost difference')
    record = claim.model_dump(mode='json', exclude_none=True)
    if record != bks.metadata.get('optimality'):
        stamped = BenchmarkBKS(**{**bks.model_dump(), 'metadata': {**bks.metadata, 'optimality': record}})
        preserve_claim_record(path, bks)
        write_bks_file(stamped, path)
    return path

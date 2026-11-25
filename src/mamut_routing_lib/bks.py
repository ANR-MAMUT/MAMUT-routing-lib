"""Reprice and store static incumbents using the exact binary64 objective.

An existing record keeps its place on an exact tie. Historical displayed
costs can come from earlier arithmetic rules, so compare their submitted
routes again on the current instance. Callers serialize writers per instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

from mamut_routing_lib.artifacts import (
    get_bks_path_for_instance,
    get_instance_identifier,
    hydrate_collection_instance,
    load_benchmark_instance,
    load_bks,
    write_bks_file,
)
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_static import PreparedStatic, prepare_static
from mamut_routing_lib.exact_store import assert_candidate_metadata, evaluation_metadata, preserve_claim_record, verify_stored
from mamut_routing_lib.exact_types import ExactBackend, ExactEvaluation, ExactRouteError
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, LoadedStaticInstance, SolverReportedOptimality

DEFAULT_BKS_AUTHORS = "Florian Rascoussier (0nyr) and Adrien Pichon (Anzury)"

BKSUpdateAction = Literal["created", "replaced", "kept_existing"]
"""What a store write did: wrote a first record, replaced a worse one, or kept the incumbent."""

BKSPreviewAction = Literal["would_create", "would_replace", "would_keep", "invalid", "stored_invalid"]
"""What ``preview_bks_update`` reports a store write would do, without writing."""


@dataclass(frozen=True)
class BKSUpdateResult:
    """The candidate's display cost and route count, even when the incumbent wins.

    A display overflow gives None. If an incumbent existed, previous_path is
    its store path, including when that path now holds the replacement.
    ``previous_path`` therefore does not identify a backup of the replaced
    record. Only an incumbent that carries an optimality claim is archived,
    under ``.bks-history/`` next to it. Any other replaced record is gone.
    """

    action: BKSUpdateAction
    path: Path
    previous_path: Path | None
    candidate_cost: int | float | None
    candidate_num_routes: int
    tie: bool = False

    def __post_init__(self) -> None:
        if self.action not in get_args(BKSUpdateAction):
            raise ValueError(f"Unknown BKS update action {self.action!r}")


def _candidate(
    prepared: PreparedStatic,
    name: str,
    solution: BenchmarkSolution | BenchmarkBKS,
    authors: str,
    metadata: dict[str, Any] | None,
    manifest=None,
) -> tuple[BenchmarkBKS, ExactEvaluation]:
    if solution.instance_name != name:
        raise ValueError(f"Candidate instance_name {solution.instance_name!r} does not match the instance {name!r}")
    if not isinstance(authors, str) or not authors.strip():
        raise ValueError("BKS metadata.authors must be a non-empty string")
    try:
        evaluation = prepared.evaluate(solution.routes)
    except ExactRouteError as exc:
        raise ValueError(f"Candidate BKS is invalid: {exc}") from exc
    if solution.cost is not None and solution.cost != evaluation.display_cost:
        raise ValueError("Candidate BKS is invalid: provided cost does not match the exact cost's binary64 projection")

    merged = dict(metadata or {})
    assert_candidate_metadata(solution, merged, evaluation, manifest)
    if "optimality" in merged:
        claim = SolverReportedOptimality.model_validate(merged["optimality"])
        if claim.target_profile != evaluation.profile:
            raise ValueError("Claim target_profile differs from the evaluated static profile")
    merged.update(authors=authors, validated_cost=evaluation.display_cost, validated_num_routes=len(evaluation.routes),
                  **evaluation_metadata(evaluation, manifest))
    merged.setdefault("date", datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"))
    return BenchmarkBKS(
        instance_name=name, objective_function=prepared.objective,
        routes=[list(route) for route in evaluation.routes], cost=evaluation.display_cost, metadata=merged,
    ), evaluation


def create_bks_from_solution(
    instance: LoadedStaticInstance,
    solution: BenchmarkSolution,
    objective_function: ObjectiveFunction,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    backend: ExactBackend = "native",
) -> BenchmarkBKS:
    """Create a canonical exact BKS from a solution and a materialized instance."""
    prepared = prepare_static(instance, backend=backend, objective_function=objective_function)
    candidate_bks, _ = _candidate(prepared, get_instance_identifier(instance), solution, authors, metadata)
    return candidate_bks


def _store(
    path: Path, prepared: PreparedStatic, candidate_bks: BenchmarkBKS, candidate_evaluation: ExactEvaluation, manifest=None,
) -> BKSUpdateResult:
    previous = None
    action: BKSUpdateAction = "created"
    tie = False
    if path.exists():
        previous = path
        incumbent_bks = load_bks(path)
        if (incumbent_bks.instance_name != candidate_bks.instance_name
                or incumbent_bks.objective_function != candidate_bks.objective_function):
            raise ValueError(f"Stored BKS has a different instance or objective at {path}")
        try:
            incumbent_evaluation = prepared.evaluate(incumbent_bks.routes)
            verify_stored(incumbent_bks, incumbent_evaluation, manifest)
        except ValueError as exc:
            raise ValueError(f"Stored BKS is invalid at {path}: {exc}") from exc
        candidate_objective_key = candidate_evaluation.objective_key
        incumbent_objective_key = incumbent_evaluation.objective_key
        if candidate_objective_key < incumbent_objective_key:
            preserve_claim_record(path, incumbent_bks)
            action = "replaced"
        else:
            tie = candidate_objective_key == incumbent_objective_key
            action = "kept_existing"
    if action != "kept_existing":
        write_bks_file(candidate_bks, path)
    return BKSUpdateResult(action, path, previous, candidate_evaluation.display_cost, len(candidate_evaluation.routes), tie)


def save_bks_if_improved(
    instance: LoadedStaticInstance,
    candidate_bks: BenchmarkBKS,
    instance_path: str | Path,
    *,
    backend: ExactBackend = "native",
) -> BKSUpdateResult:
    """Recheck the candidate and incumbent and replace only on a strict improvement.

    Without a manifest, the supplied instance must equal the descriptor (and its
    hydrated matrix) stored at ``instance_path``. Otherwise this raises ``ValueError``.
    """
    from mamut_routing_lib.manifests import adjacent_manifest
    manifest = adjacent_manifest(instance_path, candidate_bks.objective_function, supplied_checkable=instance)
    if manifest is not None:
        instance = manifest.checkable
    elif instance != hydrate_collection_instance(load_benchmark_instance(instance_path), instance_path):
        raise ValueError(f"Supplied materialized input differs from the descriptor or matrix at {instance_path}")
    prepared = prepare_static(instance, backend=backend, objective_function=candidate_bks.objective_function)
    candidate_bks, candidate_evaluation = _candidate(
        prepared, get_instance_identifier(instance), candidate_bks, candidate_bks.metadata.get("authors"),
        candidate_bks.metadata, manifest,
    )
    return _store(get_bks_path_for_instance(instance_path, prepared.objective), prepared, candidate_bks, candidate_evaluation, manifest)


def save_solution_as_bks_if_improved(
    instance_path: str | Path,
    objective_function: ObjectiveFunction,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    backend: ExactBackend = "native",
    supplied_checkable: LoadedStaticInstance | None = None,
) -> BKSUpdateResult:
    """Load and prepare the instance once, check both submissions and store the winner."""
    from mamut_routing_lib.manifests import adjacent_manifest
    manifest = adjacent_manifest(instance_path, objective_function, supplied_checkable=supplied_checkable)
    instance = manifest.checkable if manifest else hydrate_collection_instance(load_benchmark_instance(instance_path), instance_path)
    if manifest is None and supplied_checkable is not None and supplied_checkable != instance:
        raise ValueError(f"Supplied materialized input differs from the descriptor or matrix at {instance_path}")
    prepared = prepare_static(instance, backend=backend, objective_function=objective_function)
    candidate_bks, candidate_evaluation = _candidate(
        prepared, get_instance_identifier(instance), solution, authors, metadata, manifest,
    )
    return _store(get_bks_path_for_instance(instance_path, prepared.objective), prepared, candidate_bks, candidate_evaluation, manifest)

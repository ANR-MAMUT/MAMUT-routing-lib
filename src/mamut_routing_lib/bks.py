"""Reprice and store static incumbents using the exact binary64 objective.

An existing record keeps its place on an exact tie. Historical displayed
costs can come from earlier arithmetic rules, so compare their submitted
routes again on the current instance. Callers serialize writers per instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_lib.artifacts import (
    AnyBenchmarkInstance,
    get_bks_path_for_instance,
    get_instance_identifier,
    hydrate_collection_instance,
    load_benchmark_instance,
    load_bks,
    save_bks,
)
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_static import PreparedStatic, prepare_static
from mamut_routing_lib.exact_store import preserve_claim_record, standard_metadata, verify_stored
from mamut_routing_lib.exact_types import ExactBackend, ExactEvaluation, ExactRouteError
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, SolverReportedOptimality

DEFAULT_BKS_AUTHORS = "Florian Rascoussier (0nyr) and Adrien Pichon (Anzury)"


@dataclass(frozen=True)
class BKSUpdateResult:
    """Outcome and candidate display cost. Exact display overflow gives None."""

    action: str
    path: Path
    previous_path: Path | None
    candidate_cost: int | float | None
    candidate_num_routes: int
    tie: bool = False


def _candidate(
    prepared: PreparedStatic,
    name: str,
    solution: BenchmarkSolution | BenchmarkBKS,
    authors: str,
    metadata: dict[str, Any] | None,
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
    if "standard" in solution.metadata or "standard" in merged:
        raise ValueError("Cannot verify metadata.standard without an authenticated family manifest")
    if "exact_evaluation" in solution.metadata and "exact_evaluation" in merged:
        if solution.metadata["exact_evaluation"] != merged["exact_evaluation"]:
            raise ValueError("Conflicting exact evaluation assertions for the submitted solution")
    if "exact_evaluation" in solution.metadata or "exact_evaluation" in merged:
        assertion = BenchmarkBKS(instance_name=name, objective_function=prepared.objective, routes=solution.routes,
                                 cost=solution.cost, metadata={**solution.metadata, **merged})
        verify_stored(assertion, evaluation)
    if "optimality" in merged:
        claim = SolverReportedOptimality.model_validate(merged["optimality"])
        if claim.target_profile != evaluation.profile:
            raise ValueError("Claim target_profile differs from the evaluated static profile")
    merged.update(authors=authors, validated_cost=evaluation.display_cost, validated_num_routes=len(evaluation.routes),
                  exact_evaluation=standard_metadata(evaluation))
    merged.setdefault("date", datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"))
    return BenchmarkBKS(
        instance_name=name, objective_function=prepared.objective,
        routes=[list(route) for route in evaluation.routes], cost=evaluation.display_cost, metadata=merged,
    ), evaluation


def create_bks_from_solution(
    instance: AnyBenchmarkInstance,
    solution: BenchmarkSolution,
    objective_function: ObjectiveFunction,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    backend: ExactBackend = "native",
) -> BenchmarkBKS:
    """Create a canonical exact BKS from a solution and a materialized instance."""
    prepared = prepare_static(instance, backend=backend, objective_function=objective_function)
    candidate, _ = _candidate(prepared, get_instance_identifier(instance), solution, authors,
                              metadata)
    return candidate


def _store(path: Path, prepared: PreparedStatic, candidate: BenchmarkBKS, evaluation: ExactEvaluation) -> BKSUpdateResult:
    previous = None
    action = "created"
    tie = False
    if path.exists():
        previous = path
        existing = load_bks(path)
        if existing.instance_name != candidate.instance_name or existing.objective_function != candidate.objective_function:
            raise ValueError(f"Stored BKS has a different instance or objective at {path}")
        try:
            incumbent = prepared.evaluate(existing.routes)
            verify_stored(existing, incumbent)
        except ValueError as exc:
            raise ValueError(f"Stored BKS is invalid at {path}: {exc}") from exc
        if evaluation.objective_key < incumbent.objective_key:
            if "standard" in existing.metadata:
                raise ValueError("Cannot replace a manifest-bound incumbent without verifying its family manifest")
            preserve_claim_record(path, existing)
            action = "replaced"
        else:
            tie = evaluation.objective_key == incumbent.objective_key
            action = "kept_existing"
    if action != "kept_existing":
        save_bks(candidate, path)
    return BKSUpdateResult(action, path, previous, evaluation.display_cost, len(evaluation.routes), tie)


def save_bks_if_improved(
    instance: AnyBenchmarkInstance,
    bks: BenchmarkBKS,
    instance_path: str | Path,
    *,
    backend: ExactBackend = "native",
) -> BKSUpdateResult:
    """Recheck the candidate and incumbent and replace only on a strict improvement."""
    prepared = prepare_static(instance, backend=backend, objective_function=bks.objective_function)
    candidate, evaluation = _candidate(prepared, get_instance_identifier(instance), bks,
                                       bks.metadata.get("authors"), bks.metadata)
    return _store(get_bks_path_for_instance(instance_path, prepared.objective), prepared, candidate, evaluation)


def save_solution_as_bks_if_improved(
    instance_path: str | Path,
    objective_function: ObjectiveFunction,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    backend: ExactBackend = "native",
) -> BKSUpdateResult:
    """Load and prepare the instance once, check both submissions and store the winner."""
    instance = hydrate_collection_instance(load_benchmark_instance(instance_path), instance_path)
    prepared = prepare_static(instance, backend=backend, objective_function=objective_function)
    candidate, evaluation = _candidate(prepared, get_instance_identifier(instance), solution, authors,
                                       metadata)
    return _store(get_bks_path_for_instance(instance_path, prepared.objective), prepared, candidate, evaluation)

"""Creating and replacing best-known solutions (static families).

Two guarantees: the stored ``cost`` is always the checker's value
(``create_bks_from_solution`` re-prices and raises on an invalid solution),
and a stored BKS is replaced only by a strictly better one under its
objective, after the stored file was itself re-validated
(``save_bks_if_improved``). The store is a plain read/compare/write with no
lock: concurrent writers for the same instance must be serialized by the
caller (one coordinator per instance), otherwise a worse candidate can win
the race. Time-dependent families use ``mamut_routing_lib.td.bks``.
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
from mamut_routing_lib.checker import check_solution, is_better_solution
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution

DEFAULT_BKS_AUTHORS = "Florian Rascoussier (0nyr) and Adrien Pichon (Anzury)"


@dataclass(frozen=True)
class BKSUpdateResult:
    """What a store call did: ``action`` is ``created``, ``replaced`` or ``kept_existing``; ``path`` is the BKS file."""
    action: str
    path: Path
    previous_path: Path | None
    candidate_cost: int | float
    candidate_num_routes: int


def create_bks_from_solution(
    instance: AnyBenchmarkInstance,
    solution: BenchmarkSolution,
    objective_function: ObjectiveFunction,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkBKS:
    """Validate ``solution`` on ``instance`` and wrap it as a ``BenchmarkBKS``.

    Raises ``ValueError`` when the checker rejects the solution or ``authors``
    is empty. The BKS ``cost`` is the checker's value; ``metadata`` gets
    ``authors``, ``validated_cost``, ``validated_num_routes`` and an ISO ``date``
    (UTC) unless already present. ``instance`` must be an embedded-matrix model
    (hydrate collection instances first).
    """
    check_result = check_solution(instance, solution)
    if not check_result.is_valid():
        raise ValueError(f"Cannot create BKS from invalid solution: {check_result.error_message}")

    if not authors.strip():
        raise ValueError("authors must be a non-empty string")

    merged_metadata = dict(metadata or {})
    merged_metadata["authors"] = authors
    merged_metadata.setdefault("validated_cost", check_result.routing_cost)
    merged_metadata.setdefault("validated_num_routes", check_result.num_routes)
    merged_metadata.setdefault("date", datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"))
    return BenchmarkBKS(
        instance_name=get_instance_identifier(instance),
        objective_function=objective_function,
        routes=solution.routes,
        cost=check_result.routing_cost,
        metadata=merged_metadata,
    )


def save_bks_if_improved(
    instance: AnyBenchmarkInstance,
    bks: BenchmarkBKS,
    instance_path: str | Path,
) -> BKSUpdateResult:
    """Write ``bks`` next to ``instance_path`` unless a strictly better BKS is stored there.

    The existing file, if any, is re-validated on ``instance`` (an invalid
    stored BKS raises rather than being silently overwritten), then compared
    with ``checker.is_better_solution`` under ``bks.objective_function``.
    """
    authors = bks.metadata.get("authors")
    if not isinstance(authors, str) or not authors.strip():
        raise ValueError("BKS metadata.authors is required")

    bks_path = get_bks_path_for_instance(instance_path, bks.objective_function)
    existing_path = bks_path if bks_path.exists() else None

    if existing_path is None:
        save_bks(bks, bks_path)
        return BKSUpdateResult(
            action="created",
            path=bks_path,
            previous_path=None,
            candidate_cost=bks.cost if bks.cost is not None else float("inf"),
            candidate_num_routes=bks.num_routes,
        )

    existing_bks = load_bks(existing_path)
    existing_check = check_solution(instance, existing_bks)
    if not existing_check.is_valid():
        raise ValueError(f"Stored BKS is invalid at {existing_path}: {existing_check.error_message}")

    candidate_cost = bks.cost if bks.cost is not None else float("inf")
    existing_cost = existing_bks.cost if existing_bks.cost is not None else float("inf")
    if is_better_solution(
        bks.routes,
        candidate_cost,
        existing_bks.routes,
        existing_cost,
        bks.objective_function,
    ):
        save_bks(bks, bks_path)
        return BKSUpdateResult(
            action="replaced",
            path=bks_path,
            previous_path=existing_path,
            candidate_cost=candidate_cost,
            candidate_num_routes=bks.num_routes,
        )

    return BKSUpdateResult(
        action="kept_existing",
        path=bks_path,
        previous_path=existing_path,
        candidate_cost=candidate_cost,
        candidate_num_routes=bks.num_routes,
    )


def save_solution_as_bks_if_improved(
    instance_path: str | Path,
    objective_function: ObjectiveFunction,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
) -> BKSUpdateResult:
    """Load, hydrate, validate and store in one call.

    Loads the instance at ``instance_path`` (slim collection instances are
    hydrated from their sidecars), builds the BKS with
    ``create_bks_from_solution`` and applies ``save_bks_if_improved``. Raises
    ``TypeError`` for time-dependent instances.
    """
    instance = hydrate_collection_instance(load_benchmark_instance(instance_path), instance_path)
    bks = create_bks_from_solution(
        instance,
        solution,
        objective_function,
        authors=authors,
        metadata=metadata,
    )
    return save_bks_if_improved(instance, bks, instance_path)

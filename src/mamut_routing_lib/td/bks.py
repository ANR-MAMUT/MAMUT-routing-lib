"""TD-aware best-known-solution helpers.

The generic :mod:`mamut_routing_lib.bks` helpers go through the static
``check_solution``, which refuses TD instances. These mirrors validate and
price candidates with the canonical TD checker (:func:`check_td_solution`)
instead. The objective is always :attr:`ObjectiveFunction.DURATION` — the only
objective the TD standard stores BKS for — and the stored cost is always the
checker's canonical value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mamut_routing_lib.artifacts import get_bks_path_for_instance, load_bks, save_bks
from mamut_routing_lib.bks import BKSUpdateResult
from mamut_routing_lib.checker import is_better_solution
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution
from mamut_routing_lib.td.artifacts import LoadedTDInstance, load_td_instance
from mamut_routing_lib.td.checker import check_td_solution


def _as_loaded(instance: LoadedTDInstance | str | Path) -> LoadedTDInstance:
    if isinstance(instance, LoadedTDInstance):
        return instance
    return load_td_instance(instance)


def create_td_bks_from_solution(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkBKS:
    """Validate ``solution`` with the TD checker and wrap it as a Duration BKS.

    The BKS cost is the checker's canonical value. If ``solution.cost`` is set,
    ``check_td_solution`` itself rejects any deviation from that value (exact
    doubles, no tolerance), so a solver that disagrees with the reference
    checker can never write to the BKS store.
    """
    loaded = _as_loaded(loaded)
    check_result = check_td_solution(loaded, solution)
    if not check_result.is_valid():
        raise ValueError(f"Cannot create BKS from invalid TD solution: {check_result.error_message}")

    if not authors.strip():
        raise ValueError("authors must be a non-empty string")

    merged_metadata = dict(metadata or {})
    merged_metadata["authors"] = authors
    merged_metadata.setdefault("validated_cost", check_result.routing_cost)
    merged_metadata.setdefault("validated_num_routes", check_result.num_routes)
    merged_metadata.setdefault("date", datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"))
    return BenchmarkBKS(
        instance_name=loaded.instance.instance_name,
        objective_function=ObjectiveFunction.DURATION,
        routes=solution.routes,
        cost=check_result.routing_cost,
        metadata=merged_metadata,
    )


def save_td_solution_as_bks_if_improved(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
) -> BKSUpdateResult:
    """TD counterpart of ``save_solution_as_bks_if_improved`` (Duration only).

    The candidate and any stored BKS are both validated by the TD checker; the
    stored file is replaced only on a strict Duration improvement. Accepts a
    :class:`LoadedTDInstance` directly so callers that already hold the loaded
    instance (solvers, sweep runners) avoid re-reading the ATF sidecar.
    """
    loaded = _as_loaded(loaded)
    bks = create_td_bks_from_solution(loaded, solution, authors=authors, metadata=metadata)
    bks_path = get_bks_path_for_instance(loaded.instance_path, ObjectiveFunction.DURATION)
    existing_path = bks_path if bks_path.exists() else None

    candidate_cost = bks.cost if bks.cost is not None else float("inf")

    if existing_path is None:
        save_bks(bks, bks_path)
        return BKSUpdateResult(
            action="created",
            path=bks_path,
            previous_path=None,
            candidate_cost=candidate_cost,
            candidate_num_routes=bks.num_routes,
        )

    existing_bks = load_bks(existing_path)
    existing_check = check_td_solution(loaded, existing_bks)
    if not existing_check.is_valid():
        raise ValueError(f"Stored BKS is invalid at {existing_path}: {existing_check.error_message}")

    existing_cost = existing_bks.cost if existing_bks.cost is not None else float("inf")
    if is_better_solution(
        bks.routes,
        candidate_cost,
        existing_bks.routes,
        existing_cost,
        ObjectiveFunction.DURATION,
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

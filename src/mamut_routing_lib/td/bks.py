"""TD-aware best-known-solution helpers.

The generic :mod:`mamut_routing_lib.bks` helpers go through the static
``check_solution``, which refuses TD instances. These mirrors validate and
price candidates with the canonical TD checker (:func:`check_td_solution`)
instead. The objective defaults to :attr:`ObjectiveFunction.DURATION`;
families whose contract is :attr:`ObjectiveFunction.FLEET_COST_DURATION`
(Blauth2024) pass it explicitly. Either way the stored cost is always the
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
from mamut_routing_lib.models import BenchmarkBKS, BenchmarkSolution, OptimalityMetadata
from mamut_routing_lib.td.artifacts import LoadedTDInstance, load_td_instance
from mamut_routing_lib.td.checker import TD_OBJECTIVES, check_td_solution


def _as_loaded(instance: LoadedTDInstance | str | Path) -> LoadedTDInstance:
    if isinstance(instance, LoadedTDInstance):
        return instance
    return load_td_instance(instance)


def _validate_td_objective(objective_function: ObjectiveFunction) -> ObjectiveFunction:
    if objective_function not in TD_OBJECTIVES:
        raise ValueError(
            f"TD BKS store only supports {[o.value for o in TD_OBJECTIVES]}, got {objective_function!r}"
        )
    return objective_function


def create_td_bks_from_solution(
    loaded: LoadedTDInstance | str | Path,
    solution: BenchmarkSolution,
    *,
    authors: str,
    metadata: dict[str, Any] | None = None,
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
) -> BenchmarkBKS:
    """Validate ``solution`` with the TD checker and wrap it as a BKS.

    The BKS cost is the checker's canonical value under
    ``objective_function``. If ``solution.cost`` is set, ``check_td_solution``
    itself rejects any deviation from that value (exact doubles, no
    tolerance), so a solver that disagrees with the reference checker can
    never write to the BKS store.
    """
    loaded = _as_loaded(loaded)
    _validate_td_objective(objective_function)
    check_result = check_td_solution(loaded, solution, objective_function)
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
        objective_function=objective_function,
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
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
) -> BKSUpdateResult:
    """TD counterpart of ``save_solution_as_bks_if_improved``.

    The candidate and any stored BKS are both validated by the TD checker
    under ``objective_function``; the stored file is replaced only on a strict
    improvement. Each objective has its own store file
    (``<Name>.bks.<Objective>.json``). Accepts a :class:`LoadedTDInstance`
    directly so callers that already hold the loaded instance (solvers, sweep
    runners) avoid re-reading the ATF sidecar.
    """
    loaded = _as_loaded(loaded)
    _validate_td_objective(objective_function)
    bks = create_td_bks_from_solution(
        loaded, solution, authors=authors, metadata=metadata, objective_function=objective_function
    )
    bks_path = get_bks_path_for_instance(loaded.instance_path, objective_function)
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
    existing_check = check_td_solution(loaded, existing_bks, objective_function)
    if not existing_check.is_valid():
        raise ValueError(f"Stored BKS is invalid at {existing_path}: {existing_check.error_message}")

    existing_cost = existing_bks.cost if existing_bks.cost is not None else float("inf")
    if is_better_solution(
        bks.routes,
        candidate_cost,
        existing_bks.routes,
        existing_cost,
        objective_function,
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


#: Tolerance for the ``proven_optimum``-vs-stored-cost consistency guard in
#: :func:`annotate_td_bks_optimality`. A dust-level difference (fold-order
#: float noise between two equally optimal solutions) is acceptable when the
#: stamp carries an explanatory ``note``; anything larger means the proof and
#: the stored solution disagree and the stamp is refused.
OPTIMALITY_COST_TOLERANCE = 1e-6


def annotate_td_bks_optimality(
    loaded: LoadedTDInstance | str | Path,
    optimality: OptimalityMetadata | dict[str, Any],
    objective_function: ObjectiveFunction = ObjectiveFunction.DURATION,
) -> Path:
    """Stamp the stored BKS of ``loaded`` with an optimality proof.

    The stored BKS (of ``objective_function``'s store file) is re-validated by
    the TD checker before being rewritten with ``metadata["optimality"]`` set.
    When the stamp declares a ``proven_optimum``, it must match the stored
    cost: an absolute difference beyond :data:`OPTIMALITY_COST_TOLERANCE`
    raises, and a dust-level difference is accepted only when the stamp's
    ``note`` explains it. Everything else in the stored file (routes, cost,
    authorship, other metadata) is left untouched — the proof stamp records
    who *proved* the solution optimal, which is independent of who *found* it.
    """
    loaded = _as_loaded(loaded)
    _validate_td_objective(objective_function)
    if not isinstance(optimality, OptimalityMetadata):
        optimality = OptimalityMetadata.model_validate(optimality)

    bks_path = get_bks_path_for_instance(loaded.instance_path, objective_function)
    if not bks_path.exists():
        raise FileNotFoundError(f"No stored {objective_function.value} BKS to annotate at {bks_path}")

    existing_bks = load_bks(bks_path)
    existing_check = check_td_solution(loaded, existing_bks, objective_function)
    if not existing_check.is_valid():
        raise ValueError(f"Stored BKS is invalid at {bks_path}: {existing_check.error_message}")

    stored_cost = existing_bks.cost
    if optimality.proven_optimum is not None and stored_cost is not None:
        deviation = abs(optimality.proven_optimum - stored_cost)
        if deviation > OPTIMALITY_COST_TOLERANCE:
            raise ValueError(
                f"proven_optimum {optimality.proven_optimum!r} does not match the stored "
                f"cost {stored_cost!r} at {bks_path} (|diff| = {deviation})"
            )
        if deviation != 0.0 and not optimality.note:
            raise ValueError(
                f"proven_optimum {optimality.proven_optimum!r} differs from the stored "
                f"cost {stored_cost!r} at {bks_path} by dust ({deviation}); a "
                "self-contained `note` explaining the difference is required"
            )

    metadata = dict(existing_bks.metadata)
    metadata["optimality"] = optimality.model_dump(mode="json", exclude_none=True)
    # Reconstruct (rather than model_copy) so the metadata validator runs on the
    # stamped payload before it reaches the store.
    stamped = BenchmarkBKS(**{**existing_bks.model_dump(mode="json"), "metadata": metadata})
    save_bks(stamped, bks_path)
    return bks_path

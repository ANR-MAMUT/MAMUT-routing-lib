"""Read-only entry points for the two exact binary64 routing contracts."""

from collections.abc import Sequence
from pathlib import Path

from mamut_routing_lib.artifacts import AnyBenchmarkInstance, hydrate_collection_instance, load_benchmark_instance
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_static import PreparedStatic, prepare_static
from mamut_routing_lib.exact_td import PreparedTD, prepare_td
from mamut_routing_lib.exact_types import STATIC_PROFILE, TD_PROFILE, ExactBackend, ExactEvaluation
from mamut_routing_lib.td.artifacts import LoadedTDInstance, load_td_instance
from mamut_routing_lib.td.models import BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW


def prepare_exact(
    instance: str | Path | AnyBenchmarkInstance | LoadedTDInstance,
    *,
    profile: str | None = None,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | None = None,
) -> PreparedStatic | PreparedTD:
    """Prepare a snapshot for repeated submissions. A file path resolves referenced artifacts.

    The instance type selects the exact profile. An explicit profile asserts the expected
    type. Use prepare_td(instance, atfs) for an already materialized TD descriptor/arc pair.
    """
    path = Path(instance) if isinstance(instance, (str, Path)) else None
    if path is not None:
        instance = load_benchmark_instance(path)
    td = isinstance(instance, (LoadedTDInstance, BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW))
    expected = TD_PROFILE if td else STATIC_PROFILE
    if profile is not None and profile != expected:
        raise ValueError(f"Expected profile {expected!r} for this instance, got {profile!r}.")
    if td:
        if not isinstance(instance, LoadedTDInstance):
            if path is None:
                raise TypeError("Pass the TD instance file, a LoadedTDInstance or use prepare_td(instance, atfs).")
            instance = load_td_instance(path)
        return prepare_td(instance.instance, instance.atfs, backend=backend, objective_function=objective_function)
    return prepare_static(hydrate_collection_instance(instance, path), backend=backend, objective_function=objective_function)


def evaluate_exact(
    instance: str | Path | AnyBenchmarkInstance | LoadedTDInstance,
    routes: Sequence[Sequence[int]],
    *,
    profile: str | None = None,
    backend: ExactBackend = "native",
    objective_function: ObjectiveFunction | None = None,
) -> ExactEvaluation:
    """Check a submission and return its lossless cost and any TD witnesses.

    Infeasible submissions raise ExactRouteError. Missing native extensions raise
    ImportError. Use backend="python" to request the independent Fraction evaluator.
    """
    return prepare_exact(instance, profile=profile, backend=backend, objective_function=objective_function).evaluate(routes)

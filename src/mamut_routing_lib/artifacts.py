from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mamut_routing_lib.enums import BenchmarkName, MetricVariant, ObjectiveFunction, ProblemType
from mamut_routing_lib.json_utils import load_json_from_file, save_json_to_file
from mamut_routing_lib.models import (
    BenchmarkBKS,
    BenchmarkInstance,
    BenchmarkInstanceCVRP,
    InstanceMetadata,
)


DEFAULT_MAMUT_ROUTING_ROOT_ENV = "MAMUT_ROUTING_ROOT"
DEFAULT_BENCHMARKS_ROOT_ENV = "MAMUT_ROUTING_BENCHMARKS_ROOT"

AnyBenchmarkInstance = BenchmarkInstance | BenchmarkInstanceCVRP


def _path_from_env(env_name: str) -> Path | None:
    value = os.getenv(env_name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def get_default_mamut_routing_root() -> Path:
    root = _path_from_env(DEFAULT_MAMUT_ROUTING_ROOT_ENV)
    if root is None:
        raise RuntimeError(
            f"{DEFAULT_MAMUT_ROUTING_ROOT_ENV} is not set. "
            "Pass explicit paths to discovery/loading APIs or configure the environment."
        )
    return root


def get_default_benchmarks_root() -> Path:
    benchmark_root = _path_from_env(DEFAULT_BENCHMARKS_ROOT_ENV)
    if benchmark_root is not None:
        return benchmark_root
    return get_default_mamut_routing_root() / "benchmarks"


def get_instance_identifier(instance: AnyBenchmarkInstance) -> str:
    return instance.instance_name


def _enum_or_str(value: object) -> str:
    return str(value.value if hasattr(value, "value") else value)


def build_instance_id(
    *,
    problem_type: ProblemType | str,
    benchmark_name: BenchmarkName | str,
    num_customers: int,
    instance_name: str,
    metric_variant: MetricVariant | str | None = None,
    place_slug: str | None = None,
    subset: str | None = None,
) -> str:
    """Build a stable path-derived instance id for CLI/API selection.

    Historical layouts use problem/benchmark/size/name. Variant layouts include
    metric and place to keep IDs unique across sibling MAMUT2026 variants.
    Ortec2022-style layouts insert a ``subset`` segment between benchmark and
    size to keep IDs unique across the ``final``/``public`` partitions.
    """
    parts = [
        _enum_or_str(problem_type).lower(),
        _enum_or_str(benchmark_name).lower(),
    ]
    if subset is not None:
        parts.append(str(subset).lower())
    if metric_variant is not None:
        parts.append(_enum_or_str(metric_variant).lower())
    if place_slug is not None:
        parts.append(str(place_slug).lower())
    parts.extend([f"n{num_customers}", instance_name])
    return "-".join(parts)


@dataclass(frozen=True)
class DiscoveredBenchmarkInstance:
    problem_type: ProblemType
    benchmark_name: str
    metric_variant: MetricVariant | None
    place_slug: str | None
    num_customers: int | None
    instance_id: str
    instance_name: str
    instance_path: Path
    subset: str | None = None

    def load(self) -> AnyBenchmarkInstance:
        return load_benchmark_instance(self.instance_path)


def _parse_num_customers(part: str) -> int | None:
    if not part.startswith("n="):
        return None
    return int(part.removeprefix("n="))


@dataclass(frozen=True)
class LayoutInfo:
    """Path-layout-derived view of a benchmark instance.

    Three on-disk layouts are supported under ``benchmarks/``:

    - 4-part historical:
      ``<problem>/<benchmark>/n=<N>/<file>.vrp.json``
    - 5-part subset-partitioned (e.g. Ortec2022):
      ``<problem>/<benchmark>/<subset>/n=<N>/<file>.vrp.json``
    - 7-part Mamut2026:
      ``<problem>/<benchmark>/<metric>/<place>/n=<N>/<instance_name>/<file>.vrp.json``

    The historical layout has neither ``metric_variant`` nor ``place_slug``.
    The subset-partitioned layout has neither ``metric_variant`` nor
    ``place_slug`` either, but adds ``subset`` (e.g. ``final``/``public``).
    """

    problem_type: ProblemType
    benchmark_name: str
    metric_variant: MetricVariant | None
    place_slug: str | None
    num_customers: int
    instance_name: str
    subset: str | None = None


def parse_layout(relative_path: Path, instance_path: Path) -> LayoutInfo:
    parts = relative_path.parts
    if len(parts) == 4:
        problem_type = ProblemType(parts[0])
        benchmark_name = parts[1]
        num_customers = _parse_num_customers(parts[2])
        if num_customers is None:
            raise ValueError(f"Unsupported size bucket in benchmark instance layout: {relative_path}")
        instance_name = instance_path.stem.removesuffix(".vrp")
        return LayoutInfo(
            problem_type=problem_type,
            benchmark_name=benchmark_name,
            metric_variant=None,
            place_slug=None,
            num_customers=num_customers,
            instance_name=instance_name,
        )

    if len(parts) == 5:
        # Subset-partitioned historical-like layout, e.g. Ortec2022:
        # <problem>/<benchmark>/<subset>/n=<N>/<file>.vrp.json
        problem_type = ProblemType(parts[0])
        benchmark_name = parts[1]
        subset = parts[2]
        num_customers = _parse_num_customers(parts[3])
        if num_customers is None:
            raise ValueError(f"Unsupported size bucket in benchmark instance layout: {relative_path}")
        instance_name = instance_path.stem.removesuffix(".vrp")
        return LayoutInfo(
            problem_type=problem_type,
            benchmark_name=benchmark_name,
            metric_variant=None,
            place_slug=None,
            num_customers=num_customers,
            instance_name=instance_name,
            subset=subset,
        )

    if len(parts) == 7:
        num_customers = _parse_num_customers(parts[4])
        if num_customers is None:
            raise ValueError(f"Unsupported size bucket in benchmark instance layout: {relative_path}")
        return LayoutInfo(
            problem_type=ProblemType(parts[0]),
            benchmark_name=parts[1],
            metric_variant=MetricVariant(parts[2]),
            place_slug=parts[3],
            num_customers=num_customers,
            instance_name=parts[5],
        )

    raise ValueError(f"Unsupported benchmark instance layout: {relative_path}")


def _discover_from_relative_path(relative_path: Path, instance_path: Path) -> DiscoveredBenchmarkInstance:
    layout = parse_layout(relative_path, instance_path)
    return DiscoveredBenchmarkInstance(
        problem_type=layout.problem_type,
        benchmark_name=layout.benchmark_name,
        metric_variant=layout.metric_variant,
        place_slug=layout.place_slug,
        num_customers=layout.num_customers,
        instance_id=build_instance_id(
            problem_type=layout.problem_type,
            benchmark_name=layout.benchmark_name,
            metric_variant=layout.metric_variant,
            place_slug=layout.place_slug,
            num_customers=layout.num_customers,
            instance_name=layout.instance_name,
            subset=layout.subset,
        ),
        instance_name=layout.instance_name,
        instance_path=instance_path,
        subset=layout.subset,
    )


def discover_benchmark_instances(
    benchmarks_root: Path | None = None,
    *,
    problem_types: Iterable[ProblemType] | None = None,
    benchmark_names: Iterable[BenchmarkName | str] | None = None,
    metric_variants: Iterable[MetricVariant] | None = None,
    places: Iterable[str] | None = None,
    instance_ids: Iterable[str] | None = None,
) -> list[DiscoveredBenchmarkInstance]:
    benchmark_root = (benchmarks_root or get_default_benchmarks_root()).resolve()
    allowed_problem_types = {item.value if isinstance(item, ProblemType) else str(item) for item in (problem_types or [])}
    allowed_benchmark_names = {item.value if isinstance(item, BenchmarkName) else str(item) for item in (benchmark_names or [])}
    allowed_metric_variants = {item.value if isinstance(item, MetricVariant) else str(item) for item in (metric_variants or [])}
    allowed_places = {str(item) for item in (places or [])}
    allowed_instance_ids = {str(item) for item in (instance_ids or [])}

    discovered: list[DiscoveredBenchmarkInstance] = []
    for instance_path in sorted(benchmark_root.rglob("*.vrp.json")):
        relative_path = instance_path.relative_to(benchmark_root)
        item = _discover_from_relative_path(relative_path, instance_path)

        if allowed_problem_types and item.problem_type.value not in allowed_problem_types:
            continue
        if allowed_benchmark_names and item.benchmark_name not in allowed_benchmark_names:
            continue
        if allowed_metric_variants:
            if item.metric_variant is None or item.metric_variant.value not in allowed_metric_variants:
                continue
        if allowed_places:
            if item.place_slug is None or item.place_slug not in allowed_places:
                continue
        if allowed_instance_ids and item.instance_id not in allowed_instance_ids:
            continue

        discovered.append(item)

    return discovered


def load_benchmark_instance(instance_path: str | Path) -> AnyBenchmarkInstance:
    payload = load_json_from_file(instance_path)
    if (
        payload.get("benchmark_name") == BenchmarkName.MAMUT_2026.value
        and "metadata" in payload
        and "service_times" not in payload
    ):
        return BenchmarkInstanceCVRP(**payload)
    return BenchmarkInstance(**payload)


def has_structured_metadata(instance: AnyBenchmarkInstance) -> bool:
    """Return True if the instance carries a validated InstanceMetadata payload.

    BenchmarkInstanceCVRP always does. Unified BenchmarkInstance does for Mamut2026
    (pydantic Union resolves to InstanceMetadata when the structured fields match);
    historical Sintef/Dimacs instances carry a plain dict.
    """
    return isinstance(getattr(instance, "metadata", None), InstanceMetadata)


def get_bks_path_for_instance(
    instance_path: str | Path,
    objective_function: ObjectiveFunction,
) -> Path:
    path = Path(instance_path)
    base_name = path.name.removesuffix(".vrp.json")
    return path.with_name(f"{base_name}.bks.{objective_function.value}.json")


def load_bks(bks_path: str | Path) -> BenchmarkBKS:
    return BenchmarkBKS(**load_json_from_file(bks_path))


def save_bks(bks: BenchmarkBKS, bks_path: str | Path) -> None:
    save_json_to_file(bks.model_dump(mode="json"), bks_path)

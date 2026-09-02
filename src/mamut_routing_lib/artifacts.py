from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from mamut_routing_lib.td.models import AnyTDBenchmarkInstance

from mamut_routing_lib.enums import BenchmarkName, MetricVariant, ObjectiveFunction, ProblemType
from mamut_routing_lib.json_utils import load_json_from_file, save_json_to_file
from mamut_routing_lib.models import (
    AnyCollectionStaticInstance,
    ArcCostsDistancesRef,
    ArcCostsEuclidean,
    BenchmarkBKS,
    BenchmarkInstance,
    BenchmarkInstanceCVRP,
    BenchmarkInstanceCVRPCollection,
    BenchmarkInstanceVRPTWCollection,
    InstanceMetadata,
)
from mamut_routing_lib.sidecars import (
    COLLECTION_MARKER_FILENAME,
    load_collection_marker,
    require_collection_root,
)


DEFAULT_MAMUT_ROUTING_ROOT_ENV = "MAMUT_ROUTING_ROOT"
DEFAULT_BENCHMARKS_ROOT_ENV = "MAMUT_ROUTING_BENCHMARKS_ROOT"

AnyBenchmarkInstance = BenchmarkInstance | BenchmarkInstanceCVRP | AnyCollectionStaticInstance


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
    metric and place to keep IDs unique across sibling Poryos2026 variants.
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
    base_instance_name: str | None = None
    subinstance: str | None = None
    tw_set: str | None = None

    def load(self) -> "AnyBenchmarkInstance | AnyTDBenchmarkInstance":
        return load_benchmark_instance(self.instance_path)


def _parse_num_customers(part: str) -> int | None:
    if not part.startswith("n="):
        return None
    return int(part.removeprefix("n="))


@dataclass(frozen=True)
class LayoutInfo:
    """Path-layout-derived view of a benchmark instance.

    Four on-disk layouts are supported under ``benchmarks/``:

    - 4-part historical:
      ``<problem>/<benchmark>/n=<N>/<file>.vrp.json``
    - 5-part subset-partitioned (e.g. Ortec2022):
      ``<problem>/<benchmark>/<subset>/n=<N>/<file>.vrp.json``
    - 7-part Poryos2026 v1 (retired with the v1 family, still parsed):
      ``<problem>/<benchmark>/<metric>/<place>/n=<N>/<instance_name>/<file>.vrp.json``
    - family-first **collection** (marker-rooted, e.g. ``Poryos2026/``), paths
      relative to the collection root, parsed by ``parse_collection_layout``:
      ``<problem>/<metric>/<city>/n=<N>/<base>/<file>.vrp.json`` (CVRP/VRPTW)
      ``<problem>/<city>/n=<N>/<base>/<subinstance>/<file>.vrp.json`` (TD)

    The historical layout has neither ``metric_variant`` nor ``place_slug``.
    The subset-partitioned layout has neither ``metric_variant`` nor
    ``place_slug`` either, but adds ``subset`` (e.g. ``final``/``public``).
    Collection layouts expose ``base_instance_name`` and (TD) ``subinstance``.
    Collection VRPTW instances additionally expose ``tw_set``: ``td-shared``
    for the bare base name (windows shared with the TDVRPTW twins), or the
    set tag of a static-only ``<base>-tw-<set>`` instance.
    """

    problem_type: ProblemType
    benchmark_name: str
    metric_variant: MetricVariant | None
    place_slug: str | None
    num_customers: int
    instance_name: str
    subset: str | None = None
    base_instance_name: str | None = None
    subinstance: str | None = None
    tw_set: str | None = None


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


def parse_collection_layout(
    relative_path: Path,
    instance_path: Path,
    family: str,
) -> LayoutInfo:
    """Parse a family-first collection path (relative to the collection root).

    Two shapes, both 6 parts:

    - static: ``<problem>/<metric>/<city>/n=<N>/<base>/<file>.vrp.json``
    - TD: ``<problem>/<city>/n=<N>/<base>/<subinstance>/<file>.vrp.json``

    The problem type discriminates: TDVRP/TDVRPTW have no metric slot (the
    time dependence is the metric) and add the subinstance directory instead.
    Static instance files are named after their base directory; VRPTW extra
    TW-set instances (``<base>-tw-<set>.vrp.json``) sit next to the TD-paired
    ``<base>.vrp.json`` and expose their tag via ``LayoutInfo.tw_set``.
    """
    parts = relative_path.parts
    if len(parts) != 6:
        raise ValueError(f"Unsupported collection instance layout: {relative_path}")
    problem_type = ProblemType(parts[0])
    instance_name = instance_path.stem.removesuffix(".vrp")
    if problem_type in (ProblemType.TDVRP, ProblemType.TDVRPTW):
        num_customers = _parse_num_customers(parts[2])
        if num_customers is None:
            raise ValueError(f"Unsupported size bucket in collection instance layout: {relative_path}")
        base, subinstance = parts[3], parts[4]
        if instance_name != f"{base}-{subinstance}":
            raise ValueError(
                f"collection TD instance name {instance_name!r} does not equal "
                f"<base>-<subinstance> ({base!r}, {subinstance!r}): {relative_path}"
            )
        return LayoutInfo(
            problem_type=problem_type,
            benchmark_name=family,
            metric_variant=None,
            place_slug=parts[1],
            num_customers=num_customers,
            instance_name=instance_name,
            base_instance_name=base,
            subinstance=subinstance,
        )
    num_customers = _parse_num_customers(parts[3])
    if num_customers is None:
        raise ValueError(f"Unsupported size bucket in collection instance layout: {relative_path}")
    base = parts[4]
    tw_set: str | None = None
    if instance_name != base:
        tw_prefix = f"{base}-tw-"
        if problem_type is ProblemType.VRPTW and instance_name.startswith(tw_prefix) and len(
            instance_name
        ) > len(tw_prefix):
            tw_set = instance_name[len(tw_prefix) :]
        else:
            raise ValueError(
                f"collection static instance name {instance_name!r} does not equal "
                f"its base directory {base!r} (VRPTW extra TW sets use "
                f"<base>-tw-<set>): {relative_path}"
            )
    elif problem_type is ProblemType.VRPTW:
        tw_set = "td-shared"
    return LayoutInfo(
        problem_type=problem_type,
        benchmark_name=family,
        metric_variant=MetricVariant(parts[1]),
        place_slug=parts[2],
        num_customers=num_customers,
        instance_name=instance_name,
        base_instance_name=base,
        tw_set=tw_set,
    )


def _discovered_from_layout(layout: LayoutInfo, instance_path: Path) -> DiscoveredBenchmarkInstance:
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
        base_instance_name=layout.base_instance_name,
        subinstance=layout.subinstance,
        tw_set=layout.tw_set,
    )


def _discover_from_relative_path(relative_path: Path, instance_path: Path) -> DiscoveredBenchmarkInstance:
    return _discovered_from_layout(parse_layout(relative_path, instance_path), instance_path)


def find_collection_roots(benchmarks_root: Path) -> dict[Path, str]:
    """Marker-rooted collections at or directly under ``benchmarks_root``: root -> family.

    The root-marker case covers scanning a standalone collection checkout
    (the collection repo itself) as the benchmarks tree.
    """
    roots: dict[Path, str] = {}
    own_marker = benchmarks_root / COLLECTION_MARKER_FILENAME
    if own_marker.is_file():
        roots[benchmarks_root] = load_collection_marker(own_marker).family
    for marker_path in sorted(benchmarks_root.glob(f"*/{COLLECTION_MARKER_FILENAME}")):
        marker = load_collection_marker(marker_path)
        roots[marker_path.parent] = marker.family
    return roots


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

    collection_roots = find_collection_roots(benchmark_root)

    def _collection_of(instance_path: Path) -> tuple[Path, str] | None:
        for root, family in collection_roots.items():
            if instance_path.is_relative_to(root):
                return root, family
        return None

    discovered: list[DiscoveredBenchmarkInstance] = []
    for instance_path in sorted(benchmark_root.rglob("*.vrp.json")):
        collection = _collection_of(instance_path)
        if collection is not None:
            root, family = collection
            layout = parse_collection_layout(instance_path.relative_to(root), instance_path, family)
            item = _discovered_from_layout(layout, instance_path)
        else:
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


def load_benchmark_instance(instance_path: str | Path) -> "AnyBenchmarkInstance | AnyTDBenchmarkInstance":
    payload = load_json_from_file(instance_path)
    if "td" in payload:
        # Time-dependent instance: travel is described by the ATF sidecar
        # referenced by the "td" block (see mamut_routing_lib.td).
        from mamut_routing_lib.td.artifacts import td_instance_from_payload

        return td_instance_from_payload(payload)
    if "arc_costs_source" in payload:
        # Slim collection instance: matrix by sha-pinned sidecar reference
        # (hydrate with resolve_arc_costs).
        if "time_windows" in payload:
            return BenchmarkInstanceVRPTWCollection(**payload)
        return BenchmarkInstanceCVRPCollection(**payload)
    if (
        payload.get("benchmark_name") == BenchmarkName.PORYOS_2026.value
        and "metadata" in payload
        and "service_times" not in payload
    ):
        return BenchmarkInstanceCVRP(**payload)
    return BenchmarkInstance(**payload)


def instance_problem_type(instance: "AnyBenchmarkInstance | AnyTDBenchmarkInstance") -> ProblemType:
    """The problem type of a loaded instance, from its model class.

    Time-dependent models resolve to TDVRPTW/TDVRP (their metadata does not
    always carry ``problem_type``); the static ones to CVRP or VRPTW.
    """
    from mamut_routing_lib.td.models import BenchmarkInstanceTDVRP, BenchmarkInstanceTDVRPTW

    if isinstance(instance, BenchmarkInstanceTDVRPTW):
        return ProblemType.TDVRPTW
    if isinstance(instance, BenchmarkInstanceTDVRP):
        return ProblemType.TDVRP
    if isinstance(instance, (BenchmarkInstanceCVRP, BenchmarkInstanceCVRPCollection)):
        return ProblemType.CVRP
    if isinstance(instance, (BenchmarkInstance, BenchmarkInstanceVRPTWCollection)):
        return ProblemType.VRPTW
    raise TypeError(f"unsupported instance model: {type(instance).__name__}")


def resolve_arc_costs(
    instance: AnyCollectionStaticInstance,
    instance_path: str | Path,
    collection_root: str | Path | None = None,
) -> list[list[float]]:
    """Hydrate the arc-cost matrix of a slim collection instance.

    ``distances-sidecar`` sources resolve against the collection root (marker
    walk-up from the instance file, or explicit ``collection_root``) and are
    sha-verified when the reference is pinned; ``euclidean`` sources are
    materialized from the stored coordinates by the canonical rounding
    formula.
    """
    source = instance.arc_costs_source
    if isinstance(source, ArcCostsEuclidean):
        import math

        coords = [(float(x), float(y)) for x, y in instance.coordinates]
        decimals = source.decimals
        return [
            [
                0.0 if i == j else round(math.hypot(bx - ax, by - ay), decimals)
                for j, (bx, by) in enumerate(coords)
            ]
            for i, (ax, ay) in enumerate(coords)
        ]
    if isinstance(source, ArcCostsDistancesRef):
        from mamut_routing_lib.distances import compute_distances_sha256, load_instance_distances

        root = require_collection_root(instance_path, collection_root)
        distances = load_instance_distances(root / source.distances.path)
        if distances.num_customers != instance.num_customers:
            raise ValueError(
                f"distances num_customers {distances.num_customers} does not match "
                f"instance {instance.num_customers}"
            )
        if distances.metric != instance.metric_variant.value:
            raise ValueError(
                f"distances metric {distances.metric!r} does not match instance "
                f"metric_variant {instance.metric_variant.value!r}"
            )
        if source.distances.sha256 is not None:
            digest = compute_distances_sha256(distances)
            if digest != source.distances.sha256:
                raise ValueError(
                    f"distances sha256 mismatch: computed {digest}, "
                    f"instance declares {source.distances.sha256}"
                )
        return distances.values
    raise ValueError(f"unsupported arc_costs_source model: {source!r}")


def has_structured_metadata(instance: AnyBenchmarkInstance) -> bool:
    """Return True if the instance carries a validated InstanceMetadata payload.

    BenchmarkInstanceCVRP always does. Unified BenchmarkInstance does for Poryos2026
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

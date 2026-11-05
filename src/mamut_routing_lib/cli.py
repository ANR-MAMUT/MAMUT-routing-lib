from __future__ import annotations

import json
import os
import sys
import threading
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Optional, TypeVar

import typer
from tqdm import tqdm

from mamut_routing_lib.artifacts import (
    RELEASE_STAGING_DIRNAME,
    AnyBenchmarkInstance,
    BenchmarkLayoutResolver,
    DEFAULT_BENCHMARKS_ROOT_ENV,
    DEFAULT_MAMUT_ROUTING_ROOT_ENV,
    build_instance_id,
    instance_problem_type,
    is_collection_instance,
    load_benchmark_instance,
)
from mamut_routing_lib.cvrplib import (
    EDGE_WEIGHT_TYPES,
    EXPORT_FORMATS,
    ExportResult,
    VrpExportOptions,
    export_filename,
    export_instance_file,
)
from mamut_routing_lib.enums import BenchmarkName, MetricVariant, ObjectiveFunction, ProblemType
from mamut_routing_lib.models import ArcCostsDistancesRef
from mamut_routing_lib.sidecars import require_collection_root
from mamut_routing_lib.remote import (
    DEFAULT_GITHUB_TOKEN_ENV,
    DEFAULT_RELEASE_REPO_ENV,
    GitHubReleaseClient,
    GitHubReleaseSource,
    ReleaseArchiveAsset,
    ReleaseArchiveManifest,
    ReleaseArchiveScope,
    ReleaseExtractionError,
    compute_sha256,
    read_release_stamp,
    release_target_relpath,
)


app = typer.Typer(
    name="mamut-routing",
    help="CLI for the MAMUT-routing benchmark library: list and solve local benchmark instances.",
    no_args_is_help=True,
    add_completion=False,
)

remote_app = typer.Typer(
    help="Commands for remote benchmark archives published as GitHub release assets.",
    no_args_is_help=True,
)
app.add_typer(remote_app, name="remote")

export_app = typer.Typer(
    help="Export local benchmark instances to classic solver formats (CVRPLIB .vrp, Solomon .txt).",
    no_args_is_help=True,
)
app.add_typer(export_app, name="export")

bks_app = typer.Typer(
    help="Maintenance of stored best-known solutions.",
    no_args_is_help=True,
)
app.add_typer(bks_app, name="bks")

solvers_app = typer.Typer(
    help="Registered solvers: PyVRP, KAYROS and external executables (solvers config).",
    no_args_is_help=True,
)
app.add_typer(solvers_app, name="solvers")

SOLVERS_CONFIG_HELP = (
    "Solvers config TOML (default: $MAMUT_ROUTING_SOLVERS_CONFIG, then "
    "$XDG_CONFIG_HOME/mamut-routing/solvers.toml). Never read from the working directory."
)


def _get_package_version() -> str:
    package_name = "mamut-routing-lib"
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        pass
    try:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        return str(pyproject_data["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mamut-routing-lib {_get_package_version()}")
        raise typer.Exit()


@dataclass
class CLIState:
    benchmarks_dir: Path


@dataclass
class RemoteCLIState:
    repo: str
    token: str | None
    tag: str | None
    benchmarks_dir: Path

    def make_client(self) -> GitHubReleaseClient:
        source = GitHubReleaseSource(repo_full_name=self.repo, token=self.token)
        return GitHubReleaseClient(source=source)


@dataclass(frozen=True)
class LocalInstanceRecord:
    path: Path
    instance: AnyBenchmarkInstance
    instance_id: str
    instance_name: str
    problem_type: ProblemType
    benchmark_name: str
    metric_variant: MetricVariant | str | None
    place_slug: str | None
    num_customers: int
    subset: str | None = None


@dataclass(frozen=True)
class SolveSummaryRow:
    instance_id: str
    instance_name: str
    status: str
    cost: int | float | None
    route_count: int
    wall_time: float
    bks_label: str
    detail: str = ""


T = TypeVar("T")


def _resolve_default_benchmarks_dir() -> Path:
    benchmarks_root = os.getenv(DEFAULT_BENCHMARKS_ROOT_ENV)
    if benchmarks_root:
        return Path(benchmarks_root).expanduser().resolve()
    routing_root = os.getenv(DEFAULT_MAMUT_ROUTING_ROOT_ENV)
    if routing_root:
        return (Path(routing_root).expanduser() / "benchmarks").resolve()
    return Path.cwd() / "benchmarks"


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the mamut-routing-lib version and exit.",
        ),
    ] = None,
    benchmarks_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--benchmarks-dir",
            help=f"Local benchmark directory. Defaults to ${DEFAULT_BENCHMARKS_ROOT_ENV} "
            f"or ${DEFAULT_MAMUT_ROUTING_ROOT_ENV}/benchmarks or ./benchmarks.",
        ),
    ] = None,
) -> None:
    resolved_benchmarks_dir = (
        benchmarks_dir.expanduser().resolve() if benchmarks_dir is not None else _resolve_default_benchmarks_dir()
    )
    ctx.obj = CLIState(benchmarks_dir=resolved_benchmarks_dir)


@remote_app.callback()
def remote_callback(
    ctx: typer.Context,
    repo: Annotated[
        Optional[str],
        typer.Option(
            "--repo",
            help=f"GitHub repo (owner/name). Defaults to ${DEFAULT_RELEASE_REPO_ENV} or 'ANR-MAMUT/MAMUT-routing'.",
        ),
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            help=f"GitHub token. Defaults to ${DEFAULT_GITHUB_TOKEN_ENV}. Unnecessary for public releases with sufficient rate limits.",
        ),
    ] = None,
    tag: Annotated[
        Optional[str],
        typer.Option(
            "--tag",
            help="Release tag (e.g. snapshot-2026-05-22-28f9199). Defaults to the latest release.",
        ),
    ] = None,
) -> None:
    root_state: CLIState = ctx.find_root().obj
    resolved_repo = repo or os.getenv(DEFAULT_RELEASE_REPO_ENV) or "ANR-MAMUT/MAMUT-routing"
    parts = resolved_repo.split("/")
    if len(parts) != 2 or not all(parts):
        typer.echo(
            f"Error: --repo must be in 'owner/name' format, got: {resolved_repo!r}. "
            f"Example: --repo ANR-MAMUT/MAMUT-routing",
            err=True,
        )
        raise typer.Exit(code=2)
    resolved_token = token if token is not None else os.getenv(DEFAULT_GITHUB_TOKEN_ENV)
    ctx.obj = RemoteCLIState(
        repo=resolved_repo,
        token=resolved_token,
        tag=tag,
        benchmarks_dir=root_state.benchmarks_dir,
    )


def _select_assets(
    manifest: ReleaseArchiveManifest,
    *,
    filenames: list[str],
    select_all: bool,
    problem_type: ProblemType | None,
    benchmark_name: BenchmarkName | None,
) -> list[ReleaseArchiveAsset]:
    if select_all:
        return list(manifest.assets)
    if filenames:
        by_name = {asset.filename: asset for asset in manifest.assets}
        missing = [name for name in filenames if name not in by_name]
        if missing:
            raise typer.BadParameter(
                f"Unknown asset filename(s) in manifest: {', '.join(missing)}. "
                f"Run `mamut-routing remote list` to see available assets."
            )
        return [by_name[name] for name in filenames]
    if problem_type is None and benchmark_name is None:
        return []
    return manifest.select_assets(
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )


def _format_size_mb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "?"
    return f"{size_bytes / (1024 * 1024):.2f}"


def _short_sha(sha: str | None) -> str:
    if not sha:
        return "-"
    return sha[:8]


@remote_app.command("list")
def remote_list_assets(
    ctx: typer.Context,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
) -> None:
    """List benchmark archives available in the remote release manifest."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    assets = manifest.select_assets(
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )

    typer.echo(
        f"Snapshot {manifest.snapshot_id} (tag={manifest.release_tag or '-'}, "
        f"published_at={manifest.published_at}, source_commit={manifest.source_commit})"
    )
    typer.echo(f"Repository: {state.repo}")
    typer.echo(f"Assets: {len(assets)}")
    typer.echo("")

    header = f"{'FILENAME':<60}  {'SCOPE':<14}  {'PROBLEM':<6}  {'FAMILY':<14}  {'SIZE_MB':>8}  {'SHA256':<8}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for asset in assets:
        typer.echo(
            f"{asset.filename:<60}  "
            f"{asset.scope.value:<14}  "
            f"{(asset.problem_type.value if asset.problem_type else ('all' if asset.scope == ReleaseArchiveScope.FAMILY_COLLECTION else '-')):<6}  "
            f"{(asset.benchmark_name.value if asset.benchmark_name else '-'):<14}  "
            f"{_format_size_mb(asset.size_bytes):>8}  "
            f"{_short_sha(asset.checksum_sha256):<8}"
        )


@remote_app.command("fetch")
def remote_fetch_assets(
    ctx: typer.Context,
    filenames: Annotated[
        Optional[list[str]],
        typer.Argument(help="One or more asset filenames to download. Omit to use filter flags or --all."),
    ] = None,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
    select_all: Annotated[bool, typer.Option("--all", help="Download every asset in the manifest.")] = False,
    extract: Annotated[
        bool,
        typer.Option(
            "--extract/--no-extract",
            help="Extract each archive into the canonical tree under --benchmarks-dir (default).",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace an existing family directory even if it was not extracted by `remote fetch` "
            "(e.g. a git checkout). A previously fetched directory is always replaced.",
        ),
    ] = False,
) -> None:
    """Download (and optionally extract) one or more benchmark archives.

    Archives are kept at <benchmarks-dir>/<filename>. With --extract (default)
    each one lands in the canonical tree, e.g. <benchmarks-dir>/VRPTW/Sintef2008
    or <benchmarks-dir>/Poryos2026, so `list` and `discover_benchmark_instances`
    work on the fetched tree. Re-fetching replaces the family directory,
    including files written into it since (e.g. BKS saved by `solve`).
    """
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    selected = _select_assets(
        manifest,
        filenames=filenames or [],
        select_all=select_all,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )
    if not selected:
        typer.echo("No assets selected. Use positional filenames, --problem-type/--benchmark-name, or --all.", err=True)
        raise typer.Exit(code=2)

    state.benchmarks_dir.mkdir(parents=True, exist_ok=True)
    for asset in selected:
        legacy = state.benchmarks_dir / Path(asset.filename).stem / "benchmarks"
        if legacy.is_dir():
            typer.secho(
                f"Warning: {legacy.parent} is a legacy extraction (lib < 0.12, <stem>/benchmarks/...) that breaks "
                "discover_benchmark_instances on this dir; it is left in place (it may hold solve output), "
                "remove it once the canonical tree is fetched.",
                fg=typer.colors.YELLOW,
                err=True,
            )
    typer.echo(f"Downloading {len(selected)} asset(s) into {state.benchmarks_dir}")

    for asset in selected:
        with tqdm(
            total=asset.size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=asset.filename,
            leave=True,
            file=sys.stderr,
        ) as bar:
            def _on_progress(downloaded: int, total: int | None, _bar: tqdm = bar) -> None:
                if total is not None and _bar.total != total:
                    _bar.total = total
                _bar.n = downloaded
                _bar.refresh()

            try:
                destination = client.download_asset(
                    asset,
                    state.benchmarks_dir,
                    extract=extract,
                    progress_callback=_on_progress,
                    manifest=manifest,
                    force=force,
                )
            except ReleaseExtractionError as exc:
                typer.echo(f"Error: {asset.filename}: {exc}", err=True)
                raise typer.Exit(code=2) from exc
        typer.echo(f"  -> {destination}")
        if extract:
            typer.echo(f"     archive kept at {state.benchmarks_dir / asset.filename}")


@remote_app.command("verify")
def remote_verify_local(
    ctx: typer.Context,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
) -> None:
    """Verify sha256 of locally downloaded archives against the remote manifest."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    assets = manifest.select_assets(
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )

    typer.echo(f"Verifying {len(assets)} asset(s) under {state.benchmarks_dir}")
    has_failure = False
    for asset in assets:
        local_path = state.benchmarks_dir / asset.filename
        tree_stamp = None
        if asset.archive_root:
            tree_stamp = read_release_stamp(state.benchmarks_dir / release_target_relpath(asset.archive_root))
        if not local_path.exists():
            if tree_stamp is not None and tree_stamp.get("checksum_sha256") == asset.checksum_sha256:
                typer.echo(f"  EXTRACTED {asset.filename} (archive removed; extracted tree stamped with this checksum)")
            elif tree_stamp is not None:
                typer.echo(f"  STALE     {asset.filename} (extracted tree comes from {tree_stamp.get('snapshot_id') or 'another archive'})")
                has_failure = True
            else:
                typer.echo(f"  MISSING   {asset.filename}")
                has_failure = True
            continue
        if asset.checksum_sha256 is None:
            typer.echo(f"  NO_SHA    {asset.filename} (no checksum recorded in manifest)")
            continue
        actual = compute_sha256(local_path)
        if actual == asset.checksum_sha256:
            typer.echo(f"  OK        {asset.filename}")
        else:
            typer.echo(
                f"  MISMATCH  {asset.filename} (expected {asset.checksum_sha256[:12]}..., got {actual[:12]}...)"
            )
            has_failure = True

    if has_failure:
        raise typer.Exit(code=1)


def _iter_candidate_paths(state: "CLIState", instance_paths: list[Path] | None) -> Iterator[Path]:
    """Yield instance paths in deterministic (sorted) order.

    With explicit `instance_paths`, paths are validated and yielded in the given order.
    Without, the configured `--benchmarks-dir` is walked recursively and the matching
    `*.vrp.json` files are sorted before iteration so that callers (`list`, `solve`)
    process the same tree in the same order.
    """
    if instance_paths:
        candidate_paths = [Path(p).expanduser().resolve() for p in instance_paths]
        missing = [p for p in candidate_paths if not p.is_file()]
        if missing:
            typer.echo(
                f"Error: instance file(s) not found: {', '.join(str(p) for p in missing)}",
                err=True,
            )
            raise typer.Exit(code=2)
        yield from candidate_paths
        return

    if not state.benchmarks_dir.is_dir():
        typer.echo(
            f"Error: --benchmarks-dir does not exist: {state.benchmarks_dir}. Pass positional "
            f"INSTANCE_PATHS, specify a valid --benchmarks-dir or fetch archives first.",
            err=True,
        )
        raise typer.Exit(code=2)
    yield from sorted(
        path
        for path in state.benchmarks_dir.rglob("*.vrp.json")
        if RELEASE_STAGING_DIRNAME not in path.relative_to(state.benchmarks_dir).parts
    )


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _metadata_value(metadata: Any, key: str) -> Any:
    if not metadata:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _coerce_metric_variant(value: Any) -> MetricVariant | str | None:
    if value is None:
        return None
    if isinstance(value, MetricVariant):
        return value
    try:
        return MetricVariant(str(value))
    except ValueError:
        return str(value)


def _coerce_benchmark_name(value: Any) -> str:
    return str(value.value if hasattr(value, "value") else value)


def _problem_type_from_instance(instance: "AnyBenchmarkInstance") -> ProblemType:
    return instance_problem_type(instance)


def _resolve_layout(path: Path, resolver: BenchmarkLayoutResolver):
    """The layout discovery would give `path` (collections included), or None."""
    try:
        return resolver.layout_for(path)
    except ValueError:
        return None


def _local_instance_record(
    path: Path,
    instance: "AnyBenchmarkInstance",
    benchmarks_dir: Path,
    resolver: BenchmarkLayoutResolver | None = None,
) -> LocalInstanceRecord:
    metadata = getattr(instance, "metadata", None)
    instance_name = str(getattr(instance, "instance_name"))
    problem_type = _problem_type_from_instance(instance)
    benchmark_name = _coerce_benchmark_name(getattr(instance, "benchmark_name"))
    metric_variant = _coerce_metric_variant(_metadata_value(metadata, "metric_variant"))
    place_slug = _metadata_value(metadata, "place_slug")
    subset = _metadata_value(metadata, "subset")
    num_customers = int(getattr(instance, "num_customers"))

    layout = _resolve_layout(path, resolver or BenchmarkLayoutResolver(benchmarks_dir))
    if layout is not None:
        # The layout is the one discovery uses (family-first collections
        # included), so the ID below equals the discovered/website ID.
        # Path overrides metadata for fields the path actually carries. Historical
        # 4-part layouts have neither metric_variant nor place_slug — fall back to
        # whatever metadata supplies (e.g. enriched Dimacs/Sintef instances now
        # carry `metric_variant: "euclidean"` in their metadata). 5-part subset
        # layouts (e.g. Ortec2022) carry ``subset`` and override the metadata's
        # value for ID purposes.
        problem_type = layout.problem_type
        benchmark_name = layout.benchmark_name
        if layout.metric_variant is not None:
            metric_variant = layout.metric_variant
        if layout.place_slug is not None:
            place_slug = layout.place_slug
        if layout.subset is not None:
            subset = layout.subset
        metric_for_id = layout.metric_variant
        place_for_id = layout.place_slug
        subset_for_id = layout.subset
        # When the path resolves, the bucket label is the canonical n for IDs —
        # matches ``discover_benchmark_instances`` and keeps catalogue-derived
        # IDs consistent. For Ortec2022 the bucket (e.g. 200) intentionally
        # differs from the per-instance ``num_customers`` (e.g. 212).
        num_for_id = layout.num_customers
    else:
        metric_for_id = metric_variant if place_slug is not None else None
        place_for_id = place_slug
        subset_for_id = subset
        num_for_id = num_customers

    instance_id = build_instance_id(
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_for_id,
        place_slug=place_for_id,
        subset=subset_for_id,
        num_customers=num_for_id,
        instance_name=instance_name,
    )

    return LocalInstanceRecord(
        path=path,
        instance=instance,
        instance_id=instance_id,
        instance_name=instance_name,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_variant,
        place_slug=place_slug,
        num_customers=num_customers,
        subset=subset,
    )


def _instance_matches(
    record: LocalInstanceRecord,
    *,
    problem_type: ProblemType | None,
    benchmark_name: BenchmarkName | None,
    metric_variant: MetricVariant | None,
    instance_id: str | None,
    instance_name: str | None,
) -> bool:
    if problem_type is not None and record.problem_type != problem_type:
        return False

    if benchmark_name is not None:
        if record.benchmark_name != benchmark_name.value:
            return False

    if metric_variant is not None:
        inst_metric = record.metric_variant
        if isinstance(inst_metric, str):
            inst_metric = _coerce_metric_variant(inst_metric)
        if inst_metric != metric_variant:
            return False

    if instance_id is not None and record.instance_id != instance_id:
        return False

    if instance_name is not None and record.instance_name != instance_name:
        return False

    return True


def _filter_loaded_instances(
    paths: list[Path],
    *,
    problem_type: ProblemType | None,
    benchmark_name: BenchmarkName | None,
    metric_variant: MetricVariant | None,
    instance_id: str | None,
    instance_name: str | None,
    benchmarks_dir: Path,
) -> list[LocalInstanceRecord]:
    selected: list[LocalInstanceRecord] = []
    resolver = BenchmarkLayoutResolver(benchmarks_dir)
    for path in paths:
        instance = load_benchmark_instance(path)
        record = _local_instance_record(path, instance, benchmarks_dir, resolver)
        if _instance_matches(
            record,
            problem_type=problem_type,
            benchmark_name=benchmark_name,
            metric_variant=metric_variant,
            instance_id=instance_id,
            instance_name=instance_name,
        ):
            selected.append(record)
    return selected


def _missing_arc_cost_sidecar(record: LocalInstanceRecord) -> Path | None:
    """The distances sidecar a collection instance needs but the tree lacks (sha256 pins only), else None."""
    instance = record.instance
    if not is_collection_instance(instance) or not isinstance(instance.arc_costs_source, ArcCostsDistancesRef):
        return None
    try:
        root = require_collection_root(record.path)
    except ValueError:
        return None  # let the solve itself report the unresolvable root
    target = root / instance.arc_costs_source.distances.path
    return None if target.is_file() else target


def _base_objective_for_record(record: LocalInstanceRecord) -> ObjectiveFunction | None:
    if record.problem_type != ProblemType.VRPTW:
        return None
    if record.benchmark_name == BenchmarkName.SINTEF_2008.value:
        return ObjectiveFunction.HIERARCHICAL_VEHICLE_COST
    if record.benchmark_name == BenchmarkName.DIMACS_2021.value:
        return ObjectiveFunction.MONO_COST
    return None


def _warn_on_non_base_objectives(records: list[LocalInstanceRecord], objective: ObjectiveFunction) -> None:
    for record in records:
        base_objective = _base_objective_for_record(record)
        if base_objective is None or base_objective == objective:
            continue
        typer.secho(
            "Warning: "
            f"{record.instance_id} belongs to {record.benchmark_name}, whose base objective is "
            f"{base_objective.value}; solving with {objective.value}.",
            fg=typer.colors.YELLOW,
            bold=True,
            err=True,
        )


def _run_with_time_progress(
    *,
    record: LocalInstanceRecord,
    time_limit_s: int,
    enabled: bool,
    operation: Callable[[], T],
    on_interrupt: Callable[[], None] | None = None,
) -> T:
    if not enabled:
        return operation()

    desc = record.instance_id if len(record.instance_id) <= 48 else f"{record.instance_id[:45]}..."
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        started_at = time.monotonic()
        try:
            _show_time_progress(future, desc=desc, time_limit_s=time_limit_s, started_at=started_at)
        except KeyboardInterrupt:
            # Stop the solve before the executor's exit waits for it, or Ctrl-C hangs until the budget runs out.
            if on_interrupt is not None:
                on_interrupt()
            raise
        return future.result()


def _show_time_progress(future: Any, *, desc: str, time_limit_s: int, started_at: float) -> None:
    with tqdm(
        total=float(time_limit_s),
        desc=f"{desc} ({time_limit_s}s left)",
        unit="s",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        bar_format="{desc}: {percentage:3.0f}%|{bar}|",
        file=sys.stderr,
    ) as bar:
        while not future.done():
            elapsed = time.monotonic() - started_at
            target = min(elapsed, float(time_limit_s))
            remaining = max(0.0, float(time_limit_s) - elapsed)
            bar.n = target
            bar.set_description_str(f"{desc} ({remaining:.0f}s left)", refresh=False)
            bar.refresh()
            time.sleep(0.2)

        elapsed = min(time.monotonic() - started_at, float(time_limit_s))
        if elapsed > bar.n:
            bar.n = elapsed
            bar.refresh()


def _default_solve_jobs() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count - 2)


def _artifact_workdirs(records: list[LocalInstanceRecord], root: Path | None, leaf: str) -> dict[int, Path]:
    """One ``--keep-artifacts`` folder per selected instance, never shared by two of them.

    Instance ids repeat when the layout cannot be read from the path (explicit files outside a
    benchmarks tree, metric variants of one instance): those get a short hash of their path.
    """
    if root is None:
        return {}
    import hashlib

    base = root.resolve()
    counts: dict[str, int] = {}
    for record in records:
        counts[record.instance_id] = counts.get(record.instance_id, 0) + 1
    workdirs: dict[int, Path] = {}
    for record in records:
        name = record.instance_id
        if counts[name] > 1:
            name = f"{name}-{hashlib.sha256(str(record.path.resolve()).encode()).hexdigest()[:8]}"
        workdirs[id(record)] = base / name / leaf
    return workdirs


def _duplicate_resolved_paths(records: list[LocalInstanceRecord]) -> list[Path]:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for record in records:
        resolved = record.path.resolve()
        if resolved in seen:
            duplicates.append(resolved)
        else:
            seen.add(resolved)
    return duplicates


@app.command("list")
def list_instances(
    ctx: typer.Context,
    instance_paths: Annotated[
        Optional[list[Path]],
        typer.Argument(
            help="One or more benchmark instance JSON paths. If omitted, --benchmarks-dir is "
            "scanned recursively for *.vrp.json files (and filter flags apply).",
        ),
    ] = None,
    problem_type: Annotated[
        Optional[ProblemType],
        typer.Option("--problem-type", case_sensitive=False, help="Filter by CVRP or VRPTW."),
    ] = None,
    benchmark_name: Annotated[
        Optional[BenchmarkName],
        typer.Option("--benchmark-name", case_sensitive=False, help="Filter by benchmark family."),
    ] = None,
    metric_variant: Annotated[
        Optional[MetricVariant],
        typer.Option("--metric-variant", case_sensitive=False, help="Filter by metric variant."),
    ] = None,
    instance_id: Annotated[
        Optional[str],
        typer.Option("--instance-id", help="Filter by exact path-derived aggregate instance ID."),
    ] = None,
    instance_name: Annotated[
        Optional[str],
        typer.Option("--instance-name", help="Filter by exact stored instance name. Note that instance names have no uniqueness guaranty."),
    ] = None,
    paths_only: Annotated[
        bool,
        typer.Option("--paths-only/--no-paths-only", help="Print only matching file paths (one per line)."),
    ] = False,
    show_path: Annotated[
        bool,
        typer.Option("--show-path/--no-show-path", help="Include each matching file path in the table."),
    ] = False,
    summary: Annotated[
        bool,
        typer.Option("--summary/--no-summary", help="Append a recap with counts per problem / benchmark / metric / size."),
    ] = True,
) -> None:
    """List local benchmark instances available to `solve`.

    Mirrors the selection model of `solve`: positional INSTANCE_PATHS or a recursive
    scan of `--benchmarks-dir` filtered by --problem-type / --benchmark-name /
    --metric-variant / --instance-id / --instance-name. Use `--paths-only` to pipe into solve, e.g.

        mamut-routing list --problem-type CVRP --paths-only \\
            | xargs -r mamut-routing solve --time-limit-s 30
    """
    state: CLIState = ctx.obj
    candidate_paths = _iter_candidate_paths(state, list(instance_paths) if instance_paths else None)

    scanned_count = 0
    matched_count = 0
    counts_problem: dict[str, int] = {}
    counts_benchmark: dict[str, int] = {}
    counts_metric: dict[str, int] = {}
    counts_size: dict[int, int] = {}

    if not paths_only:
        base_header = (
            f"{'INSTANCE_ID':<64}  {'INSTANCE_NAME':<24}  {'PROBLEM':<6}  {'BENCHMARK':<12}  "
            f"{'METRIC':<10}  {'PLACE':<14}  {'n':>5}"
        )
        header = f"{base_header}  PATH" if show_path else base_header
        typer.echo(header)
        typer.echo("-" * len(header))

    resolver = BenchmarkLayoutResolver(state.benchmarks_dir)
    for path in candidate_paths:
        scanned_count += 1
        instance = load_benchmark_instance(path)
        record = _local_instance_record(path, instance, state.benchmarks_dir, resolver)
        if not _instance_matches(
            record,
            problem_type=problem_type,
            benchmark_name=benchmark_name,
            metric_variant=metric_variant,
            instance_id=instance_id,
            instance_name=instance_name,
        ):
            continue

        matched_count += 1
        if paths_only:
            typer.echo(str(path))
            continue

        problem_key = record.problem_type.value
        benchmark_key = record.benchmark_name
        metric_key = str(_enum_value(record.metric_variant) or "-")
        counts_problem[problem_key] = counts_problem.get(problem_key, 0) + 1
        counts_benchmark[benchmark_key] = counts_benchmark.get(benchmark_key, 0) + 1
        counts_metric[metric_key] = counts_metric.get(metric_key, 0) + 1
        counts_size[record.num_customers] = counts_size.get(record.num_customers, 0) + 1

        row = (
            f"{record.instance_id:<64}  "
            f"{record.instance_name:<24}  "
            f"{record.problem_type.value:<6}  "
            f"{record.benchmark_name:<12}  "
            f"{(_enum_value(record.metric_variant) or '-'):<10}  "
            f"{(record.place_slug or '-'):<14}  "
            f"{record.num_customers:>5}"
        )
        if show_path:
            row = f"{row}  {path}"
        typer.echo(row)

    if matched_count == 0:
        typer.echo(
            "No instances matched. Adjust filter flags or point --benchmarks-dir at a "
            "tree containing *.vrp.json files.",
            err=True,
        )
        raise typer.Exit(code=2)

    if paths_only or not summary:
        return

    def _fmt(counts: dict) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))

    typer.echo("")
    typer.echo("Summary:")
    typer.echo(f"  Scanned        : {scanned_count}")
    typer.echo(f"  Total          : {matched_count}")
    typer.echo(f"  Problem types  : {_fmt(counts_problem)}")
    typer.echo(f"  Benchmarks     : {_fmt(counts_benchmark)}")
    typer.echo(f"  Metric variants: {_fmt(counts_metric)}")
    if counts_size:
        size_fmt = ", ".join(f"n={size}: {count}" for size, count in sorted(counts_size.items()))
        typer.echo(f"  Customer sizes : {size_fmt}")


SOLVER_HELP = (
    "Solver to run: pyvrp (default; CVRP/VRPTW), kayros (TDVRP/TDVRPTW Duration), or an external "
    "executable -- filo2, ails2 (CVRP, euclidean variants), hgs-cvrp (CVRP, any metric; patched reader), "
    "lkh3 (CVRP/VRPTW) -- configured in the solvers config. `mamut-routing solvers list` shows what is "
    "available here."
)


@app.command("solve")
def solve_instances(
    ctx: typer.Context,
    instance_paths: Annotated[
        Optional[list[Path]],
        typer.Argument(
            help="One or more benchmark instance JSON paths. If omitted, --benchmarks-dir is "
            "scanned recursively for *.vrp.json files (and filter flags apply).",
        ),
    ] = None,
    problem_type: Annotated[
        Optional[ProblemType],
        typer.Option("--problem-type", case_sensitive=False, help="Filter by problem type."),
    ] = None,
    benchmark_name: Annotated[
        Optional[BenchmarkName],
        typer.Option("--benchmark-name", case_sensitive=False, help="Filter by benchmark family."),
    ] = None,
    metric_variant: Annotated[
        Optional[MetricVariant],
        typer.Option("--metric-variant", case_sensitive=False, help="Filter by metric variant."),
    ] = None,
    instance_id: Annotated[
        Optional[str],
        typer.Option("--instance-id", help="Filter by exact derived aggregate instance ID."),
    ] = None,
    instance_name: Annotated[
        Optional[str],
        typer.Option("--instance-name", help="Filter by exact stored instance name."),
    ] = None,
    solver: Annotated[str, typer.Option("--solver", help=SOLVER_HELP)] = "pyvrp",
    solvers_config: Annotated[Optional[Path], typer.Option("--solvers-config", help=SOLVERS_CONFIG_HELP)] = None,
    keep_artifacts: Annotated[
        Optional[Path],
        typer.Option(
            "--keep-artifacts",
            help="Keep each external solver's input, output and log under DIR/<instance-id>/ (debugging, audits).",
        ),
    ] = None,
    objective: Annotated[
        Optional[ObjectiveFunction],
        typer.Option(
            "--objective",
            case_sensitive=False,
            help="Objective. Default: MonoCost for CVRP/VRPTW (CVRP is always MonoCost), Duration for TD instances.",
        ),
    ] = None,
    time_limit_s: Annotated[
        int,
        typer.Option("--time-limit-s", min=1, help="Wall-clock budget per instance, in seconds."),
    ] = 15,
    seed: Annotated[int, typer.Option("--seed", help="Solver random seed.")] = 42,
    save_bks: Annotated[
        bool,
        typer.Option(
            "--save-bks/--no-save-bks",
            help="Store the solution as the instance's BKS when it strictly improves it (needs --authors). Off by default.",
        ),
    ] = False,
    authors: Annotated[
        Optional[str],
        typer.Option(
            "--authors",
            envvar="MAMUT_ROUTING_BKS_AUTHORS",
            help="Who to credit in a stored BKS (required with --save-bks; or set $MAMUT_ROUTING_BKS_AUTHORS).",
        ),
    ] = None,
    display: Annotated[
        bool,
        typer.Option("--display/--no-display", help="Forward PyVRP's per-iteration display to stdout."),
    ] = False,
    jobs: Annotated[
        Optional[int],
        typer.Option(
            "--jobs",
            min=1,
            help=(
                "Parallel solve workers for multi-instance selections. Defaults to max(1, CPU count - 2). "
                "Single-instance solves keep the time progress bar."
            ),
        ),
    ] = None,
) -> None:
    """Solve benchmark instances with a registered solver and check every result.

    Every solution is priced by the reference checker; COST is the checker's
    cost. With --save-bks --authors NAME, a solution that strictly beats the
    stored BKS (exact decimal costs; ties keep the incumbent) is written next
    to the instance. PyVRP needs the 'pyvrp' extra:
    pip install "mamut-routing-lib[pyvrp]".
    """
    from mamut_routing_lib.objectives import default_objective
    from mamut_routing_lib.solutions import submit_bks
    from mamut_routing_lib.solvers import (
        InstanceNotSupportedError,
        SolveMonitor,
        SolverUnavailableError,
        UnknownSolverError,
        get_solver,
        profile_instance,
        run_solver,
    )
    from mamut_routing_lib.solvers.external.config import SolverConfigError
    from mamut_routing_lib.solvers.external.process import terminate_all

    state: CLIState = ctx.obj

    try:
        solver_obj = get_solver(solver, config=solvers_config)
    except UnknownSolverError as error:
        raise typer.BadParameter(str(error), param_hint="--solver") from error
    except SolverConfigError as error:
        typer.echo(f"Error: solvers config: {error}", err=True)
        raise typer.Exit(code=2) from error
    availability = solver_obj.availability()
    if not availability.available:
        hint = f" ({availability.hint})" if availability.hint else ""
        typer.echo(f"Error: solver {solver_obj.name} is not available: {availability.reason}{hint}", err=True)
        raise typer.Exit(code=2)
    if save_bks and not (authors and authors.strip()):
        typer.echo(
            "Error: --save-bks needs --authors (who to credit in the BKS), or $MAMUT_ROUTING_BKS_AUTHORS.",
            err=True,
        )
        raise typer.Exit(code=2)

    candidate_paths = list(_iter_candidate_paths(state, list(instance_paths) if instance_paths else None))

    selected = _filter_loaded_instances(
        candidate_paths,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_variant,
        instance_id=instance_id,
        instance_name=instance_name,
        benchmarks_dir=state.benchmarks_dir,
    )
    if not selected:
        typer.echo(
            "No instances selected. Pass positional paths, adjust filter flags, or "
            "point --benchmarks-dir at a tree containing *.vrp.json files.",
            err=True,
        )
        raise typer.Exit(code=2)

    explicit_paths = bool(instance_paths)
    td_types = (ProblemType.TDVRP, ProblemType.TDVRPTW)
    time_dependent: list[LocalInstanceRecord] = []
    if not solver_obj.capabilities.time_dependent:
        time_dependent = [r for r in selected if r.problem_type in td_types]
    if time_dependent:
        if explicit_paths:
            typer.echo(
                f"Error: {len(time_dependent)} time-dependent instance(s) selected (e.g. "
                f"{time_dependent[0].instance_id}); {solver_obj.name} covers "
                f"{', '.join(sorted(p.value for p in solver_obj.capabilities.problem_types))} only "
                "(time-dependent instances: --solver kayros, or check them with mamut_routing_lib.td).",
                err=True,
            )
            raise typer.Exit(code=2)
        typer.secho(
            f"Warning: skipping {len(time_dependent)} time-dependent instance(s); "
            f"{solver_obj.name} covers static CVRP and VRPTW only.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        selected = [r for r in selected if r.problem_type not in td_types]

    unmaterialized = {id(r): path for r in selected if (path := _missing_arc_cost_sidecar(r)) is not None}
    error_rows: list[SolveSummaryRow] = []
    if unmaterialized:
        missing_records = [r for r in selected if id(r) in unmaterialized]
        selected = [r for r in selected if id(r) not in unmaterialized]
        if explicit_paths:
            error_rows.extend(
                SolveSummaryRow(
                    instance_id=r.instance_id,
                    instance_name=r.instance_name,
                    status="error",
                    cost=None,
                    route_count=0,
                    wall_time=0.0,
                    bks_label="-",
                    detail=f"distances sidecar not in the tree (sha256-pinned; materialize it first): {unmaterialized[id(r)]}",
                )
                for r in missing_records
            )
        else:
            typer.secho(
                f"Warning: skipping {len(missing_records)} instance(s) whose distances sidecar is sha256-pinned "
                f"but not in the tree (e.g. {missing_records[0].instance_id}); materialize it first.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    if objective is not None and objective != ObjectiveFunction.MONO_COST:
        cvrp_in_batch = [r for r in selected if r.problem_type is ProblemType.CVRP]
        if cvrp_in_batch and objective == ObjectiveFunction.HIERARCHICAL_VEHICLE_COST:
            raise typer.BadParameter(
                f"--objective {objective.value} is VRPTW-only but the selection includes "
                f"{len(cvrp_in_batch)} CVRP instance(s) (e.g. {cvrp_in_batch[0].instance_id}). "
                "Run CVRP and VRPTW in separate invocations.",
                param_hint="--objective",
            )

    objectives = {id(r): objective or default_objective(r.instance) for r in selected}
    for record in selected:
        _warn_on_non_base_objectives([record], objectives[id(record)])

    unsupported: list[tuple[LocalInstanceRecord, str]] = []
    for record in selected:
        support = solver_obj.supports(profile_instance(record.instance, record.path), objectives[id(record)])
        if not support.supported:
            unsupported.append((record, "; ".join(support.reasons)))
    if unsupported:
        skipped_ids = {id(record) for record, _ in unsupported}
        selected = [r for r in selected if id(r) not in skipped_ids]
        if explicit_paths:
            error_rows.extend(
                SolveSummaryRow(
                    instance_id=record.instance_id,
                    instance_name=record.instance_name,
                    status="error",
                    cost=None,
                    route_count=0,
                    wall_time=0.0,
                    bks_label="-",
                    detail=f"{solver_obj.name} cannot solve it: {reason}",
                )
                for record, reason in unsupported
            )
        else:
            typer.secho(
                f"Warning: skipping {len(unsupported)} instance(s) {solver_obj.name} cannot solve "
                f"(e.g. {unsupported[0][0].instance_id}: {unsupported[0][1]}).",
                fg=typer.colors.YELLOW,
                err=True,
            )

    if not selected and not error_rows:
        typer.echo("No solvable instances left after skipping.", err=True)
        raise typer.Exit(code=2)

    multi_instance = len(selected) >= 2
    if multi_instance and display:
        typer.echo(
            "Error: --display is only supported when solving a single selected instance. "
            "Narrow the selection or omit --display for parallel batch solves.",
            err=True,
        )
        raise typer.Exit(code=2)

    if multi_instance:
        duplicate_paths = _duplicate_resolved_paths(selected)
        if duplicate_paths:
            typer.echo(
                "Error: duplicate instance path(s) selected for a parallel solve: "
                f"{', '.join(str(path) for path in duplicate_paths)}",
                err=True,
            )
            raise typer.Exit(code=2)

    resolved_jobs = jobs if jobs is not None else _default_solve_jobs()
    effective_jobs = min(resolved_jobs, len(selected)) if selected else 1

    solve_message = (
        f"Solving {len(selected)} instance(s)  solver={solver_obj.name}  time_limit={time_limit_s}s  "
        f"seed={seed}  save_bks={save_bks}"
    )
    if multi_instance:
        solve_message = f"{solve_message}  jobs={effective_jobs}"
    typer.echo(solve_message)
    rows: list[SolveSummaryRow | None] = [None] * len(selected)

    monitors: list[Any] = []
    interrupted = threading.Event()
    workdirs = _artifact_workdirs(selected, keep_artifacts, f"{solver_obj.name}-seed{seed}")

    def _stop_everything() -> None:
        # Solvers in their own process group never see the terminal's Ctrl-C: stop them all now,
        # before any executor waits for its running solves.
        interrupted.set()
        for monitor in list(monitors):
            monitor.request_stop()
        terminate_all()

    def _solve_record(record: LocalInstanceRecord) -> SolveSummaryRow:
        run_objective = objectives[id(record)]
        monitor = SolveMonitor(None, interval_s=3600.0)
        monitors.append(monitor)
        if interrupted.is_set():
            monitor.request_stop()
        workdir = workdirs.get(id(record))

        def _solve_selected_instance() -> Any:
            return run_solver(
                solver_obj,
                record.path,
                time_limit_s=time_limit_s,
                seed=seed,
                objective_function=run_objective,
                instance=record.instance,
                display=display,
                monitor=monitor,
                workdir=workdir,
                keep_artifacts=workdir is not None,
                config=solvers_config,
            )

        if multi_instance:
            solved = _solve_selected_instance()
        else:
            solved = _run_with_time_progress(
                record=record,
                time_limit_s=time_limit_s,
                enabled=not display,
                operation=_solve_selected_instance,
                on_interrupt=_stop_everything,
            )

        result = solved.result
        if result.status == "cancelled" or interrupted.is_set():
            return SolveSummaryRow(
                instance_id=record.instance_id,
                instance_name=record.instance_name,
                status="error",
                cost=None,
                route_count=0,
                wall_time=result.wall_time,
                bks_label="-",
                detail="interrupted",
            )
        if result.status == "failed":
            return SolveSummaryRow(
                instance_id=record.instance_id,
                instance_name=record.instance_name,
                status="error",
                cost=None,
                route_count=0,
                wall_time=result.wall_time,
                bks_label="-",
                detail=result.error or "the solver failed",
            )
        detail = ""
        if solved.valid:
            status = "feasible"
        else:
            status = "infeasible"
            if solved.check is not None:
                detail = f"checker: {solved.validation_status}: {solved.check.error_message}"
            elif result.status == "timeout":
                detail = f"timeout: {result.error or 'the solver hit the hard timeout'} (no solution to check)"
            elif result.error:
                detail = result.error
            else:
                detail = "the solver returned no routes"
        bks_label = "-" if save_bks else "skipped"
        if save_bks and solved.valid:
            update = submit_bks(
                record.path,
                result.routes,
                authors=str(authors),
                objective_function=run_objective,
                metadata=solved.bks_metadata(),
                instance=record.instance,
            )
            bks_label = f"{update.action} (tie)" if update.tie else update.action
        return SolveSummaryRow(
            instance_id=record.instance_id,
            instance_name=record.instance_name,
            status=status,
            cost=solved.cost,
            route_count=len(result.routes),
            wall_time=result.wall_time,
            bks_label=bks_label,
            detail=detail,
        )

    def _solve_record_safe(record: LocalInstanceRecord) -> SolveSummaryRow:
        # One bad instance must not take the whole batch (and its summary) down.
        started = time.perf_counter()
        try:
            return _solve_record(record)
        except (InstanceNotSupportedError, SolverUnavailableError) as exc:
            detail = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported as an error row
            detail = f"{type(exc).__name__}: {exc}"
        return SolveSummaryRow(
            instance_id=record.instance_id,
            instance_name=record.instance_name,
            status="error",
            cost=None,
            route_count=0,
            wall_time=time.perf_counter() - started,
            bks_label="-",
            detail=detail,
        )

    try:
        if multi_instance:
            future_indices = {}
            with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
                for index, record in enumerate(selected):
                    future_indices[executor.submit(_solve_record_safe, record)] = index
                try:
                    with tqdm(
                        total=len(selected),
                        desc=f"Solving ({effective_jobs} jobs)",
                        unit="run",
                        leave=True,
                        dynamic_ncols=True,
                        file=sys.stderr,
                    ) as bar:
                        for future in as_completed(future_indices):
                            index = future_indices[future]
                            rows[index] = future.result()
                            bar.update(1)
                except KeyboardInterrupt:
                    for future in future_indices:
                        future.cancel()
                    _stop_everything()
                    raise
        elif selected:
            rows[0] = _solve_record_safe(selected[0])
    except KeyboardInterrupt:
        _stop_everything()
        finished = [row for row in rows if row is not None and row.detail != "interrupted"]
        message = "Interrupted: every running solve was stopped and wrote no BKS."
        written = [row for row in finished if row.bks_label.startswith(("created", "replaced"))]
        if written:
            message += f" {len(written)} solve(s) that finished before the interrupt wrote their BKS."
        typer.echo(message, err=True)
        raise typer.Exit(code=130)

    solved_rows = [row for row in rows if row is not None] + error_rows
    any_failure = any(row.status != "feasible" for row in solved_rows)

    header = (
        f"{'INSTANCE_ID':<64}  {'INSTANCE_NAME':<24}  {'STATUS':<10}  "
        f"{'COST':>14}  {'ROUTES':>6}  {'WALL_S':>7}  {'BKS':<14}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for row in solved_rows:
        if row.cost is None:
            cost_str = f"{'-':>14}"
        elif isinstance(row.cost, float) and not row.cost.is_integer():
            cost_str = f"{row.cost:>14.10g}"
        else:
            cost_str = f"{int(row.cost):>14}"
        typer.echo(
            f"{row.instance_id:<64}  {row.instance_name:<24}  {row.status:<10}  {cost_str}  {row.route_count:>6}  "
            f"{row.wall_time:>7.2f}  {row.bks_label:<14}"
        )

    for row in solved_rows:
        if row.status == "error":
            typer.echo(f"error: {row.instance_id}: {row.detail}", err=True)
        elif row.detail:
            typer.echo(f"{row.status}: {row.instance_id}: {row.detail}", err=True)
    counts = {status: sum(1 for row in solved_rows if row.status == status) for status in ("feasible", "infeasible", "error")}
    skipped_td = len(time_dependent) if not explicit_paths else 0
    skipped_sidecar = len(unmaterialized) if not explicit_paths else 0
    skipped_unsupported = len(unsupported) if not explicit_paths else 0
    typer.echo(
        f"Summary: feasible {counts['feasible']} · infeasible {counts['infeasible']} · errors {counts['error']} · "
        f"skipped (TD) {skipped_td} · skipped (sidecar) {skipped_sidecar} · skipped (unsupported) {skipped_unsupported}"
    )

    if any_failure:
        raise typer.Exit(code=1)


@dataclass(frozen=True)
class ExportJob:
    record: LocalInstanceRecord
    output_path: Path


def _export_output_path(record_path: Path, benchmarks_dir: Path, output_dir: Path | None, filename: str) -> Path:
    """Next to the source by default; under ``output_dir`` mirroring the path
    relative to ``benchmarks_dir`` (flat for files outside it)."""
    if output_dir is None:
        return record_path.with_name(filename)
    try:
        relative_parent = record_path.resolve().relative_to(benchmarks_dir.resolve()).parent
    except ValueError:
        relative_parent = Path()
    return output_dir / relative_parent / filename


def _run_export_job(job: ExportJob, options: VrpExportOptions, overwrite: bool, collection_root: Path | None) -> ExportResult:
    return export_instance_file(
        job.record.path,
        job.output_path,
        collection_root=collection_root,
        options=options,
        overwrite=overwrite,
    )


def _export_worker(args: tuple[ExportJob, VrpExportOptions, bool, Path | None]) -> ExportResult | Exception:
    try:
        return _run_export_job(*args)
    except Exception as exc:  # noqa: BLE001 - reported per instance in the summary table
        return exc


@export_app.command("vrp")
def export_vrp(
    ctx: typer.Context,
    instance_paths: Annotated[
        Optional[list[Path]],
        typer.Argument(
            help="One or more benchmark instance JSON paths. If omitted, --benchmarks-dir is "
            "scanned recursively for *.vrp.json files (and filter flags apply).",
        ),
    ] = None,
    problem_type: Annotated[
        Optional[ProblemType],
        typer.Option("--problem-type", case_sensitive=False, help="Filter by CVRP or VRPTW."),
    ] = None,
    benchmark_name: Annotated[
        Optional[BenchmarkName],
        typer.Option("--benchmark-name", case_sensitive=False, help="Filter by benchmark family."),
    ] = None,
    metric_variant: Annotated[
        Optional[MetricVariant],
        typer.Option("--metric-variant", case_sensitive=False, help="Filter by metric variant."),
    ] = None,
    instance_id: Annotated[
        Optional[str],
        typer.Option("--instance-id", help="Filter by exact path-derived aggregate instance ID."),
    ] = None,
    instance_name: Annotated[
        Optional[str],
        typer.Option("--instance-name", help="Filter by exact stored instance name."),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Write outputs under this directory, mirroring each path relative to --benchmarks-dir "
            "(flat for files outside it). Default: next to each source .vrp.json.",
        ),
    ] = None,
    export_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="'vrp' (CVRPLIB/TSPLIB .vrp; CVRP or CVRPTW sections) or 'solomon' "
            "(Solomon/Gehring-Homberger .txt, VRPTW with euclidean metric only).",
        ),
    ] = "vrp",
    edge_weight_type: Annotated[
        str,
        typer.Option(
            "--edge-weight-type",
            help="'EXPLICIT' writes the full cost matrix (faithful to the published costs). 'EUC_2D' writes "
            "coordinates only, for instances whose costs are a rounding of the Euclidean distance of their "
            "coordinates (not Dimacs2021, whose costs are floor(10 * d)); classic readers then use TSPLIB nint "
            "distances, which can differ from the published costs by that rounding.",
        ),
    ] = "EXPLICIT",
    comment: Annotated[
        Optional[str],
        typer.Option("--comment", help="Override the COMMENT line (default: reconstructed from the instance)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite existing outputs (default: report them as 'exists')."),
    ] = False,
    collection_root: Annotated[
        Optional[Path],
        typer.Option(
            "--collection-root",
            help="Collection root (directory holding mamut-collection.json) for slim instances copied "
            "out of their collection tree.",
        ),
    ] = None,
    jobs: Annotated[
        Optional[int],
        typer.Option("--jobs", min=1, help="Parallel workers for batches. Defaults to max(1, CPU count - 2)."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet/--no-quiet", help="Only print errors and the summary."),
    ] = False,
) -> None:
    """Export benchmark instances to the classic CVRPLIB .vrp format (or Solomon .txt).

    Selection mirrors `list` and `solve`: positional INSTANCE_PATHS or a recursive
    scan of `--benchmarks-dir` filtered by --problem-type / --benchmark-name /
    --metric-variant / --instance-id / --instance-name. Slim collection instances
    (Poryos2026, Mamut2026) hydrate their matrix from the sha-pinned distances
    sidecar. Time-dependent instances have no static matrix: explicit TD paths
    are an error, scanned ones are skipped with a warning.

        mamut-routing --benchmarks-dir ./benchmarks export vrp \\
            --problem-type CVRP --benchmark-name Mamut2026 --output-dir ./vrp-out
    """
    state: CLIState = ctx.obj
    export_format = export_format.lower()
    edge_weight_type = edge_weight_type.upper()
    if export_format not in EXPORT_FORMATS:
        raise typer.BadParameter(f"must be one of {', '.join(EXPORT_FORMATS)}", param_hint="--format")
    if edge_weight_type not in EDGE_WEIGHT_TYPES:
        raise typer.BadParameter(f"must be one of {', '.join(EDGE_WEIGHT_TYPES)}", param_hint="--edge-weight-type")
    options = VrpExportOptions(edge_weight_type=edge_weight_type, format=export_format, comment=comment)  # type: ignore[arg-type]

    explicit_paths = list(instance_paths) if instance_paths else None
    candidate_paths = list(_iter_candidate_paths(state, explicit_paths))
    selected = _filter_loaded_instances(
        candidate_paths,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_variant,
        instance_id=instance_id,
        instance_name=instance_name,
        benchmarks_dir=state.benchmarks_dir,
    )
    if not selected:
        typer.echo(
            "No instances selected. Pass positional paths, adjust filter flags, or "
            "point --benchmarks-dir at a tree containing *.vrp.json files.",
            err=True,
        )
        raise typer.Exit(code=2)

    time_dependent = [r for r in selected if r.problem_type in (ProblemType.TDVRP, ProblemType.TDVRPTW)]
    if time_dependent and explicit_paths:
        typer.echo(
            "Error: time-dependent instance(s) have no static matrix and cannot be exported as classic .vrp: "
            + ", ".join(str(r.path) for r in time_dependent),
            err=True,
        )
        raise typer.Exit(code=2)
    skipped_td = 0
    if time_dependent:
        skipped_td = len(time_dependent)
        typer.secho(
            f"Warning: skipping {skipped_td} time-dependent instance(s) (no static matrix to export).",
            fg=typer.colors.YELLOW,
            bold=True,
            err=True,
        )
        selected = [r for r in selected if r not in time_dependent]

    duplicate_paths = _duplicate_resolved_paths(selected)
    if duplicate_paths:
        typer.echo(
            "Error: duplicate instance path(s) selected: " + ", ".join(str(path) for path in duplicate_paths),
            err=True,
        )
        raise typer.Exit(code=2)

    resolved_output_dir = output_dir.expanduser().resolve() if output_dir is not None else None
    resolved_collection_root = collection_root.expanduser().resolve() if collection_root is not None else None
    jobs_list = [
        ExportJob(
            record=record,
            output_path=_export_output_path(
                record.path, state.benchmarks_dir, resolved_output_dir, export_filename(record.path.name, options)
            ),
        )
        for record in selected
    ]
    output_targets = [job.output_path.resolve() for job in jobs_list]
    if len(set(output_targets)) != len(output_targets):
        typer.echo("Error: two selected instances map to the same output path; use --output-dir.", err=True)
        raise typer.Exit(code=2)

    effective_jobs = min(jobs if jobs is not None else _default_solve_jobs(), len(jobs_list))
    if not quiet:
        typer.echo(
            f"Exporting {len(jobs_list)} instance(s)  format={export_format}  "
            f"edge_weight_type={edge_weight_type}  force={force}"
            + (f"  jobs={effective_jobs}" if len(jobs_list) > 1 else "")
        )

    outcomes: list[ExportResult | Exception | None] = [None] * len(jobs_list)
    worker_args = [(job, options, force, resolved_collection_root) for job in jobs_list]
    if len(jobs_list) > 1 and effective_jobs > 1:
        with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            futures = {executor.submit(_export_worker, args): index for index, args in enumerate(worker_args)}
            with tqdm(
                total=len(jobs_list),
                desc=f"Exporting ({effective_jobs} jobs)",
                unit="file",
                leave=False,
                dynamic_ncols=True,
                disable=quiet,
                file=sys.stderr,
            ) as bar:
                for future in as_completed(futures):
                    outcomes[futures[future]] = future.result()
                    bar.update(1)
    else:
        for index, args in enumerate(worker_args):
            outcomes[index] = _export_worker(args)

    counts = {"written": 0, "exists": 0, "unsupported": 0, "error": 0}
    header = f"{'INSTANCE_ID':<64}  {'PROBLEM':<7}  {'n':>5}  {'STATUS':<11}  OUTPUT"
    if not quiet:
        typer.echo(header)
        typer.echo("-" * len(header))
    for job, outcome in zip(jobs_list, outcomes):
        record = job.record
        if isinstance(outcome, Exception):
            status, detail = "error", f"{type(outcome).__name__}: {outcome}"
        else:
            assert outcome is not None
            status, detail = outcome.status, (str(outcome.output_path) if outcome.status == "written" else outcome.message)
        counts[status] += 1
        if quiet and status in ("written", "exists"):
            continue
        typer.echo(
            f"{record.instance_id:<64}  {record.problem_type.value:<7}  {record.num_customers:>5}  "
            f"{status:<11}  {detail}",
            err=status == "error",
        )

    typer.echo("")
    typer.echo("Summary:")
    typer.echo(f"  Written        : {counts['written']}")
    typer.echo(f"  Existing       : {counts['exists']}")
    typer.echo(f"  Unsupported    : {counts['unsupported']}")
    typer.echo(f"  Skipped (TD)   : {skipped_td}")
    typer.echo(f"  Errors         : {counts['error']}")
    if counts["error"] or counts["unsupported"]:
        raise typer.Exit(code=1)


@remote_app.command("manifest")
def remote_show_manifest(ctx: typer.Context) -> None:
    """Print the parsed release manifest as JSON."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2))



def _reprice_one(bks_path: str, options: dict[str, Any]) -> dict[str, Any]:
    from mamut_routing_lib.td.reprice import reprice_td_bks

    return reprice_td_bks(bks_path, **options).as_dict()


@bks_app.command("reprice-td")
def reprice_td(
    paths: Annotated[
        list[Path],
        typer.Argument(help="TD BKS files (*.bks.Duration.json, *.bks.FleetCostDuration.json) or directories to scan."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--write", help="Report what would change without writing (default: write)."),
    ] = False,
    report: Annotated[
        Optional[Path],
        typer.Option("--report", help="Write the per-file results as a JSON list to this path."),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option("--jobs", min=1, help="Worker processes (each file loads its own instance)."),
    ] = 1,
    atf_dir: Annotated[
        Optional[Path],
        typer.Option("--atf-dir", help="Fallback directory for ATF sidecars that are pinned but not committed."),
    ] = None,
    allow_missing_sidecars: Annotated[
        bool,
        typer.Option("--allow-missing-sidecars", help="Do not fail when a sidecar is missing (the file is skipped)."),
    ] = False,
    verify_pins: Annotated[
        bool,
        typer.Option("--verify-pins", help="Check the sha256 pins of every sidecar read (slower)."),
    ] = False,
    note_date: Annotated[
        Optional[str],
        typer.Option("--note-date", help="Date written in metadata.repriced and stamp notes (default: today, UTC)."),
    ] = None,
) -> None:
    """Re-price stored TD BKS under the current checker contract (routes untouched).

    Rewrites a BKS only when a checker output changed (cost, validated_cost,
    route_durations, route_departure_times); a moved cost is recorded in
    metadata.repriced and an optimality stamp gets its proven_optimum updated
    with a note. Exit status 1 when a file is infeasible, errors, would break a
    stamp, or (without --allow-missing-sidecars) misses a sidecar.
    """
    from mamut_routing_lib.td.reprice import iter_td_bks_files

    files = [str(path) for path in iter_td_bks_files(list(paths))]
    if not files:
        typer.echo("No TD BKS files found.", err=True)
        raise typer.Exit(code=2)
    options: dict[str, Any] = {
        "dry_run": dry_run,
        "atf_dir": str(atf_dir) if atf_dir is not None else None,
        "verify_sidecar_sha256": verify_pins,
        "note_date": note_date,
    }
    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(files), desc="reprice-td", unit="bks", disable=len(files) < 2)
    if jobs == 1:
        for path in files:
            results.append(_reprice_one(path, options))
            progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_reprice_one, path, options): path for path in files}
            for future in as_completed(futures):
                results.append(future.result())
                progress.update(1)
    progress.close()
    results.sort(key=lambda item: item["bks_path"])

    counts: dict[str, int] = {}
    for item in results:
        counts[item["action"]] = counts.get(item["action"], 0) + 1
    cost_moves = [item for item in results if "cost" in item["changed_fields"]]
    typer.echo(f"{len(results)} TD BKS files: " + ", ".join(f"{action} {n}" for action, n in sorted(counts.items())))
    typer.echo(f"cost moved: {len(cost_moves)} (stamped: {sum(1 for item in cost_moves if item['stamped'])})")
    for item in cost_moves:
        if abs(item["new_cost"] - item["previous_cost"]) > 1e-6:
            typer.echo(f"  {item['bks_path']}: {item['previous_cost']!r} -> {item['new_cost']!r}")
    failures = [item for item in results if item["action"] in {"infeasible", "error", "stamp-conflict"}]
    missing = [item for item in results if item["action"] == "skipped-missing-sidecar"]
    for item in failures + missing:
        typer.echo(f"{item['action']}: {item['bks_path']}: {item['detail']}", err=True)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    if failures or (missing and not allow_missing_sidecars):
        raise typer.Exit(code=1)


@solvers_app.command("list")
def solvers_list(
    solvers_config: Annotated[Optional[Path], typer.Option("--solvers-config", help=SOLVERS_CONFIG_HELP)] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the solvers as JSON.")] = False,
) -> None:
    """List every registered solver: availability here, version, what it solves, and how to enable it."""
    from mamut_routing_lib.solvers import list_solvers
    from mamut_routing_lib.solvers.external.config import SolverConfigError, resolve_config_path

    try:
        solvers = list_solvers(config=solvers_config)
        config_path = resolve_config_path(solvers_config)
    except SolverConfigError as error:
        typer.echo(f"Error: solvers config: {error}", err=True)
        raise typer.Exit(code=2) from error
    rows = []
    for solver_obj in solvers:
        availability = solver_obj.availability()
        descriptor = getattr(solver_obj, "descriptor", None)
        rows.append(
            {
                "name": solver_obj.name,
                "label": solver_obj.label,
                "available": availability.available,
                "reason": availability.reason,
                "hint": availability.hint,
                "version": availability.version,
                "executable": availability.executable,
                "source": getattr(descriptor, "source", "builtin"),
                "capabilities": solver_obj.capabilities.to_json(),
            }
        )
    if as_json:
        typer.echo(json.dumps({"config": str(config_path) if config_path else None, "solvers": rows}, indent=1))
        return
    typer.echo(f"solvers config: {config_path or 'none'}")
    header = f"{'NAME':<12}  {'AVAILABLE':<9}  {'VERSION':<12}  {'SOLVES':<52}  NOTE"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        solves = "; ".join(
            f"{problem} {'/'.join(objectives)}" for problem, objectives in row["capabilities"]["objectives"].items()
        )
        note = row["reason"] or ""
        if row["hint"]:
            note = f"{note} -- {row['hint']}" if note else row["hint"]
        typer.echo(
            f"{row['name']:<12}  {'yes' if row['available'] else 'no':<9}  {str(row['version'] or '-'):<12}  "
            f"{solves:<52}  {note}"
        )


# --------------------------------------------------------------------------- #
# External solutions: check and submit
# --------------------------------------------------------------------------- #

BKS_AUTHORS_ENV = "MAMUT_ROUTING_BKS_AUTHORS"

_IDS_HELP = (
    "How stop ids are numbered in the solution file: 'auto' (1..n with 0 as a depot marker, "
    "unless the file covers 0..n-1 exactly), 'one-based' or 'zero-based'."
)


def _load_external_solution(instance_path: Path, solution_path: Path, ids: str) -> tuple[Any, Any]:
    """(instance, ParsedSolution) or exit 2 with the reason."""
    from mamut_routing_lib.solutions import SolutionFormatError, read_solution_file

    if ids not in ("auto", "one-based", "zero-based"):
        raise typer.BadParameter("use auto, one-based or zero-based", param_hint="--ids")
    try:
        instance = load_benchmark_instance(instance_path)
    except Exception as error:  # noqa: BLE001 - reported as a usage error
        typer.echo(f"Error: cannot load {instance_path}: {error}", err=True)
        raise typer.Exit(code=2) from error
    try:
        parsed = read_solution_file(solution_path, instance, convention=ids)  # type: ignore[arg-type]
    except (SolutionFormatError, OSError) as error:
        typer.echo(f"Error: {solution_path}: {error}", err=True)
        raise typer.Exit(code=2) from error
    if ids == "auto" and parsed.ids == "zero-based":
        typer.secho(
            f"Note: {solution_path.name} numbers its stops 0..{instance.num_customers - 1}: read as 0-based customer ids "
            "(pass --ids one-based if 0 is the depot).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    return instance, parsed


def _resolve_objective(instance: Any, objective: ObjectiveFunction | None) -> ObjectiveFunction:
    """The family's published objective, or ``objective`` if it applies to the instance (else exit 2)."""
    from mamut_routing_lib.objectives import base_objective, require_supported_objective

    if objective is None:
        return base_objective(instance)
    try:
        require_supported_objective(instance, objective)
    except ValueError as error:
        typer.echo(f"Error: --objective {error}.", err=True)
        raise typer.Exit(code=2) from error
    _warn_objective(instance, objective)
    return objective


def _warn_objective(instance: Any, objective: ObjectiveFunction) -> None:
    from mamut_routing_lib.objectives import base_objective

    published = base_objective(instance)
    if published != objective:
        typer.secho(
            f"Warning: {instance.instance_name} publishes its BKS under {published.value}; "
            f"using {objective.value}.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _exact_cost_text(instance: Any, instance_path: Path, routes: list[list[int]]) -> str | None:
    """The static checker's exact decimal cost (``204030.827``, not the float sum ``204030.82699999996``)."""
    from mamut_routing_lib.artifacts import hydrate_collection_instance
    from mamut_routing_lib.checker import compute_exact_solution_cost
    from mamut_routing_lib.solutions import is_time_dependent

    if is_time_dependent(instance):
        return None
    try:
        exact = compute_exact_solution_cost(hydrate_collection_instance(instance, instance_path), routes)
    except Exception:  # noqa: BLE001 - the float cost is still printed
        return None
    places = next((k for k in range(31) if (10**k) % exact.denominator == 0), None)
    if places is None:  # not a finite decimal: cannot come from decimal arc costs
        return None
    scaled = abs(exact.numerator) * (10**places // exact.denominator)
    whole, fraction = divmod(scaled, 10**places)
    text = f"{whole}.{fraction:0{places}d}".rstrip("0").rstrip(".") if places else str(whole)
    return f"-{text}" if exact < 0 else text


def _number(value: float) -> str:
    """A cost as written: integers without a decimal point, floats in full (shortest round-trip)."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _meta_pairs(pairs: list[str] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--meta")
        try:
            values[key.strip()] = json.loads(raw)
        except ValueError:
            values[key.strip()] = raw
    return values


@app.command("check")
def check_external_solution(
    instance_path: Annotated[Path, typer.Argument(help="The instance (.vrp.json), static or time-dependent.")],
    solution_path: Annotated[
        Path,
        typer.Argument(help="Solution file: CVRPLIB .sol (Route #k / Cost), JSON routes, or an LKH-3 MTSP solution."),
    ],
    objective: Annotated[
        Optional[ObjectiveFunction],
        typer.Option("--objective", case_sensitive=False, help="Objective (default: the family's published one)."),
    ] = None,
    ids: Annotated[str, typer.Option("--ids", help=_IDS_HELP)] = "auto",
    as_json: Annotated[bool, typer.Option("--json", help="Print the verdict as JSON.")] = False,
) -> None:
    """Check an externally found solution with the reference checker.

    The routes are mapped onto the instance's ids (depot 0, customers 1..n) and
    priced by the same checker that validates every BKS, static or
    time-dependent. A declared cost that differs from the checker's is reported,
    not trusted. Exit status 0 when valid, 1 when the checker rejects the routes,
    2 when the file cannot be read or does not fit the instance.
    """
    from mamut_routing_lib.solutions import check_routes

    instance, parsed = _load_external_solution(instance_path, solution_path, ids)
    resolved = _resolve_objective(instance, objective)
    result = check_routes(instance, instance_path, parsed.routes, resolved)
    valid = result.is_valid()
    mismatch = (
        valid
        and parsed.declared_cost is not None
        and result.routing_cost is not None
        and abs(float(parsed.declared_cost) - float(result.routing_cost)) > 1e-6 * max(1.0, abs(float(result.routing_cost)))
    )
    payload = {
        "valid": valid,
        "status": result.status.value,
        "error": result.error_message or None,
        "cost": result.routing_cost,
        "exact_cost": _exact_cost_text(instance, instance_path, parsed.routes) if valid else None,
        "num_routes": result.num_routes if valid else len(parsed.routes),
        "declared_cost": parsed.declared_cost,
        "declared_cost_matches": None if parsed.declared_cost is None or not valid else not mismatch,
        "objective_function": resolved.value,
        "format": parsed.format,
        "ids": parsed.ids,
        "instance_name": instance.instance_name,
        "problem_type": _problem_type_from_instance(instance).value,
        "routes": parsed.routes,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=1))
    else:
        typer.echo(f"instance   {instance.instance_name} ({payload['problem_type']})")
        typer.echo(f"solution   {solution_path} ({parsed.format}, {parsed.ids} ids, {len(parsed.routes)} routes)")
        typer.echo(f"objective  {resolved.value}")
        typer.echo(f"verdict    {result.status.value}" + (f": {result.error_message}" if result.error_message else ""))
        if valid:
            declared = "" if parsed.declared_cost is None else f" (declared {_number(parsed.declared_cost)})"
            typer.echo(f"cost       {payload['exact_cost'] or result.routing_cost}{declared}")
    if mismatch:
        typer.secho(
            f"Warning: the file declares cost {_number(parsed.declared_cost)} but the checker gives {result.routing_cost} "
            "(another metric variant, rounding, or a different instance).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    if not valid:
        raise typer.Exit(code=1)


@bks_app.command("submit")
def bks_submit(
    instance_path: Annotated[Path, typer.Argument(help="The instance (.vrp.json) inside a benchmark tree.")],
    solution_path: Annotated[
        Path,
        typer.Argument(help="Solution file: CVRPLIB .sol (Route #k / Cost), JSON routes, or an LKH-3 MTSP solution."),
    ],
    authors: Annotated[
        Optional[str],
        typer.Option(
            "--authors",
            envvar=BKS_AUTHORS_ENV,
            help="Who found the solution, as it should be credited (required; or set $MAMUT_ROUTING_BKS_AUTHORS).",
        ),
    ] = None,
    objective: Annotated[
        Optional[ObjectiveFunction],
        typer.Option("--objective", case_sensitive=False, help="Objective (default: the family's published one)."),
    ] = None,
    method: Annotated[Optional[str], typer.Option("--method", help="Method label, e.g. 'lkh3' or a paper reference.")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Seed of the run that found it.")] = None,
    time_limit_s: Annotated[Optional[float], typer.Option("--time-limit-s", help="Budget of that run, in seconds.")] = None,
    solver_version: Annotated[Optional[str], typer.Option("--solver-version", help="Solver version or commit.")] = None,
    meta: Annotated[
        Optional[list[str]],
        typer.Option("--meta", help="Extra metadata as KEY=VALUE (JSON values are decoded); repeatable."),
    ] = None,
    ids: Annotated[str, typer.Option("--ids", help=_IDS_HELP)] = "auto",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report what the store would do without writing.")] = False,
) -> None:
    """Propose an externally found solution as the BKS of an instance.

    The solution is checked, then stored next to the instance as
    <base>.bks.<Objective>.json only if it strictly beats the stored BKS (exact
    decimal costs for static instances; the TD checker for time-dependent ones).
    A tie keeps the incumbent. Exit status 0 for created / replaced / kept, 1
    when the checker rejects the solution, 2 on a usage error.
    """
    from mamut_routing_lib.solutions import preview_bks_update, submit_bks

    instance, parsed = _load_external_solution(instance_path, solution_path, ids)
    resolved = _resolve_objective(instance, objective)
    if dry_run:
        preview = preview_bks_update(instance_path, parsed.routes, objective_function=resolved, instance=instance)
        typer.echo(json.dumps(preview, indent=1))
        if preview["action"] == "invalid":
            raise typer.Exit(code=1)
        return
    if not authors or not authors.strip():
        typer.echo(f"Error: --authors is required (or set ${BKS_AUTHORS_ENV}).", err=True)
        raise typer.Exit(code=2)
    metadata: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "time_limit_s": time_limit_s,
        "solver_version": solver_version,
        "submitted_from": solution_path.name,
        **_meta_pairs(meta),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    try:
        update = submit_bks(
            instance_path,
            parsed.routes,
            authors=authors,
            objective_function=resolved,
            metadata=metadata,
            instance=instance,
        )
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        json.dumps(
            {
                "action": update.action,
                "tie": update.tie,
                "path": str(update.path),
                "previous_path": str(update.previous_path) if update.previous_path else None,
                "candidate_cost": update.candidate_cost,
                "candidate_num_routes": update.candidate_num_routes,
                "objective_function": resolved.value,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    app()

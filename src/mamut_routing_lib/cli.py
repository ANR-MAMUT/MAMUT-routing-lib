from __future__ import annotations

import json
import os
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Optional, TypeVar

import typer
from tqdm import tqdm

from mamut_routing_lib.artifacts import (
    AnyBenchmarkInstance,
    DEFAULT_BENCHMARKS_ROOT_ENV,
    DEFAULT_MAMUT_ROUTING_ROOT_ENV,
    build_instance_id,
    load_benchmark_instance,
    parse_layout,
)
from mamut_routing_lib.enums import BenchmarkName, MetricVariant, ObjectiveFunction, ProblemType
from mamut_routing_lib.models import BenchmarkInstanceCVRP
from mamut_routing_lib.remote import (
    DEFAULT_GITHUB_TOKEN_ENV,
    DEFAULT_RELEASE_REPO_ENV,
    GitHubReleaseClient,
    GitHubReleaseSource,
    ReleaseArchiveAsset,
    ReleaseArchiveManifest,
    ReleaseArchiveScope,
    compute_sha256,
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


@dataclass(frozen=True)
class SolveSummaryRow:
    instance_id: str
    instance_name: str
    status: str
    cost: int | float | None
    route_count: int
    wall_time: float
    bks_label: str


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
        typer.Option("--tag", help="Release tag (e.g. v0.0.1). Defaults to the latest release."),
    ] = None,
) -> None:
    root_state: CLIState = ctx.find_root().obj
    resolved_repo = repo or os.getenv(DEFAULT_RELEASE_REPO_ENV) or "ANR-MAMUT/MAMUT-routing"
    parts = resolved_repo.split("/")
    if len(parts) != 2 or not all(parts):
        typer.echo(
            f"Error: --repo must be in 'owner/name' format, got: {resolved_repo!r}. "
            f"Example: --repo ANR-MAMUT/MAMUT-routing-dummy",
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
    scope: ReleaseArchiveScope | None,
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
    if scope is None and problem_type is None and benchmark_name is None:
        return []
    return manifest.select_assets(
        scope=scope,
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
    scope: Annotated[Optional[ReleaseArchiveScope], typer.Option("--scope", case_sensitive=False)] = None,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
) -> None:
    """List benchmark archives available in the remote release manifest."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    assets = manifest.select_assets(
        scope=scope,
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
            f"{(asset.problem_type.value if asset.problem_type else '-'):<6}  "
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
    scope: Annotated[Optional[ReleaseArchiveScope], typer.Option("--scope", case_sensitive=False)] = None,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
    select_all: Annotated[bool, typer.Option("--all", help="Download every asset in the manifest.")] = False,
    extract: Annotated[bool, typer.Option("--extract/--no-extract", help="Extract zip archives after download.")] = True,
) -> None:
    """Download (and optionally extract) one or more benchmark archives."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    selected = _select_assets(
        manifest,
        filenames=filenames or [],
        select_all=select_all,
        scope=scope,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )
    if not selected:
        typer.echo("No assets selected. Use positional filenames, --scope/--problem-type/--benchmark-name, or --all.", err=True)
        raise typer.Exit(code=2)

    state.benchmarks_dir.mkdir(parents=True, exist_ok=True)
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

            destination = client.download_asset(
                asset,
                state.benchmarks_dir,
                extract=extract,
                progress_callback=_on_progress,
            )
        typer.echo(f"  -> {destination}")


@remote_app.command("verify")
def remote_verify_local(
    ctx: typer.Context,
    scope: Annotated[Optional[ReleaseArchiveScope], typer.Option("--scope", case_sensitive=False)] = None,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
) -> None:
    """Verify sha256 of locally downloaded archives against the remote manifest."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    assets = manifest.select_assets(
        scope=scope,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )

    typer.echo(f"Verifying {len(assets)} asset(s) under {state.benchmarks_dir}")
    has_failure = False
    for asset in assets:
        local_path = state.benchmarks_dir / asset.filename
        if not local_path.exists():
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
    yield from sorted(state.benchmarks_dir.rglob("*.vrp.json"))


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
    if isinstance(instance, BenchmarkInstanceCVRP):
        return ProblemType.CVRP
    metadata = getattr(instance, "metadata", None)
    problem_type = _metadata_value(metadata, "problem_type")
    if problem_type is not None:
        return ProblemType(problem_type)
    return ProblemType.VRPTW


def _resolve_layout_under(path: Path, benchmarks_dir: Path):
    """Return the parsed layout if `path` lives under `benchmarks_dir` and matches one
    of the supported layouts, otherwise None.
    """
    try:
        relative = path.resolve().relative_to(benchmarks_dir.resolve())
    except ValueError:
        return None
    try:
        return parse_layout(relative, path)
    except ValueError:
        return None


def _local_instance_record(path: Path, instance: "AnyBenchmarkInstance", benchmarks_dir: Path) -> LocalInstanceRecord:
    metadata = getattr(instance, "metadata", None)
    instance_name = str(getattr(instance, "instance_name"))
    problem_type = _problem_type_from_instance(instance)
    benchmark_name = _coerce_benchmark_name(getattr(instance, "benchmark_name"))
    metric_variant = _coerce_metric_variant(_metadata_value(metadata, "metric_variant"))
    place_slug = _metadata_value(metadata, "place_slug")
    num_customers = int(getattr(instance, "num_customers"))

    layout = _resolve_layout_under(path, benchmarks_dir)
    if layout is not None:
        # Path overrides metadata for fields the path actually carries. Historical
        # 4-part layouts have neither metric_variant nor place_slug — fall back to
        # whatever metadata supplies (e.g. enriched Dimacs/Sintef instances now
        # carry `metric_variant: "euclidean"` in their metadata).
        problem_type = layout.problem_type
        benchmark_name = layout.benchmark_name
        if layout.metric_variant is not None:
            metric_variant = layout.metric_variant
        if layout.place_slug is not None:
            place_slug = layout.place_slug
        metric_for_id = layout.metric_variant
        place_for_id = layout.place_slug
    else:
        metric_for_id = metric_variant if place_slug is not None else None
        place_for_id = place_slug

    instance_id = build_instance_id(
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_for_id,
        place_slug=place_for_id,
        num_customers=num_customers,
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
    for path in paths:
        instance = load_benchmark_instance(path)
        record = _local_instance_record(path, instance, benchmarks_dir)
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
        typer.echo(
            "Warning: "
            f"{record.instance_id} belongs to {record.benchmark_name}, whose base objective is "
            f"{base_objective.value}; solving with {objective.value}.",
            err=True,
        )


def _run_with_time_progress(
    *,
    record: LocalInstanceRecord,
    time_limit_s: int,
    enabled: bool,
    operation: Callable[[], T],
) -> T:
    if not enabled:
        return operation()

    desc = record.instance_id if len(record.instance_id) <= 48 else f"{record.instance_id[:45]}..."
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        started_at = time.monotonic()
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

        return future.result()


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

    for path in candidate_paths:
        scanned_count += 1
        instance = load_benchmark_instance(path)
        record = _local_instance_record(path, instance, state.benchmarks_dir)
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
        typer.Option("--instance-id", help="Filter by exact derived aggregate instance ID."),
    ] = None,
    instance_name: Annotated[
        Optional[str],
        typer.Option("--instance-name", help="Filter by exact stored instance name."),
    ] = None,
    objective: Annotated[
        ObjectiveFunction,
        typer.Option(
            "--objective",
            case_sensitive=False,
            help="VRPTW objective. CVRP forces MonoCost regardless.",
        ),
    ] = ObjectiveFunction.MONO_COST,
    time_limit_s: Annotated[
        int,
        typer.Option("--time-limit-s", min=1, help="Wall-clock budget per instance, in seconds."),
    ] = 15,
    seed: Annotated[int, typer.Option("--seed", help="Solver random seed.")] = 42,
    save_bks: Annotated[
        bool,
        typer.Option("--save-bks/--no-save-bks", help="Write a BKS file next to the instance if improved."),
    ] = True,
    display: Annotated[
        bool,
        typer.Option("--display/--no-display", help="Forward PyVRP's per-iteration display to stdout."),
    ] = False,
) -> None:
    """Solve one or more benchmark instances with custom PyVRP's HGS-based solver variants.

    Requires the 'pyvrp' optional dependency:
    pip install "mamut-routing-lib[pyvrp]"
    """
    try:
        from mamut_routing_lib.solvers.pyvrp import solve_and_update_bks, solve_instance
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2)

    state: CLIState = ctx.obj

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

    if objective != ObjectiveFunction.MONO_COST:
        cvrp_in_batch = [r for r in selected if isinstance(r.instance, BenchmarkInstanceCVRP)]
        if cvrp_in_batch:
            raise typer.BadParameter(
                f"--objective {objective.value} is VRPTW-only but the selection includes "
                f"{len(cvrp_in_batch)} CVRP instance(s) (e.g. {cvrp_in_batch[0].instance_id}). "
                "Run CVRP and VRPTW in separate invocations.",
                param_hint="--objective",
            )

    _warn_on_non_base_objectives(selected, objective)

    typer.echo(f"Solving {len(selected)} instance(s)  time_limit={time_limit_s}s  seed={seed}  save_bks={save_bks}")
    any_failure = False
    rows: list[SolveSummaryRow] = []
    for record in selected:
        path = record.path
        instance = record.instance
        run_objective = objective

        def _solve_selected_instance() -> tuple[Any, Any]:
            if save_bks:
                return solve_and_update_bks(
                    instance,
                    instance_path=path,
                    time_limit_s=time_limit_s,
                    seed=seed,
                    objective_function=run_objective,
                    display=display,
                )
            method_result = solve_instance(
                instance,
                time_limit_s=time_limit_s,
                seed=seed,
                objective_function=run_objective,
                display=display,
            )
            return method_result, None

        method_result, bks_update = _run_with_time_progress(
            record=record,
            time_limit_s=time_limit_s,
            enabled=not display,
            operation=_solve_selected_instance,
        )

        if method_result.solver_is_feasible:
            status = "feasible"
            cost = method_result.solver_cost
        else:
            status = "infeasible"
            cost = None
            any_failure = True

        bks_label = bks_update.action if bks_update is not None else ("-" if save_bks else "skipped")
        rows.append(
            SolveSummaryRow(
                instance_id=record.instance_id,
                instance_name=record.instance_name,
                status=status,
                cost=cost,
                route_count=method_result.route_count,
                wall_time=method_result.wall_time,
                bks_label=bks_label,
            )
        )

    header = (
        f"{'INSTANCE_ID':<64}  {'INSTANCE_NAME':<24}  {'STATUS':<10}  "
        f"{'COST':>10}  {'ROUTES':>6}  {'WALL_S':>7}  {'BKS':<14}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for row in rows:
        cost_str = f"{row.cost:>10}" if row.cost is not None else f"{'-':>10}"
        typer.echo(
            f"{row.instance_id:<64}  {row.instance_name:<24}  {row.status:<10}  {cost_str}  {row.route_count:>6}  "
            f"{row.wall_time:>7.2f}  {row.bks_label:<14}"
        )

    if any_failure:
        raise typer.Exit(code=1)


@remote_app.command("manifest")
def remote_show_manifest(ctx: typer.Context) -> None:
    """Print the parsed release manifest as JSON."""
    state: RemoteCLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()

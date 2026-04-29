from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from tqdm import tqdm

from mamut_routing_lib.artifacts import (
    AnyBenchmarkInstance,
    DEFAULT_BENCHMARKS_ROOT_ENV,
    DEFAULT_MAMUT_ROUTING_ROOT_ENV,
    load_benchmark_instance,
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
        try:
            import tomllib

            pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
            pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            return str(pyproject_data["project"]["version"])
        except Exception:
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


def _resolve_candidate_paths(state: "CLIState", instance_paths: list[Path] | None) -> list[Path]:
    if instance_paths:
        candidate_paths = [Path(p).expanduser().resolve() for p in instance_paths]
        missing = [p for p in candidate_paths if not p.is_file()]
        if missing:
            typer.echo(
                f"Error: instance file(s) not found: {', '.join(str(p) for p in missing)}",
                err=True,
            )
            raise typer.Exit(code=2)
        return candidate_paths

    if not state.benchmarks_dir.is_dir():
        typer.echo(
            f"Error: --benchmarks-dir does not exist: {state.benchmarks_dir}. Pass positional "
            f"INSTANCE_PATHS, specify a valid --benchmarks-dir or fetch archives first.",
            err=True,
        )
        raise typer.Exit(code=2)
    return sorted(state.benchmarks_dir.rglob("*.vrp.json"))


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _instance_descriptor(instance: "AnyBenchmarkInstance") -> dict[str, str | int | None]:
    is_cvrp = isinstance(instance, BenchmarkInstanceCVRP)
    metadata = getattr(instance, "metadata", None)
    return {
        "instance_id": getattr(instance, "instance_id", None) or getattr(instance, "instance_name", None),
        "problem_type": "CVRP" if is_cvrp else "VRPTW",
        "benchmark_name": _enum_value(getattr(instance, "benchmark_name", None)),
        "metric_variant": _enum_value(getattr(metadata, "metric_variant", None) if metadata else None),
        "place_slug": getattr(metadata, "place_slug", None) if metadata else None,
        "num_customers": getattr(instance, "num_customers", None),
    }


def _filter_loaded_instances(
    paths: list[Path],
    *,
    problem_type: ProblemType | None,
    benchmark_name: BenchmarkName | None,
    metric_variant: MetricVariant | None,
    instance_id: str | None,
) -> list[tuple[Path, "AnyBenchmarkInstance"]]:
    selected: list[tuple[Path, AnyBenchmarkInstance]] = []
    for path in paths:
        instance = load_benchmark_instance(path)

        if problem_type is not None:
            is_cvrp = isinstance(instance, BenchmarkInstanceCVRP)
            inst_problem = ProblemType.CVRP if is_cvrp else ProblemType.VRPTW
            if inst_problem != problem_type:
                continue

        if benchmark_name is not None:
            inst_benchmark = getattr(instance, "benchmark_name", None)
            if inst_benchmark is None or inst_benchmark != benchmark_name:
                continue

        if metric_variant is not None:
            metadata = getattr(instance, "metadata", None)
            inst_metric = getattr(metadata, "metric_variant", None) if metadata else None
            if inst_metric is None or inst_metric != metric_variant:
                continue

        if instance_id is not None:
            inst_id = getattr(instance, "instance_id", None) or getattr(instance, "instance_name", None)
            if inst_id != instance_id:
                continue

        selected.append((path, instance))
    return selected


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
        typer.Option("--instance-id", help="Filter by exact instance id."),
    ] = None,
    paths_only: Annotated[
        bool,
        typer.Option("--paths-only/--no-paths-only", help="Print only matching file paths (one per line)."),
    ] = False,
    summary: Annotated[
        bool,
        typer.Option("--summary/--no-summary", help="Append a recap with counts per problem / benchmark / metric / size."),
    ] = True,
) -> None:
    """List local benchmark instances available to `solve`.

    Mirrors the selection model of `solve`: positional INSTANCE_PATHS or a recursive
    scan of `--benchmarks-dir` filtered by --problem-type / --benchmark-name /
    --metric-variant / --instance-id. Use `--paths-only` to pipe into solve, e.g.

        mamut-routing list --problem-type CVRP --paths-only \\
            | xargs -r mamut-routing solve --time-limit-s 30
    """
    state: CLIState = ctx.obj
    candidate_paths = _resolve_candidate_paths(state, list(instance_paths) if instance_paths else None)
    selected = _filter_loaded_instances(
        candidate_paths,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_variant,
        instance_id=instance_id,
    )

    if not selected:
        typer.echo(
            "No instances matched. Adjust filter flags or point --benchmarks-dir at a "
            "tree containing *.vrp.json files.",
            err=True,
        )
        raise typer.Exit(code=2)

    if paths_only:
        for path, _ in selected:
            typer.echo(str(path))
        return

    descriptors = [(path, _instance_descriptor(instance)) for path, instance in selected]

    typer.echo(f"Discovered {len(selected)} instance(s) under {state.benchmarks_dir}")
    header = (
        f"{'INSTANCE_ID':<32}  {'PROBLEM':<6}  {'BENCHMARK':<12}  "
        f"{'METRIC':<10}  {'PLACE':<14}  {'n':>5}  PATH"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for path, descriptor in descriptors:
        typer.echo(
            f"{(descriptor['instance_id'] or '-'):<32}  "
            f"{(descriptor['problem_type'] or '-'):<6}  "
            f"{(descriptor['benchmark_name'] or '-'):<12}  "
            f"{(descriptor['metric_variant'] or '-'):<10}  "
            f"{(descriptor['place_slug'] or '-'):<14}  "
            f"{(descriptor['num_customers'] if descriptor['num_customers'] is not None else '-'):>5}  "
            f"{path}"
        )

    if not summary:
        return

    counts_problem: dict[str, int] = {}
    counts_benchmark: dict[str, int] = {}
    counts_metric: dict[str, int] = {}
    counts_size: dict[int, int] = {}
    for _, descriptor in descriptors:
        problem_key = str(descriptor["problem_type"] or "-")
        benchmark_key = str(descriptor["benchmark_name"] or "-")
        metric_key = str(descriptor["metric_variant"] or "-")
        counts_problem[problem_key] = counts_problem.get(problem_key, 0) + 1
        counts_benchmark[benchmark_key] = counts_benchmark.get(benchmark_key, 0) + 1
        counts_metric[metric_key] = counts_metric.get(metric_key, 0) + 1
        size = descriptor["num_customers"]
        if isinstance(size, int):
            counts_size[size] = counts_size.get(size, 0) + 1

    def _fmt(counts: dict) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))

    typer.echo("")
    typer.echo("Summary:")
    typer.echo(f"  Total          : {len(selected)}")
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
        typer.Option("--instance-id", help="Filter by exact instance id."),
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
    ] = 30,
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

    candidate_paths = _resolve_candidate_paths(state, list(instance_paths) if instance_paths else None)

    selected = _filter_loaded_instances(
        candidate_paths,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
        metric_variant=metric_variant,
        instance_id=instance_id,
    )
    if not selected:
        typer.echo(
            "No instances selected. Pass positional paths, adjust filter flags, or "
            "point --benchmarks-dir at a tree containing *.vrp.json files.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(f"Solving {len(selected)} instance(s)  time_limit={time_limit_s}s  seed={seed}  save_bks={save_bks}")
    header = f"{'INSTANCE_ID':<32}  {'STATUS':<10}  {'COST':>10}  {'ROUTES':>6}  {'WALL_S':>7}  {'BKS':<14}"
    typer.echo(header)
    typer.echo("-" * len(header))

    any_failure = False
    for path, instance in selected:
        is_cvrp = isinstance(instance, BenchmarkInstanceCVRP)
        run_objective = ObjectiveFunction.MONO_COST if is_cvrp else objective
        if save_bks:
            method_result, bks_update = solve_and_update_bks(
                instance,
                instance_path=path,
                time_limit_s=time_limit_s,
                seed=seed,
                objective_function=run_objective,
                display=display,
            )
        else:
            method_result = solve_instance(
                instance,
                time_limit_s=time_limit_s,
                seed=seed,
                objective_function=run_objective,
                display=display,
            )
            bks_update = None

        if method_result.solver_is_feasible:
            status = "feasible"
            cost = method_result.solver_cost
        else:
            status = "infeasible"
            cost = None
            any_failure = True

        bks_label = bks_update.action if bks_update is not None else ("-" if save_bks else "skipped")
        identifier = (
            getattr(instance, "instance_id", None)
            or getattr(instance, "instance_name", None)
            or path.stem
        )
        cost_str = f"{cost:>10}" if cost is not None else f"{'-':>10}"
        typer.echo(
            f"{identifier:<32}  {status:<10}  {cost_str}  {method_result.route_count:>6}  "
            f"{method_result.wall_time:>7.2f}  {bks_label:<14}"
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

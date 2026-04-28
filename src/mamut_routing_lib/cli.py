from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from mamut_routing_lib.artifacts import (
    DEFAULT_BENCHMARKS_ROOT_ENV,
    DEFAULT_MAMUT_ROUTING_ROOT_ENV,
)
from mamut_routing_lib.enums import BenchmarkName, ProblemType
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
    help="CLI for the MAMUT-routing benchmark library: list, fetch, and verify "
    "remote benchmark archives published as GitHub release assets.",
    no_args_is_help=True,
    add_completion=False,
)


@dataclass
class CLIState:
    repo: str
    token: str | None
    tag: str | None
    output_dir: Path

    def make_client(self) -> GitHubReleaseClient:
        source = GitHubReleaseSource(repo_full_name=self.repo, token=self.token)
        return GitHubReleaseClient(source=source)


def _resolve_default_output_dir() -> Path:
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
            help=f"GitHub token. Defaults to ${DEFAULT_GITHUB_TOKEN_ENV}.",
        ),
    ] = None,
    tag: Annotated[
        Optional[str],
        typer.Option("--tag", help="Release tag (e.g. v0.0.1). Defaults to the latest release."),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help=f"Local directory for downloads/verification. Defaults to ${DEFAULT_BENCHMARKS_ROOT_ENV} "
            f"or ${DEFAULT_MAMUT_ROUTING_ROOT_ENV}/benchmarks or ./benchmarks.",
        ),
    ] = None,
) -> None:
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
    resolved_output_dir = (output_dir.expanduser().resolve() if output_dir is not None else _resolve_default_output_dir())
    ctx.obj = CLIState(
        repo=resolved_repo,
        token=resolved_token,
        tag=tag,
        output_dir=resolved_output_dir,
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
                f"Run `mamut-routing list` to see available assets."
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


@app.command("list")
def list_assets(
    ctx: typer.Context,
    scope: Annotated[Optional[ReleaseArchiveScope], typer.Option("--scope", case_sensitive=False)] = None,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
) -> None:
    """List benchmark archives available in the remote release manifest."""
    state: CLIState = ctx.obj
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


@app.command("fetch")
def fetch_assets(
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
    state: CLIState = ctx.obj
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

    state.output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {len(selected)} asset(s) into {state.output_dir}")

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
                state.output_dir,
                extract=extract,
                progress_callback=_on_progress,
            )
        typer.echo(f"  -> {destination}")


@app.command("verify")
def verify_local(
    ctx: typer.Context,
    scope: Annotated[Optional[ReleaseArchiveScope], typer.Option("--scope", case_sensitive=False)] = None,
    problem_type: Annotated[Optional[ProblemType], typer.Option("--problem-type", case_sensitive=False)] = None,
    benchmark_name: Annotated[Optional[BenchmarkName], typer.Option("--benchmark-name", case_sensitive=False)] = None,
) -> None:
    """Verify sha256 of locally downloaded archives against the remote manifest."""
    state: CLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    assets = manifest.select_assets(
        scope=scope,
        problem_type=problem_type,
        benchmark_name=benchmark_name,
    )

    typer.echo(f"Verifying {len(assets)} asset(s) under {state.output_dir}")
    has_failure = False
    for asset in assets:
        local_path = state.output_dir / asset.filename
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


@app.command("manifest")
def show_manifest(ctx: typer.Context) -> None:
    """Print the parsed release manifest as JSON."""
    state: CLIState = ctx.obj
    client = state.make_client()
    manifest = client.fetch_manifest(tag=state.tag)
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()

# MAMUT-routing-lib

Modern Python library for CVRP and VRPTW benchmark models, validation, BKS management, and snapshot retrieval.

## MAMUT project context

This repository is part of the [MAMUT project](https://github.com/ANR-MAMUT) ([ANR-22-CE22-0016](https://anr.fr/Project-ANR-22-CE22-0016)), an academic research project aiming to advance the state of the art in combinatorial optimization for logistics and transportation problems. 

## Scope

`mamut_routing_lib` is a standalone Python contract/runtime layer to work with the routing benchmarks curated in the [MAMUT-routing](https://github.com/ANR-MAMUT/MAMUT-routing) repository. It is inspired by projects like [VRPLIB](https://github.com/PyVRP/VRPLIB) and is intended as a general-purpose library for working with CVRP and VRPTW benchmark instances, both historical and newly generated as well as their associated BKS and metadata.

It provides:

- historical VRPTW benchmark models
- generated CVRP and VRPTW benchmark models
- local benchmark discovery and JSON I/O
- solution checking
- BKS creation and replacement logic
- optional remote snapshot archive retrieval

This repository does not own site generation, publication-history generation, migration pipelines, or solver integrations. It is a pure contract and runtime library for benchmark data management intended to be used by researchers and practitioners alike, both inside and outside the MAMUT project.

## Installation

```bash
pip install mamut-routing-lib
```

or, using the modern [`uv`](https://github.com/astral-sh/uv) Python package manager:

```bash
uv add mamut-routing-lib
```

## Local Loading

```python
from pathlib import Path

from mamut_routing_lib import discover_benchmark_instances

items = discover_benchmark_instances(
    benchmarks_root=Path("/path/to/benchmarks"),
)
```

## Remote Snapshot Retrieval

The optional remote module consumes release manifests and release assets published by a benchmark repository such as `MAMUT-routing`.

Default environment variables:

- `MAMUT_ROUTING_RELEASE_REPO`
- `MAMUT_ROUTING_GITHUB_TOKEN`
- `MAMUT_ROUTING_ROOT`
- `MAMUT_ROUTING_BENCHMARKS_ROOT`

## Command-line interface

A `mamut-routing` CLI is available with the optional `cli` extra:

```bash
pip install "mamut-routing-lib[cli]"
# or with uv
uv add "mamut-routing-lib[cli]"
```

It exposes local benchmark commands by default, plus a `remote` command group
backed by the remote retrieval module:

```bash
# List archives available in the latest release of the configured repo
mamut-routing remote --repo ANR-MAMUT/MAMUT-routing-dummy --tag v0.0.1 list

# Filter by scope/problem-type/benchmark-name
mamut-routing remote --tag v0.0.1 list --problem-type CVRP --scope problem_family

# Download (and extract) one or more archives into --benchmarks-dir
mamut-routing --benchmarks-dir ./benchmarks remote --tag v0.0.1 \
    fetch CVRP-Mamut2026-snapshot-2026-04-24-621056e.zip

# Or fetch by filter:
mamut-routing remote --tag v0.0.1 fetch --problem-type CVRP --benchmark-name Mamut2026

# Verify local zip checksums against the remote manifest
mamut-routing --benchmarks-dir ./benchmarks remote --tag v0.0.1 verify

# Print the parsed manifest as JSON
mamut-routing remote --tag v0.0.1 manifest | jq .snapshot_id
```

The `--benchmarks-dir` flag is also read from `MAMUT_ROUTING_BENCHMARKS_ROOT`
or `MAMUT_ROUTING_ROOT`. Remote flags `--repo`, `--token`, and `--tag` are read
from `MAMUT_ROUTING_RELEASE_REPO` and `MAMUT_ROUTING_GITHUB_TOKEN` where
applicable.

## Solving with PyVRP

An optional `[pyvrp]` extra wraps PyVRP's HGS metaheuristic so users can solve
CVRP and VRPTW instances directly from the library.

```bash
# Python API only
pip install "mamut-routing-lib[pyvrp]"

# Both the CLI (mamut-routing solve) and the API
pip install "mamut-routing-lib[cli,pyvrp]"
```

Python:

```python
from mamut_routing_lib import load_benchmark_instance, ObjectiveFunction
from mamut_routing_lib.solvers.pyvrp import solve_instance, solve_and_update_bks

instance = load_benchmark_instance("path/to/instance.vrp.json")
result = solve_instance(instance, time_limit_s=30, seed=42)
print(result.solver_is_feasible, result.solver_cost, result.route_count)

# Or solve-and-write-BKS in one call
result, update = solve_and_update_bks(
    instance,
    instance_path="path/to/instance.vrp.json",
    time_limit_s=30,
    seed=42,
    objective_function=ObjectiveFunction.HIERARCHICAL_VEHICLE_COST,
)
print(update.action if update else "infeasible")
```

CLI (requires `[cli,pyvrp]`):

```bash
# Inspect what's locally available before solving
mamut-routing --benchmarks-dir ./benchmarks list \
    --problem-type CVRP --benchmark-name Mamut2026

# Pipe the matching paths into solve
mamut-routing --benchmarks-dir ./benchmarks list \
    --problem-type CVRP --paths-only \
    | xargs -r mamut-routing solve --time-limit-s 30

# Solve specific instances
mamut-routing solve path/to/inst1.vrp.json path/to/inst2.vrp.json \
    --time-limit-s 30 --seed 42

# Or discover under --benchmarks-dir and filter
mamut-routing --benchmarks-dir ./benchmarks solve \
    --problem-type VRPTW --benchmark-name Mamut2026 \
    --objective hierarchical_vehicle_cost \
    --time-limit-s 60
```

## Development

```bash
# Install editable with CLI extras and test deps
uv pip install -e ".[cli]"
uv pip install pytest

# Hermetic offline test suite (no network)
pytest -v tests/

# Opt-in real-network smoke test (downloads ~1.8 MB from a public release)
MAMUT_ROUTING_TEST_NETWORK=1 pytest -v tests/test_remote_network.py
```

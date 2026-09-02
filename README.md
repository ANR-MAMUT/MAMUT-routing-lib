# MAMUT-routing-lib

Modern Python library for CVRP, VRPTW and time-dependent (TDVRPTW/TDVRP) benchmark models, validation, BKS management, and snapshot retrieval.

[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/ANR-MAMUT/MAMUT-routing-lib/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/ANR-MAMUT/MAMUT-routing-lib)

## MAMUT project context

This repository is part of the [MAMUT project](https://github.com/ANR-MAMUT) ([ANR-22-CE22-0016](https://anr.fr/Project-ANR-22-CE22-0016)), an academic research project aiming to advance the state of the art in combinatorial optimization for logistics and transportation problems. 

## Scope

`mamut_routing_lib` is a standalone Python contract/runtime layer to work with the routing benchmarks curated in the [MAMUT-routing](https://github.com/ANR-MAMUT/MAMUT-routing) repository. It is inspired by projects like [VRPLIB](https://github.com/PyVRP/VRPLIB) and is intended as a general-purpose library for working with CVRP and VRPTW benchmark instances, both historical and newly generated as well as their associated BKS and metadata.

It provides:

- historical VRPTW benchmark models
- generated CVRP and VRPTW benchmark models
- time-dependent (TDVRPTW/TDVRP) benchmark models with arrival-time-function sidecars and an exact, epsilon-free checker for the Duration and FleetCostDuration objectives (`mamut_routing_lib.td`; FleetCostDuration = duration + `fleet_fixed_cost` per used vehicle, the Blauth2024 contract)
- local benchmark discovery and JSON I/O
- solution checking
- BKS creation and replacement logic
- optional remote snapshot archive retrieval
- export of static instances to the classic CVRPLIB `.vrp` (and Solomon `.txt`) formats for solvers that do not read `.vrp.json`

The time-dependent layer is the pricing authority of [KAYROS](https://github.com/0nyr/kayros), the MAMUT time-dependent VRP solver: KAYROS finds routes, this library's checker defines and validates their cost.

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
mamut-routing remote --repo ANR-MAMUT/MAMUT-routing list

# Filter by problem-type/benchmark-name
mamut-routing remote list --problem-type CVRP --benchmark-name Poryos2026

# Download (and extract) one or more archives into --benchmarks-dir
mamut-routing --benchmarks-dir ./benchmarks remote \
    fetch CVRP-Poryos2026-snapshot-2026-05-22-28f9199.zip

# Or fetch by filter:
mamut-routing remote fetch --problem-type CVRP --benchmark-name Poryos2026

# Verify local zip checksums against the remote manifest
mamut-routing --benchmarks-dir ./benchmarks remote verify

# Print the parsed manifest as JSON
mamut-routing remote manifest | jq .snapshot_id
```

Release archives are published at the problem-family level, for example
`CVRP-Poryos2026` or `VRPTW-Sintef2008`. Extracted archives are placed in a
directory named after the archive stem, containing the archived `benchmarks/...`
tree.

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
    --problem-type CVRP --benchmark-name Poryos2026

# Include source file paths in the table when needed
mamut-routing --benchmarks-dir ./benchmarks list --show-path

# Pipe the matching paths into solve
mamut-routing --benchmarks-dir ./benchmarks list \
    --problem-type CVRP --paths-only \
    | xargs -r mamut-routing solve --time-limit-s 30

# Solve specific instances
mamut-routing solve path/to/inst1.vrp.json path/to/inst2.vrp.json \
    --time-limit-s 30 --seed 42

# Or discover under --benchmarks-dir and filter
mamut-routing --benchmarks-dir ./benchmarks solve \
    --problem-type VRPTW --benchmark-name Poryos2026 \
    --objective hierarchical_vehicle_cost \
    --time-limit-s 60
```

## Exporting to CVRPLIB `.vrp` (classic solvers)

Solvers that do not read the `.vrp.json` contract can consume the classic
TSPLIB-derived CVRPLIB format instead. `mamut_routing_lib.cvrplib` converts any
static instance (CVRP or VRPTW, embedded matrix or slim collection instance)
into one `.vrp` file per instance, with the same selection model as `list` and
`solve`:

```bash
# One instance: writes <name>.vrp next to the source .vrp.json
mamut-routing export vrp path/to/inst.vrp.json

# A whole family under --benchmarks-dir, mirrored into --output-dir
mamut-routing --benchmarks-dir ./benchmarks export vrp \
    --problem-type CVRP --benchmark-name Mamut2026 --output-dir ./vrp-out --jobs 4

# Coordinates-only TSPLIB file (euclidean-metric instances only, see below)
mamut-routing export vrp inst.vrp.json --edge-weight-type EUC_2D

# Solomon / Gehring-Homberger .txt (VRPTW, euclidean metric only)
mamut-routing export vrp R1_4_6.vrp.json --format solomon
```

```python
from mamut_routing_lib import load_benchmark_instance
from mamut_routing_lib.cvrplib import VrpExportOptions, export_instance_file, instance_to_vrp_text

text = instance_to_vrp_text(load_benchmark_instance(path), instance_path=path)
result = export_instance_file(path, options=VrpExportOptions(edge_weight_type="EXPLICIT"))
```

The default output is `EDGE_WEIGHT_TYPE : EXPLICIT` with a `FULL_MATRIX`
section, so the solver sees exactly the published costs (3-decimal floats for
the Poryos2026/Mamut2026 collections, whose matrix is hydrated from the
sha-pinned distances sidecar; integers or full-precision floats for the
historical families). It is byte-identical to the `.vrp` files committed next to
the collection CVRP instances. VRPTW instances get `TYPE : CVRPTW` with
`TIME_WINDOW_SECTION` / `SERVICE_TIME_SECTION` (the dialect read by
[VRPLIB](https://github.com/PyVRP/VRPLIB) and PyVRP) and a `VEHICLES` header
when the fleet is fixed. Node ids are 1-based, the depot is node 1.

Caveats:

- `--edge-weight-type EUC_2D` drops the matrix; TSPLIB readers then compute
  `nint(euclidean)` distances, which are **not** the published 3-decimal costs,
  so BKS values do not transfer. It is refused for `shortest`/`fastest`
  instances, whose road metrics need the explicit matrix.
- Time-dependent instances (TDVRP/TDVRPTW) have no static matrix and are
  refused: explicit paths are an error, scanned ones are skipped with a warning.
- Existing outputs are reported as `exists` and left alone unless `--force`.

## Development

```bash
# Install editable with CLI extras and test deps
uv pip install -e ".[cli]"
uv pip install pytest

# Hermetic offline test suite (no network)
pytest -v tests/

# Opt-in real-network smoke test (downloads ~1.6 MB from the public MAMUT-routing release)
MAMUT_ROUTING_TEST_NETWORK=1 pytest -v tests/test_remote_network.py
```

## Archival and reproducibility

`MAMUT-routing-lib` is archived by [Software Heritage](https://www.softwareheritage.org/); the badge above tracks the archive status of the GitHub origin:

- [Software Heritage origin](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/ANR-MAMUT/MAMUT-routing-lib)
- [Software Heritage archival visits](https://archive.softwareheritage.org/browse/origin/visits/?origin_url=https://github.com/ANR-MAMUT/MAMUT-routing-lib)

For academic referencing, use Software Heritage identifiers (SWHIDs) to cite the exact archived revision or release tag rather than the moving repository origin — e.g. the precise version of the validation rules, the Duration checker, or the BKS replacement logic used in an experiment.

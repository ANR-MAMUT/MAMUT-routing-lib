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

# Filter by problem-type/benchmark-name (a problem-type filter also keeps the
# family-first collections, which ship every problem type of their family)
mamut-routing remote list --problem-type CVRP

# Download and extract archives into --benchmarks-dir
mamut-routing --benchmarks-dir ./benchmarks remote \
    fetch Poryos2026-snapshot-2026-09-23-70ca946.zip

# Or fetch by filter:
mamut-routing --benchmarks-dir ./benchmarks remote fetch --benchmark-name Sintef2008

# Verify local zips (or extracted trees) against the remote manifest
mamut-routing --benchmarks-dir ./benchmarks remote verify

# Print the parsed manifest as JSON
mamut-routing remote manifest | jq .snapshot_id
```

A release ships one archive per classic (problem type, family), e.g.
`VRPTW-Sintef2008-snapshot-<id>.zip`, and one per family-first collection,
e.g. `Poryos2026-snapshot-<id>.zip`. `fetch` keeps each zip at
`<benchmarks-dir>/<filename>` and extracts it into the canonical tree
(`<benchmarks-dir>/VRPTW/Sintef2008`, `<benchmarks-dir>/Poryos2026`), the same
layout as a repository checkout, so `list` and
`discover_benchmark_instances(<benchmarks-dir>)` work on a fetched tree. An
extracted directory carries a `.mamut-release.json` stamp and is replaced by
the next fetch of that family (including files written into it since, such as
BKS saved by `solve`); an existing directory without a stamp (a git checkout,
local data) is only replaced with `fetch --force`. Trees extracted by lib <
0.12 (`<benchmarks-dir>/<archive stem>/benchmarks/...`) are not discoverable:
re-fetch, then delete them.

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

# Or discover under --benchmarks-dir and filter (Sintef2008 BKS use the
# hierarchical objective)
mamut-routing --benchmarks-dir ./benchmarks solve \
    --problem-type VRPTW --benchmark-name Sintef2008 \
    --objective hierarchicalvehiclecost \
    --time-limit-s 60
```

`solve` covers CVRP and VRPTW: scanned time-dependent instances are skipped
with a warning (an explicit TD path is an error), and so are collection
instances whose distances sidecar is a sha256 pin not present in the tree. A
failing instance becomes an `error` row instead of stopping the batch; the
table and a summary line are always printed. Exit status: 0 when every solved
instance is feasible, 1 when one is infeasible or errors, 2 on usage errors.

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

# Coordinates-only TSPLIB file (instances whose costs the coordinates define, see below)
mamut-routing export vrp inst.vrp.json --edge-weight-type EUC_2D

# Solomon / Gehring-Homberger .txt (VRPTW, same condition)
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

- `--edge-weight-type EUC_2D` and `--format solomon` keep only the
  coordinates. They are offered only when every published arc cost is a
  rounding of the Euclidean distance of the stored coordinates
  (`coordinates_define_arc_costs`): full-precision (Sintef2008) and 3-decimal
  (collections) costs qualify; `shortest`/`fastest` road metrics and Dimacs2021
  (original coordinates, `floor(10 * d)` costs and x10 times) do not, and
  `EXPLICIT` is the faithful form for them. TSPLIB readers compute
  `nint(euclidean)` distances, which can still differ from the published costs
  by that rounding, so BKS values do not transfer exactly.
- Time-dependent instances (TDVRP/TDVRPTW) have no static matrix and are
  refused: explicit paths are an error, scanned ones are skipped with a warning.
- Existing outputs are reported as `exists` and left alone unless `--force`.

## Time-dependent checker contract and re-pricing

The TD checker (`mamut_routing_lib.td.check_td_solution`) defines every TD
cost; its route fold is versioned as `TD_CHECKER_CONTRACT`. Since 0.12.0 it is
`td-fold/2`: vertex ready-time maps (`max(t, earliest) + service`) and the
depot due-date cut are applied exactly to the accumulator's breakpoints, and
arcs compose with the slope-one rule, so integer data is folded without any
rounding (see the module docstring and `docs/benchmarks/formats/time-dependent.md`
of the benchmark repository). After a contract change every stored TD BKS is
re-priced once, routes untouched:

```bash
# Dry run with a JSON report, then write
mamut-routing bks reprice-td benchmarks/TDVRPTW benchmarks/TDVRP --dry-run --jobs 8 --report reprice.json
mamut-routing bks reprice-td benchmarks/TDVRPTW benchmarks/TDVRP --jobs 8
```

A file is rewritten only when a checker output changed (`cost`,
`validated_cost`, `route_durations`, `route_departure_times`); a moved cost is
recorded in `metadata.repriced`, and an optimality stamp gets its
`proven_optimum` updated with a note.

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

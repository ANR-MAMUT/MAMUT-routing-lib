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

Kayros finds TD routes and reports results under its own checker contract. This library can independently evaluate those routes under the selected benchmark profile. [Kayros](https://github.com/0nyr/kayros)

This repository does not own site generation, publication-history generation, migration pipelines, or solver integrations. It is a pure contract and runtime library for benchmark data management intended to be used by researchers and practitioners alike, both inside and outside the MAMUT project.

## Installation

```bash
pip install mamut-routing-lib
```

or, using the modern [`uv`](https://github.com/astral-sh/uv) Python package manager:

```bash
uv add mamut-routing-lib
```

## Exact route evaluation on v0.13

For static instances, `binary64-exact-v1` reads each matrix entry as a binary64 value and sums its exact rational value over all traversed arcs. For time-dependent instances, `td-binary64-exact-v1` reads the arrival-time functions the same way and computes the minimum feasible duration and the earliest departure attaining it for every route. The instance objective determines how those route values are compared and whether a fixed fleet charge applies.

| Profile | Default C++ backend | Selectable Python reference |
| --- | --- | --- |
| `binary64-exact-v1` | `cpp-dyadic-prepared` (integer arithmetic with promotion) | `python-fraction` |
| `td-binary64-exact-v1` | `cpp-gmp-compose` (rational function composition) | `python-fraction-partition` |

```python
from mamut_routing_lib import evaluate_exact, prepare_exact

# instance_path points to a .vrp.json file. Its referenced sidecars resolve relative to it.
prepared = prepare_exact(instance_path)
result = prepared.evaluate(routes)  # reuse prepared for other submissions
assert result.cost == evaluate_exact(instance_path, routes, backend="python").cost
print(result.cost)          # fractions.Fraction
print(result.display_cost)  # nearest binary64 value, or None on overflow
for route, witness in zip(result.routes, result.route_evaluations):
    print(route, witness.duration, witness.departure_time)
```

The CLI commands `check` and `bks submit` accept `--backend native` (default) or `--backend python`. Source builds of the default wheel require a C++20 toolchain, Boost headers, and GMP C/C++ development headers and libraries. To build a reference-only package, run `MAMUT_REFERENCE_ONLY=1 python -m pip install .` in a checkout. Select `backend="python"` explicitly with that installation. Requesting the native backend without its extension raises an installation error.

BKS submission evaluates the candidate and the incumbent under the same profile and compares their rational objective values. Exact ties keep the incumbent. New records carry the reduced cost, canonical route digest and actual backend identity under `metadata.exact_evaluation`. TD records also carry a reduced duration and earliest departure for each route in stored order. The top-level `cost` is a display value. On display overflow it is `null` and the exact rational remains available. Family revision and verified manifest hashes are required to issue `metadata.standard`.

A Kayros optimality claim names the contract under which the solver made it. Repricing the same routes under the v0.13 TD profile preserves the historical target, usually `td-fold/1`. If a better submission displaces an annotated incumbent, the store keeps the original BKS bytes under `.bks-history/<sha256>.json`. The new submission can carry its own solver claim.

## Migrating Python code to v0.13

| Previous API | v0.13 API | Meaning |
| --- | --- | --- |
| `BenchmarkInstance` | `BenchmarkInstanceVRPTW` | The static VRPTW model with an embedded matrix. The JSON Schema title uses the new name. The instance JSON fields keep their spelling. |
| `save_bks_if_improved(..., bks=record)` | `save_bks_if_improved(..., candidate_bks=record)` | Recompute the candidate and incumbent, then write a strict improvement. Positional calls still work. |
| `save_bks(record, path)` | `write_bks_file(record, path)` | Serialize the model to the given path, overwriting an existing file. Use `submit_bks` for checked submission. |
| `exact_cost_value(number)` | `shortest_decimal_cost_value(number)` | Recover the rational value of the shortest decimal spelling of the parsed number. |
| `compute_exact_solution_cost(instance, routes)` | `compute_shortest_decimal_solution_cost(instance, routes)` | Sum the static arc costs using that historical decimal interpretation. For the v0.13 binary64 profile use `evaluate_exact(instance, routes).cost`. |

During v0.13, importing `BenchmarkInstance` from the package or its `models` module returns the same class with a `DeprecationWarning`. The other renamed APIs use the new spellings. Type annotations can distinguish `LoadedStaticInstance` (CVRP or VRPTW with a matrix), `AnyStaticBenchmarkInstance` (also includes static collection descriptors), and `AnyBenchmarkInstance` (also includes TDVRP and TDVRPTW). All 3 unions are available at the package root.

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

A release ships one archive per family, e.g. `Sintef2008-snapshot-<id>.zip`
and `Poryos2026-snapshot-<id>.zip`, each holding every problem type of the
family and its shared sidecars. `fetch` keeps each zip at
`<benchmarks-dir>/<filename>` and extracts it into the canonical tree
(`<benchmarks-dir>/Sintef2008`, `<benchmarks-dir>/Poryos2026`), the same
layout as a repository checkout, so `list` and
`discover_benchmark_instances(<benchmarks-dir>)` work on a fetched tree. An
extracted directory carries a `.mamut-release.json` stamp and is replaced by
the next fetch of that family (including files written into it since, such as
BKS saved by `solve`); an existing directory without a stamp (a git checkout,
local data) is only replaced with `fetch --force`. Trees extracted by lib <
0.12 (`<benchmarks-dir>/<archive stem>/benchmarks/...`) are not discoverable:
re-fetch, then delete them. Snapshots published for lib <= 0.12 hold the
former layout (`benchmarks/<problem>/<family>/...`), which this version does
not read: `fetch` refuses to extract their archives.

The `--benchmarks-dir` flag is also read from `MAMUT_ROUTING_BENCHMARKS_ROOT`
or `MAMUT_ROUTING_ROOT`. Remote flags `--repo`, `--token`, and `--tag` are read
from `MAMUT_ROUTING_RELEASE_REPO` and `MAMUT_ROUTING_GITHUB_TOKEN` where
applicable.

## Solving: PyVRP, KAYROS and external solvers

Solvers are looked up by name in one registry and run behind one protocol;
every solution is priced by the reference checker before it is reported or
stored, whatever the solver claims.

| Solver | Solves | Needs |
|---|---|---|
| `pyvrp` (default) | CVRP (MonoCost), VRPTW (MonoCost, HierarchicalVehicleCost) | the `[pyvrp]` extra |
| `kayros` | TDVRP, TDVRPTW (Duration) | `pip install kayros`, in a release whose `mamut-routing-lib` requirement admits this version (`mamut-routing solvers list` warns otherwise) |
| `hgs-cvrp` | CVRP (MonoCost), any metric (the published matrix, asymmetric included) | a build with the lib's full-matrix reader patch (`mamut-routing solvers list` prints its path) |
| `filo2`, `ails2` | CVRP (MonoCost), euclidean-metric variants (they compute distances from coordinates) | the executable (AILS-II: a jar with the `-seed`/`-out` patch) |
| `lkh3` | CVRP and VRPTW (MonoCost), any metric (explicit integer matrices, asymmetric included) | the `LKH` executable |

```bash
pip install "mamut-routing-lib[cli,pyvrp]"
mamut-routing solvers list          # what is available here, and how to enable the rest
```

Python:

```python
from mamut_routing_lib import run_solver, submit_bks

solved = run_solver("pyvrp", "path/to/instance.vrp.json", time_limit_s=30, seed=42)
print(solved.valid, solved.cost, solved.num_routes)        # the checker's verdict and cost
print(solved.result.status, solved.result.solver_cost)     # the solver's own report

if solved.valid:                                           # store only if strictly better
    update = submit_bks(solved.instance_path, solved.result.routes,
                        authors="Your Name (handle)", metadata=solved.bks_metadata(),
                        checkable=solved.checkable)        # the instance the check already loaded
    print(update.action)                                   # created / replaced / kept_existing
```

One solve reads a large distance matrix (or a TD instance's travel-time
functions) once: the solver's input, the check and, through `checkable=`, the
store share it. `load_checkable(instance, path)` gives the same object to
callers that check, preview and submit a solution found elsewhere. It can be
large, so drop a `CheckedSolve` once it is stored rather than collecting them.

`run_solver` takes any registered name (or a `Solver` object), an optional
`objective_function` (default MonoCost for static instances, Duration for TD),
a `SolveMonitor` for live progress and cancellation, and `initial_routes` for
warm starts (PyVRP). The PyVRP functions of `mamut_routing_lib.solvers.pyvrp`
(`solve_instance`, `solve_cvrp`, `solve_vrptw`, `solve_and_update_bks`, which
now requires `authors=`) remain available.

CLI:

```bash
# Solve; nothing is written unless asked
mamut-routing solve path/to/inst.vrp.json --time-limit-s 30 --seed 42

# Keep the solutions (routes, checker verdict, provenance), to check or submit later
mamut-routing solve path/to/inst.vrp.json --output-dir runs/
mamut-routing bks submit path/to/inst.vrp.json runs/<instance-id>.pyvrp-seed42.json --authors "Your Name (handle)"

# Store improvements as BKS (credited to --authors, or $MAMUT_ROUTING_BKS_AUTHORS)
mamut-routing --benchmarks-dir ./benchmarks solve \
    --problem-type VRPTW --benchmark-name Sintef2008 \
    --objective hierarchicalvehiclecost --time-limit-s 60 \
    --save-bks --authors "Your Name (handle)"

# Another solver, from the solvers config
mamut-routing solve path/to/inst.vrp.json --solver lkh3 --time-limit-s 60

# Pipe the matching paths into solve
mamut-routing --benchmarks-dir ./benchmarks list --problem-type CVRP --paths-only \
    | xargs -r mamut-routing solve --time-limit-s 30
```

The COST column is the checker's cost and GAP its distance to the BKS stored
before the solve (`no BKS` when there is none; under HierarchicalVehicleCost a
different route count shows as `+1 veh`). The BKS column says what the store did
(`not saved` without `--save-bks`). `--time-limit` and `--update-bks` are
accepted as aliases, the names `mamut-tools solve` used. Instances a solver cannot take are
skipped in scans (time-dependent instances for static solvers, road metrics for
coordinate solvers, ...) and are error rows for explicit paths; so are
collection instances whose distances sidecar is a sha256 pin missing from the
tree. A failing instance becomes an `error` row instead of stopping the batch.
Ctrl-C stops every running solve: external solvers are killed with their
process group, and so is KAYROS, which runs in a child Python process because
its search has no stop hook between two incumbents (a stopped solve keeps its
last incumbent). Exit status: 0 when every solved instance is feasible and
checker-valid, 1 when one is not, 2 on usage errors.

### External solvers

External executables are declared in a TOML file: `--solvers-config PATH`,
else `$MAMUT_ROUTING_SOLVERS_CONFIG` (empty disables it), else
`$XDG_CONFIG_HOME/mamut-routing/solvers.toml` (`~/.config/...`). It is never
read from the working directory: it names programs that will be run.
`mamut-routing solvers init` writes a commented starter there (one table per
built-in solver, with its build notes), and `mamut-routing solvers list` shows
what is available and how to enable the rest.

```toml
schema_version = 1

[solvers.filo2]
executable = "/opt/filo2/build/filo2"   # built with -DENABLE_TIMELIMIT=ON
version = "43653e2"

[solvers.hgs-cvrp]
executable = "/opt/HGS-CVRP/build/hgs"   # with patches/hgs-cvrp-explicit-matrix.patch applied

[solvers.ails2]
jar = "/opt/ails2/AILSII.jar"           # the -seed/-out patched build
java = "/opt/jdk-21/bin/java"

[solvers.lkh3]
executable = "/opt/LKH-3/LKH"

[solvers.my-solver]                     # any solver reading/writing CVRPLIB files
executable = "/opt/mine/solve"
argv = ["{executable}", "{problem}", "{solution}", "{time_limit_s}", "{seed}"]
input_format = "cvrplib-explicit"       # cvrplib-x | cvrplib-explicit | lkh
output_format = "cvrplib-sol"           # cvrplib-sol | lkh-mtsp
problem_types = ["CVRP"]
# symmetric_only = false                # only if it really reads asymmetric FULL_MATRIX files
```

Relative paths are read from the config file's folder; a bare executable name is looked up on PATH. Unknown keys
and mistyped values (`fleet_limit = "false"`) are errors, with a suggestion for near-miss keys; an error in one
`[solvers.<name>]` table disables that solver only (`solvers list` shows why), and `extends` may name a table defined
anywhere in the file.

A solver declared in the config is `symmetric_only = true` unless it says otherwise, so the asymmetric
road-network variants (`fastest`, `shortest`) are refused for it rather than handed over as a `TYPE : CVRP` file a
symmetric solver would misread; `extends = "lkh3"` inherits LKH-3's asymmetric support.

Each run writes the solver's input into a scratch directory
(`--keep-artifacts DIR` keeps it), scaled to the integers the solver needs:
exactly (x1000 for 3-decimal matrices; the solver's cost then unscales exactly
and is cross-checked) when the solver's limits allow, otherwise approximately --
conservatively for time windows, so a schedule the solver finds feasible is
feasible on the published data. HGS-CVRP computes in doubles, so it reads the
published matrix as is (divided by 10, exactly, where a distance exceeds its 1e5
limit), with the coordinates only for SWAP*'s polar sectors. Coordinate solvers
get coordinates in a unit that fits their limits (FILO2 millimetres). The
solver runs in its own process group with a hard timeout (budget + grace) and is
killed with its children on cancellation. A run that ends without a solution
file reports the last lines the solver printed (an unpatched build rejecting the
input, a missing library), since the scratch directory is gone by then.
Provenance (command, executable and input sha256, scale, version, machine) goes
into the run result and into a BKS stored from it.

With `input_format = "lkh"`, `extra_args` are literal lines appended to the
LKH-3 parameter file (`extra_args = ["RUNS = 10"]`), whatever the `argv` -- a
command wrapped in `taskset` or a launcher script keeps them. For every other
format they are extra command-line tokens.

### Solutions found elsewhere

```bash
mamut-routing check path/to/inst.vrp.json found.sol         # exit 0 valid, 1 invalid, 2 unreadable
mamut-routing bks submit path/to/inst.vrp.json found.sol \
    --authors "Your Name (handle)" --method "LKH-3" --dry-run
```

Both read CVRPLIB `.sol` files (`Route #k: ...` and `Cost ...`, exponents
accepted), JSON routes and LKH-3 MTSP solution files, and map 1-based,
0-based or depot-marked stop ids onto the instance (`--ids` forces one). The
objective defaults to the one the family publishes its BKS under. A file written
by `solve --output-dir` brings its run's provenance into `bks submit`; `--meta`
cannot set the fields the store writes itself (`authors`, `date`,
`validated_cost`, ...). `--dry-run` also re-checks the stored BKS
(`stored_invalid` when it fails the checker).

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

## Replaying historical records

The static compatibility helpers `check_solution` and `compute_shortest_decimal_solution_cost` retain the released arithmetic for historical callers. The latter interprets shortest decimal spellings. Use `evaluate_exact` for the v0.13 binary64 rational cost.

The v0.12 entry point `mamut_routing_lib.td.check_td_solution` uses `TD_CHECKER_CONTRACT` (`td-fold/2`). Its `bks reprice-td` command is retained for working with legacy records. To preview its effects on a historical checkout, run `mamut-routing bks reprice-td benchmarks --dry-run --jobs 8 --report reprice.json`. Use the exact entry points above for v0.13 submissions. The historical repricer refuses a record that carries exact evaluation or certification metadata.

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

## Academic citation

The authors, in citation order, are Adrien Pichon, Florian Rascoussier, Romain Billot, Lina Fahed, Alexandru-Liviu Olteanu, Marc Sevaux and Christine Solnon. The 2 lead authors are ordered alphabetically by surname, followed by the scientific supervisors in alphabetical order. Cite this project using [CITATION.cff](CITATION.cff), which includes all authors' ORCID identifiers.

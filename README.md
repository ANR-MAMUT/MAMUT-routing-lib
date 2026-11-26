# MAMUT-routing-lib

Modern Python library for CVRP, VRPTW and time-dependent (TDVRPTW/TDVRP) benchmark models, validation, BKS management, and snapshot retrieval.

[![PyPI](https://img.shields.io/pypi/v/mamut-routing-lib)](https://pypi.org/project/mamut-routing-lib/)
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

The library includes the solver registry, adapters and route checkers. The collection repository builds the website and release archives.

## Installation

Version 0.13.0 reads edition `r1` of the family collections. Install its CLI in a fresh Python environment. The published Linux x86-64 wheels target CPython 3.11 to 3.14. For a source build, install the native build prerequisites listed below.

```bash
python -m pip install 'mamut-routing-lib[cli]==0.13.0'
mamut-routing --version
```

To use the historical September 2026 snapshot, install `mamut-routing-lib[cli]==0.12.0` in a separate environment. The checker and archive reader work independently of the optional solver packages.

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

For a TD instance, `prepared.route_curve(route)` returns the breakpoints of the exact depot return time over all feasible departure times. Each knot has three `Fraction` fields. At `departure`, the vehicle returns at `arrival`. Between consecutive knots the return time interpolates linearly from the first knot's `right_arrival` to the second knot's `arrival`. A difference between `arrival` and `right_arrival` marks a jump. Subtract the departure from the return time to obtain the route duration. The earliest minimizing departure and minimum duration are also available through `prepared.route_timing(route)`.

The CLI commands `check` and `bks submit` accept `--backend native` (default) or `--backend python`. Source builds of the default wheel require a C++20 toolchain, Boost headers, and GMP C/C++ development headers and libraries. To build a reference-only package, run `MAMUT_REFERENCE_ONLY=1 python -m pip install .` in a checkout. Select `backend="python"` explicitly with that installation. Requesting the native backend without its extension raises an installation error.

A C++17 caller can use the same TD kernel by including `<mamut_routing_lib/td_exact.hpp>`. Configure CMake with `-DMAMUTRouting_DIR="$(python -m mamut_routing_lib.cmake)"`, call `find_package(MAMUTRouting CONFIG REQUIRED)` and link `MAMUTRouting::td_exact`. The target uses the GMP development headers and libraries from the consumer's build environment. In namespace `mamut_routing::td_exact`, construct an `Arc` from two binary64 arrays (departure and arrival) and their shared length. Keep the prepared arc for subsequent route queries. Both `route_timing` and `arrival_curve` take a sequence of pointers to these arcs, one customer window and service time per internal stop, the departure interval and an optional depot closing time. An infeasible route returns an empty optional or curve. The Python native backend prepares and retains the GMP representation of an arc on its first use within a prepared instance.

BKS submission evaluates the candidate and the incumbent under the same profile and compares their rational objective values. Exact ties keep the incumbent. For a published edition, the adjacent `<instance>.manifest.json` records the edition label, required input files, materialization method and digest of the complete binary64 matrix or ATF stream. A successful check can bind the resulting `metadata.standard` record to that exact manifest file by SHA-256. In a family awaiting its edition manifest, the store writes `metadata.exact_evaluation`. Both records include the reduced cost, canonical route digest and actual backend identity. TD records also include a reduced duration and earliest minimizing departure for each route in stored order. On display overflow the top-level `cost` is `null`.

Use `mamut-routing verify-manifest INSTANCE.vrp.json --sha256 DIGEST` to verify an edition against an independently supplied manifest digest. The Python entry point is `verify_manifest(instance_path, expected_sha256=digest)`. It resolves source files relative to the family root, verifies their hashes and recomputes the complete semantic digest. Reconstruct omitted sidecars with the family's pinned recipe when needed. A directory copied from a release archive can be verified without Git. Use `mamut-routing check INSTANCE.vrp.json SOLUTION --json` to recompute a certified result and verify its edition and exact numerical assertions. The CLI streams TD data. In Python, pass `routes=()` for a digest-only check or `routes=routes` to keep the submitted arcs for evaluation. Both verify every directed arc and knot. A separately regenerated ATF tree may be passed as `--atf-cache-root` or `atf_cache_root=Path(...)`; its directory layout follows the instance descriptor.

An in-memory `check_solution` or `td.check_td_solution` evaluates the supplied materialized instance. A static caller checking a certified BKS passes `instance_path=` to identify its edition. The TD loader already carries that path. Checking a certified record verifies the edition and checks the supplied operands against it. Repeated route evaluations on one fixed instance can reuse `prepare_exact`.

A solver claim of optimality is stored under `metadata.optimality`. Its `status` is `solver-reported-optimal`, its `target_profile` must equal the profile of the evaluated family and `solver` names the solver. Optional `campaign`, `reported_primal`, `reported_dual` and `evidence` fields describe the run, the solver's own bounds and a hash-pinned evidence file. No other field is accepted. The checker verifies the routes but not the claim. If a better submission displaces an annotated incumbent, the store keeps the original BKS bytes under `.bks-history/<sha256>.json`. The new submission can carry its own solver claim.

## Migrating Python code to v0.13

The v0.13 library requires Pydantic 2.10 or newer for TD route results with rational durations and departures. [Pydantic Fraction support](https://pydantic.dev/articles/pydantic-v2-10-release#support-for-fractionsfraction)

| Previous API | v0.13 API | Meaning |
| --- | --- | --- |
| Static `check_solution`, `check_cvrp_solution`, `check_vrptw_solution` and `td.check_td_solution` | Same names, exact family evaluation | Feasibility and cost use the materialized binary64 rationals. `result.routing_cost` is their display, with `None` on overflow. `result.exact_evaluation` carries the authoritative cost and ordered TD witnesses. The checker uses the family's published objective by default. |
| Static `compute_route_cost`, `compute_solution_cost` and TD `compute_solution_cost` | Same names, `Fraction` return value | Static helpers price the listed arcs even for a partial walk. The TD solution helper checks the submitted routes and includes any applicable fleet charge. All accept `backend="python"` for reference evaluation. |
| `td.compute_route_duration` | Same name, exact route timing | On a feasible route, `duration` and `departure_time` are fractions. The latter is the earliest minimizing departure. |
| `td.compute_route_ready_time_function` | `td.fold2.compute_route_ready_time_function` | Reproduce a ready-time curve from v0.12. Use `td.compute_route_duration` for the family-exact minimum and `NDCPWLF.evaluate_exact` for exact point interpolation. |
| `BenchmarkInstance` | `BenchmarkInstanceVRPTW` | The static VRPTW model with an embedded matrix. The JSON Schema title uses the new name. The instance JSON fields keep their spelling. |
| `save_bks_if_improved(..., bks=record)` | `save_bks_if_improved(..., candidate_bks=record)` | Recompute the candidate and incumbent, then write a strict improvement. Positional calls still work. |
| `save_bks(record, path)` | `write_bks_file(record, path)` | Serialize the model to the given path, overwriting an existing file. Use `submit_bks` for checked submission. |
| `exact_cost_value(number)` | `shortest_decimal_cost_value(number)` | Recover the rational value of the shortest decimal spelling of the parsed number. |
| `compute_exact_solution_cost(instance, routes)` | `compute_shortest_decimal_solution_cost(instance, routes)` | Sum the static arc costs using that historical decimal interpretation. For the v0.13 binary64 profile use `evaluate_exact(instance, routes).cost`. |
| `OptimalityMetadata` (a `proven: true` stamp) under `metadata.optimality` | `SolverReportedOptimality` | Removed. A file that carries a legacy stamp is not read, and `load_bks` raises `LegacyOptimalityClaimError`. Read such historical files with mamut-routing-lib 0.12.0. |
| `BKSUpdateResult.action` and the `action` of `preview_bks_update` as plain strings | `BKSUpdateAction` and `BKSPreviewAction` | Closed `Literal` vocabularies with the same values. An unknown value raises `ValueError`. |
| A `depot` other than 0 or a nonzero depot service time, refused when the evaluator is prepared | Refused when the instance is validated | Loading such a descriptor raises a validation error. |
| `save_bks_if_improved` trusting the supplied instance when no manifest exists | Same name, checked against the descriptor | The supplied instance must equal the descriptor and hydrated matrix at `instance_path`. A missing descriptor raises `FileNotFoundError`. |
| `canonical_route_order(routes)` without input checks | Same name, validated input | Routes must be nonempty, with positive integer customer ids and no repeated customer. |
| `SolutionCheckResult` with any field combination | Same model, validated | A `valid` result carries a nonnegative `num_routes` and a `routing_cost` that is `None` exactly on display overflow. |
| `mamut-routing bks reprice-td` and `mamut_routing_lib.td.reprice.reprice_td_bks` | Removed | Every current TD record carries exact metadata, which the command refused. Re-price records of earlier library versions with mamut-routing-lib 0.12.0. |
| `mamut_routing_lib.td.reprice.route_arcs` and `load_td_instance_for_routes` | `mamut_routing_lib.td.selected_arcs` | Same functions under a new module path. |

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

Version 0.13 reads family-first release archives. Until the first such snapshot is published, read the current edition from the collection's pinned family checkouts. For the September 2026 snapshot, use version 0.12.0 in a separate environment. The following commands apply to family-first releases.

```bash
# List archives available in the latest release of the configured repo
mamut-routing remote --repo ANR-MAMUT/MAMUT-routing list

# Filter by problem-type/benchmark-name (a problem-type filter also keeps the
# family-first collections, which ship every problem type of their family)
mamut-routing remote list --problem-type CVRP

# Download and extract archives into --benchmarks-dir
mamut-routing --benchmarks-dir ./benchmarks remote \
    fetch --benchmark-name Poryos2026

# Verify local zips (or extracted trees) against the remote manifest
mamut-routing --benchmarks-dir ./benchmarks remote verify --benchmark-name Poryos2026

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
| `kayros` | TDVRP, TDVRPTW (Duration, or FleetCostDuration for an instance with a fixed vehicle charge) | a compatible Kayros 2.0 development build from the collection or tools installation guide. PyPI Kayros 1.7.1 requires a library version below 0.12 |
| `hgs-cvrp` | CVRP (MonoCost), any metric (the published matrix, asymmetric included) | a build with the lib's full-matrix reader patch (`mamut-routing solvers list` prints its path) |
| `filo2`, `ails2` | CVRP (MonoCost), euclidean-metric variants (they compute distances from coordinates) | the executable (AILS-II: a jar built with the packaged `ails2-seed-out.patch` for `-seed` and `-out`) |
| `lkh3` | CVRP and VRPTW (MonoCost), any metric (explicit integer matrices, asymmetric included) | the `LKH` executable |

```bash
pip install "mamut-routing-lib[cli,pyvrp]"
mamut-routing solvers list          # what is available here, and how to enable the rest
```

To build HGS-CVRP with the installed full-matrix reader patch, run the following from an environment with `mamut-routing-lib` installed and CMake and a C++ compiler on the path.

```bash
git clone https://github.com/vidalt/HGS-CVRP.git
git -C HGS-CVRP checkout 1a927955cd2861a29d978f0d359d6e647db9319c
hgs_patch=$(python -c 'from mamut_routing_lib.solvers.external.catalog import patch_path; print(patch_path("hgs-cvrp-explicit-matrix.patch"))')
git -C HGS-CVRP apply "$hgs_patch"
cmake -S HGS-CVRP -B HGS-CVRP/build -DCMAKE_BUILD_TYPE=Release
cmake --build HGS-CVRP/build
```

Set `solvers.hgs-cvrp.executable` to the absolute path of `HGS-CVRP/build/hgs` in `solvers.toml`.

To build AILS-II with JDK 21, run the following from an environment with `mamut-routing-lib` installed. The patch adds a shared random seed and writes the best routes to the file requested by `-out`. It applies to the authors' INFORMS snapshot at `4ff3611` and the Mamut2026 campaign source at `e631331`. [AILS-II snapshot](https://github.com/INFORMSJoC/2023.0106/tree/4ff361153681635149e30305057251e9035ec6b7) · [Campaign source](https://github.com/vinymax10/AILS-CVRP/tree/e631331c76f001e652af9895e27f0fd8da8698c0)

```bash
git clone https://github.com/INFORMSJoC/2023.0106.git
cd 2023.0106
git checkout 4ff361153681635149e30305057251e9035ec6b7
git apply "$(python -c 'from mamut_routing_lib.solvers.external.catalog import patch_path; print(patch_path("ails2-seed-out.patch"))')"
mkdir -p build/classes
find src -name '*.java' > build/sources.txt
javac --release 21 -encoding UTF-8 -d build/classes @build/sources.txt
jar --create --file build/AILSII.jar --main-class SearchMethod.AILSII -C build/classes .
```

Set `solvers.ails2.jar` to the absolute path of the resulting `build/AILSII.jar` in `solvers.toml`.

Python:

```python
from mamut_routing_lib import run_solver, submit_bks

solved = run_solver("pyvrp", "path/to/instance.vrp.json", time_limit_s=30, seed=42)
print(solved.valid, solved.cost, solved.num_routes)        # the checker's verdict and cost
print(solved.result.solver_report, solved.result.solver_cost)  # original report and optional family-unit display

if solved.valid:                                           # store only if strictly better
    update = submit_bks(solved.instance_path, solved.result.routes,
                        authors="Your Name (handle)", metadata=solved.bks_metadata(),
                        checkable=solved.checkable)        # the instance the check already loaded
    print(update.action)                                   # created / replaced / kept_existing
```

A solver report preserves the objective, the original numerical value and its unit. Integer and printed decimal values retain their tokens as strings. A finite binary64 API value is a JSON number. The optional `routing_cost_component` holds a separately reported route cost, for example the distance component of PyVRP's vehicle-penalized objective. `solver_cost` rounds the comparable component times `family_units_per_solver_unit` once to binary64. A solver can submit routes without a numerical report. BKS comparisons use the checker's rational evaluation.

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
last incumbent, and so does a solve whose worker the system kills, for example
when memory runs out: the row is solved, and a line says how the worker ended).
SIGTERM and SIGHUP (`timeout 60 mamut-routing solve`, a systemd
stop, a closed terminal) end the command the same way. A command that is killed
outright (SIGKILL, a crash) cannot stop anything itself: on Linux and macOS each
solver process has a guard that ends it once the command is gone. Exit status: 0
when every solved instance is feasible and checker-valid, 1 when one is not, 2 on
usage errors, 130 after Ctrl-C, 143 after SIGTERM.

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

Relative paths are read from the config file's folder; a bare executable name is looked up on PATH, in its absolute
entries only (a relative entry such as `.` is the working directory, which is never searched). Unknown keys
and mistyped values (`fleet_limit = "false"`) are errors, with a suggestion for near-miss keys; an error in one
`[solvers.<name>]` table disables that solver only (`solvers list` shows why), and `extends` may name a table defined
anywhere in the file.

A solver declared in the config is `symmetric_only = true` unless it says otherwise, so the asymmetric
road-network variants (`fastest`, `shortest`) are refused for it rather than handed over as a `TYPE : CVRP` file a
symmetric solver would misread; `extends = "lkh3"` inherits LKH-3's asymmetric support.

A `cvrplib-sol` solution file is read with the numbering its solver declares: `id_convention = "one-based"` (the
default: customers 1..n as CVRPLIB writes them, a 0 is the depot), `"zero-based"` (customers 0..n-1) or `"auto"`.
The numbering is not guessed unless the solver asks for it, because a guess can complete an incomplete answer:
`Route #1: 0 1 2` for 3 customers is a complete zero-based file, or a one-based file that writes the depot and
misses customer 3. The four built-in solvers write one-based files.

Use `--keep-artifacts DIR` to retain an external solver's input and output. For an explicit integer matrix the adapter first uses the largest common dyadic unit of the materialized binary64 arc costs. It includes service times and time-window bounds in this calculation when the instance has time windows. The adapter writes exact integer multiples of the unit if the arc weights, total costs and time values fit the solver's declared limits. An integer solver with fixed-width arithmetic should declare these limits. For example, when all nonzero costs are the binary64 value 0.1 and the model contains no other time operands, the adapter writes weight 1 and records its value as 3602879701896397/36028797018963968.

When the required integers are too large, the adapter can export a decimal grid for search. Set `require_exact_scale = true` on an explicit integer solver to refuse that fallback. A rounded export that empties a time window is unsupported. HGS-CVRP reads a floating-point matrix and coordinate solvers compute distances internally. The run's `provenance.scale.search_model` identifies its input model. For integer matrices `provenance.scale.cost_unit` gives the exact numerator and denominator of one solver unit.

Every accepted static candidate uses `binary64-exact-v1` for final pricing and BKS comparison. Time-dependent candidates use `td-binary64-exact-v1`. A solver claim of optimality must target the profile of the evaluated family. The checker verifies the routes, not the claim.

When an integer solver returns routes and a cost for an explicit matrix, the adapter sums the actual exported weights, including the depot legs, and compares the report in solver units. A disagreement or fractional integer-solver report produces a warning. The benchmark checker still validates and prices the routes independently. A solver that only prints a rounded floating-point estimate can supply a route without this comparison.

The runner stops the solver process group on cancellation or hard timeout and includes the final log lines if a solver exits without a solution file. Run provenance records the command, input digest, scale and identifiable solver file. A Python module invocation can identify its interpreter even when the code has no single script to hash. A configured script used as the Python entry point and a Java jar passed to the JVM are recorded by their file hashes.

With `input_format = "lkh"`, each `extra_args` item supplies one parameter line, such as `RUNS = 10`. The runner owns `PROBLEM_FILE`, `MTSP_SOLUTION_FILE`, `SEED`, `TIME_LIMIT`, `PRECISION` and `SCALE`. Other input formats append `extra_args` to the command line.

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

For static v0.12 replay, import `check_solution`, `compute_route_cost` or `compute_solution_cost` from `mamut_routing_lib.static_v012`. The separately named `compute_shortest_decimal_solution_cost` diagnostic sums the shortest decimal spellings of the parsed operands.

For `td-fold/2` replay, import `check_td_solution` and the route helpers from `mamut_routing_lib.td.fold2`. Records written by earlier library versions are re-priced with mamut-routing-lib 0.12.0, whose `mamut-routing bks reprice-td` command this version no longer ships.

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

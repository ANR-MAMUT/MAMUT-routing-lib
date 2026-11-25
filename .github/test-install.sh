#!/usr/bin/env bash
# Exercise the active installation, including the exported CMake target.
set -euo pipefail
project=$(cd "$1" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
python -c 'import mamut_routing_lib._static_exact, mamut_routing_lib._td_exact, pyvrp'
python -m pytest -q -ra "$project/tests"
cmake -S "$project/tests/native_consumer" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DMAMUTRouting_DIR="$(python -m mamut_routing_lib.cmake)"
cmake --build "$build" --parallel 2
ctest --test-dir "$build" --output-on-failure

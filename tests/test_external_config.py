"""The user solvers config: lookup order, overrides, new solvers, and loud errors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mamut_routing_lib.solvers import get_solver, registry, run_solver, solver_names
from mamut_routing_lib.solvers.external.config import (
    CONFIG_ENV,
    SolverConfigError,
    configured_descriptors,
    default_config_path,
    resolve_config_path,
)

FAKE_SOLVER = """
[solvers.fake]
label = "Fake"
argv = ["{python}", "-m", "mamut_routing_lib.solvers.external._fake_solver", "--params", "{params}"]
runtime = "python"
input_format = "lkh"
output_format = "lkh-mtsp"
problem_types = ["CVRP"]
"""


def _config(tmp_path: Path, text: str, name: str = "solvers.toml") -> Path:
    path = tmp_path / name
    path.write_text("schema_version = 1\n" + text, encoding="utf-8")
    return path


def test_lookup_order(tmp_path: Path, monkeypatch) -> None:
    assert resolve_config_path() is None  # the autouse fixture points XDG at an empty directory
    default = default_config_path()
    default.parent.mkdir(parents=True)
    default.write_text("schema_version = 1\n")
    assert resolve_config_path() == default.resolve()
    explicit = _config(tmp_path, "")
    monkeypatch.setenv(CONFIG_ENV, str(explicit))
    assert resolve_config_path() == explicit.resolve()
    monkeypatch.setenv(CONFIG_ENV, "")
    assert resolve_config_path() is None  # an empty variable disables the config
    assert resolve_config_path(explicit) == explicit.resolve()
    with pytest.raises(SolverConfigError, match="not found"):
        resolve_config_path(tmp_path / "missing.toml")


def test_the_working_directory_is_never_searched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _config(tmp_path, '[solvers.filo2]\nexecutable = "/bin/sh"\n')
    (tmp_path / "mamut-routing").mkdir()
    _config(tmp_path / "mamut-routing", '[solvers.filo2]\nexecutable = "/bin/sh"\n')
    assert resolve_config_path() is None


def test_override_a_builtin(tmp_path: Path) -> None:
    path = _config(tmp_path, '[solvers.filo2]\nexecutable = "/opt/filo2"\nversion = "43653e2"\n')
    descriptor = configured_descriptors(path)["filo2"]
    assert descriptor.executable == "/opt/filo2" and descriptor.version == "43653e2"
    assert descriptor.argv[0] == "{executable}" and descriptor.coordinate_scales == (1000.0,)
    assert descriptor.source == str(path.resolve())


def test_extends_makes_a_named_variant(tmp_path: Path) -> None:
    path = _config(tmp_path, '[solvers.hgs-fork]\nextends = "hgs-cvrp"\nexecutable = "/opt/hgs"\n[solvers.hgs-fork.limits]\nmax_arc_cost = 1e6\n')
    descriptor = configured_descriptors(path)["hgs-fork"]
    assert descriptor.name == "hgs-fork" and descriptor.method == "hgs-fork"
    assert descriptor.fleet_args == ("-veh", "{vehicles}")
    assert descriptor.limits.max_arc_cost == 1e6 and descriptor.limits.max_demand == 1e5


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[solvers.filo2]\nexecutabel = "x"\n', "unknown key"),
        ('[solvers.new]\nargv = ["x"]\n', "a new solver needs"),
        ('[solvers.ails2]\njar = "relative.jar"\n', "absolute"),
        ('[solvers.x]\nextends = "nope"\n', "unknown solver"),
        ('[solvers.pyvrp]\nlabel = "x"\n', "in-process"),
        ('[solvers.td]\nargv = ["x"]\ninput_format = "lkh"\noutput_format = "lkh-mtsp"\nproblem_types = ["TDVRP"]\n', "static"),
        ('[solvers.x]\nargv = ["x"]\ninput_format = "cvrplib-x"\noutput_format = "cvrplib-sol"\nproblem_types = ["VRPTW"]\n', "time windows"),
        ("[other]\nkey = 1\n", "top-level"),
    ],
)
def test_configuration_errors_are_loud(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(SolverConfigError, match=message):
        configured_descriptors(_config(tmp_path, text))


def test_schema_version_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "solvers.toml"
    path.write_text("schema_version = 2\n")
    with pytest.raises(SolverConfigError, match="schema_version"):
        configured_descriptors(path)


def test_a_configured_solver_runs_through_the_registry(tmp_path: Path, monkeypatch, toy_cvrp_instance) -> None:
    path = _config(tmp_path, FAKE_SOLVER)
    monkeypatch.setenv(CONFIG_ENV, str(path))
    registry.reload_solvers()
    assert "fake" in solver_names()
    solver = get_solver("fake")
    assert solver.availability().available
    instance = tmp_path / "toy.vrp.json"
    instance.write_text(json.dumps(toy_cvrp_instance.model_dump(mode="json")))
    solved = run_solver("fake", instance, time_limit_s=5)
    assert solved.valid and solved.result.solver == "fake"
    assert solved.result.provenance["descriptor_source"] == str(path.resolve())


def test_explicit_config_argument(tmp_path: Path) -> None:
    path = _config(tmp_path, FAKE_SOLVER)
    assert "fake" in solver_names(config=path)
    assert "fake" not in solver_names()


_EXPLICIT = """
[solvers.mine]
runtime = "python"
input_format = "cvrplib-explicit"
output_format = "cvrplib-sol"
problem_types = ["CVRP"]
argv = ["{python}", "-m", "mamut_routing_lib.solvers.external._fake_solver", "--problem", "{problem}", "--solution", "{solution}"]
"""


def _asymmetric(toy_cvrp_instance, tmp_path: Path) -> Path:
    asym = toy_cvrp_instance.model_copy(update={"arc_costs": [[0, 5, 6], [7, 0, 3], [6, 4, 0]]})
    path = tmp_path / "asym.vrp.json"
    path.write_text(json.dumps(asym.model_dump(mode="json")))
    return path


def test_a_configured_solver_is_symmetric_only_by_default(tmp_path: Path, toy_cvrp_instance) -> None:
    from mamut_routing_lib.solvers import InstanceNotSupportedError

    path = _config(tmp_path, _EXPLICIT)
    assert configured_descriptors(path)["mine"].capabilities.symmetric_only is True
    with pytest.raises(InstanceNotSupportedError, match="symmetric"):
        run_solver("mine", _asymmetric(toy_cvrp_instance, tmp_path), time_limit_s=5, config=path)


def test_symmetric_only_false_opts_in_to_asymmetric_matrices(tmp_path: Path, toy_cvrp_instance) -> None:
    path = _config(tmp_path, _EXPLICIT + "symmetric_only = false\n")
    assert configured_descriptors(path)["mine"].capabilities.symmetric_only is False
    solved = run_solver("mine", _asymmetric(toy_cvrp_instance, tmp_path), time_limit_s=5, config=path)
    assert solved.valid


def test_extends_inherits_the_builtin_symmetry_requirement(tmp_path: Path) -> None:
    path = _config(
        tmp_path,
        '[solvers.lkh-fork]\nextends = "lkh3"\nexecutable = "/opt/lkh"\n'
        '[solvers.hgs-fork]\nextends = "hgs-cvrp"\nexecutable = "/opt/hgs"\n'
        '[solvers.filo-fork]\nextends = "filo2"\nexecutable = "/opt/filo2"\n',
    )
    descriptors = configured_descriptors(path)
    assert descriptors["lkh-fork"].capabilities.symmetric_only is False
    assert descriptors["hgs-fork"].capabilities.symmetric_only is False  # full-matrix reader patch
    assert descriptors["filo-fork"].capabilities.symmetric_only is True

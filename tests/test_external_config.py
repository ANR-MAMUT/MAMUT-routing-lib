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
        ('[solvers.filo2]\nexecutabel = "x"\n', "unknown key.*did you mean 'executable'"),
        ('[solvers.new]\nargv = ["x"]\n', "a new solver needs"),
        ('[solvers.filo2]\nfleet_limit = "false"\n', "true or false"),
        ('[solvers.filo2]\nsymmetric_only = "false"\n', "true or false"),
        ('[solvers.filo2]\nlive_progress = "always"\n', "live_progress must be one of"),
        ('[solvers.filo2]\nruntime = "jvm"\n', "runtime must be one of"),
        ('[solvers.filo2]\nwarm_start = true\n', "no warm start"),
        ('[solvers.filo2]\nargv = ["{executable}", "{problme}"]\n', "did you mean 'problem'"),
        ('[solvers.filo2]\nargv = ["{executable", "{problem}"]\n', "literal braces"),
        ('[solvers.filo2]\nvehicles_factor = "2"\n', "must be a number"),
        ('[solvers.filo2]\nprogress_pattern = "(unclosed"\n', "regular expression"),
        ('[solvers.a]\nextends = "b"\n[solvers.b]\nextends = "a"\n', "circular"),
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


def test_one_bad_table_disables_only_its_solver(tmp_path: Path, monkeypatch) -> None:
    from mamut_routing_lib.solvers import SolverUnavailableError
    from mamut_routing_lib.solvers.external.config import BrokenSolver, load_config

    path = _config(tmp_path, FAKE_SOLVER + '[solvers.filo2]\nexecutabel = "/opt/filo2"\n[solvers.my]\nargv = ["x"]\n')
    loaded = load_config(path)
    assert set(loaded.descriptors) == {"fake"} and set(loaded.errors) == {"filo2", "my"}
    monkeypatch.setenv(CONFIG_ENV, str(path))
    registry.reload_solvers()
    assert get_solver("fake").availability().available
    broken = get_solver("filo2")
    assert isinstance(broken, BrokenSolver) and broken.capabilities.problem_types  # the built-in's, for listing
    availability = broken.availability()
    assert not availability.available and "did you mean 'executable'" in availability.reason
    assert str(path.resolve()) in availability.hint
    with pytest.raises(SolverUnavailableError, match="config error"):
        get_solver("my").solve(None)
    assert get_solver("pyvrp").name == "pyvrp"
    with pytest.raises(SolverConfigError, match="executabel"):
        configured_descriptors(path)  # the strict reader still refuses the file


def test_extends_may_name_a_table_defined_later(tmp_path: Path) -> None:
    path = _config(
        tmp_path,
        '[solvers.lkh-long]\nextends = "lkh-mine"\nextra_args = ["RUNS = 10"]\n'
        '[solvers.lkh-mine]\nextends = "lkh3"\nexecutable = "/opt/lkh"\nversion = "3.0.9"\n',
    )
    descriptors = configured_descriptors(path)
    assert descriptors["lkh-long"].executable == "/opt/lkh" and descriptors["lkh-long"].version == "3.0.9"
    assert descriptors["lkh-long"].extra_args == ("RUNS = 10",)
    assert list(descriptors) == ["lkh-long", "lkh-mine"]  # file order kept


def test_an_error_in_a_parent_names_the_parent(tmp_path: Path) -> None:
    from mamut_routing_lib.solvers.external.config import load_config

    path = _config(tmp_path, '[solvers.child]\nextends = "parent"\n[solvers.parent]\nextends = "lkh3"\nfleet_limit = "no"\n')
    errors = load_config(path).errors
    assert "true or false" in errors["parent"] and "[solvers.parent] has a config error" in errors["child"]


def test_relative_paths_are_read_from_the_config_folder(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    monkeypatch.chdir(tmp_path)  # never the working directory
    path = _config(
        config_dir,
        '[solvers.filo2]\nexecutable = "./bin/filo2"\n[solvers.lkh3]\nexecutable = "LKH"\n'
        '[solvers.ails2]\njar = "AILSII.jar"\njava = "jdk/bin/java"\n',
    )
    descriptors = configured_descriptors(path)
    assert descriptors["filo2"].executable == str(config_dir.resolve() / "bin" / "filo2")
    assert descriptors["lkh3"].executable == "LKH"  # a bare name is looked up on PATH
    assert descriptors["ails2"].jar == str(config_dir.resolve() / "AILSII.jar")
    assert descriptors["ails2"].java == str(config_dir.resolve() / "jdk" / "bin" / "java")


def test_lkh_extra_args_are_literal_parameter_lines(tmp_path: Path) -> None:
    path = _config(tmp_path, '[solvers.lkh3]\nextra_args = ["SPECIAL = {not a placeholder}"]\n')
    assert configured_descriptors(path)["lkh3"].extra_args == ("SPECIAL = {not a placeholder}",)


def _uncommented_tables(text: str) -> str:
    """The starter with every commented ``[solvers.x]`` table switched on."""
    lines, active = [], False
    for line in text.splitlines():
        if line.startswith("# [solvers."):
            active = True
        elif not line.startswith("# ") or " = " not in line and not line.startswith("# ["):
            active = False
        lines.append(line[2:] if active else line)
    return "\n".join(lines) + "\n"


def test_the_starter_config_parses_and_each_table_is_valid(tmp_path: Path) -> None:
    from mamut_routing_lib.solvers.external.config import load_config, starter_config_text

    text = starter_config_text(detect=False)
    starter = tmp_path / "starter.toml"
    starter.write_text(text)
    assert load_config(starter).descriptors == {} and load_config(starter).errors == {}
    enabled = tmp_path / "enabled.toml"
    enabled.write_text(_uncommented_tables(text))
    loaded = load_config(enabled)
    assert loaded.errors == {}
    assert set(loaded.descriptors) == {"filo2", "hgs-cvrp", "ails2", "lkh3", "lkh3-long", "my-solver"}
    assert loaded.descriptors["lkh3-long"].extra_args == ("RUNS = 10",)

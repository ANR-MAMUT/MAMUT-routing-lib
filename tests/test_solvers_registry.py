"""The solver registry: built-ins, plugins and registrations by name, without importing solvers."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from mamut_routing_lib.enums import ObjectiveFunction, ProblemType
from mamut_routing_lib.solvers import registry
from mamut_routing_lib.solvers.base import Availability, SolverCapabilities, Support


class _Dummy:
    name = "dummy"
    label = "Dummy"
    capabilities = SolverCapabilities(objectives={ProblemType.CVRP: frozenset({ObjectiveFunction.MONO_COST})})

    def availability(self) -> Availability:
        return Availability(True, version="1.0")

    def supports(self, profile, objective) -> Support:
        return Support.ok()

    def solve(self, request, monitor=None):  # pragma: no cover - not called here
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(registry._REGISTERED)
    registry.reload_solvers()
    yield
    registry._REGISTERED.clear()
    registry._REGISTERED.update(saved)
    registry.reload_solvers()


def test_importing_the_registry_loads_no_solver_package() -> None:
    code = (
        "import sys, mamut_routing_lib.solvers as s; s.solver_names(); s.get_solver('pyvrp').availability();"
        "print('pyvrp' in sys.modules, 'kayros' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_pyvrp_is_built_in_and_listed_first() -> None:
    names = registry.solver_names()
    assert names[0] == "pyvrp"
    assert registry.get_solver("pyvrp").name == "pyvrp"


def test_unknown_solver_lists_the_known_names() -> None:
    with pytest.raises(registry.UnknownSolverError, match="pyvrp"):
        registry.get_solver("no-such-solver")


def test_register_solver_adds_and_refuses_silent_replacement() -> None:
    registry.register_solver(_Dummy())
    assert "dummy" in registry.solver_names()
    assert registry.get_solver("dummy").availability().version == "1.0"
    with pytest.raises(ValueError, match="already registered"):
        registry.register_solver(_Dummy())
    registry.register_solver(_Dummy, name="dummy", replace=True)  # a class works as a factory
    assert isinstance(registry.get_solver("dummy"), _Dummy)


def test_entry_point_plugins_are_loaded(monkeypatch) -> None:
    entry = SimpleNamespace(name="plugin", load=lambda: _Dummy)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda group: [entry] if group == registry.ENTRY_POINT_GROUP else [])
    registry.reload_solvers()
    assert "plugin" in registry.solver_names()
    assert isinstance(registry.get_solver("plugin"), _Dummy)


def test_a_broken_plugin_warns_instead_of_hiding_the_others(monkeypatch) -> None:
    def explode():
        raise RuntimeError("boom")

    entries = [SimpleNamespace(name="broken", load=explode), SimpleNamespace(name="plugin", load=lambda: _Dummy)]
    monkeypatch.setattr(registry.metadata, "entry_points", lambda group: entries)
    registry.reload_solvers()
    with pytest.warns(UserWarning, match="broken"):
        names = registry.solver_names()
    assert "plugin" in names and "broken" not in names


def test_a_broken_solvers_config_does_not_block_in_process_solvers(tmp_path, monkeypatch) -> None:
    from mamut_routing_lib.solvers import get_solver, reload_solvers
    from mamut_routing_lib.solvers.external.config import SolverConfigError

    broken = tmp_path / "broken.toml"
    broken.write_text('schema_version = 1\n[solvers.filo2]\nexecutable = "/opt/filo2\n')  # unclosed string
    reload_solvers()
    for config in (broken, tmp_path / "missing.toml"):
        assert get_solver("pyvrp", config=config).name == "pyvrp"
        assert get_solver("kayros", config=config).name == "kayros"
        with pytest.raises(SolverConfigError):
            get_solver("filo2", config=config)
    monkeypatch.setenv("MAMUT_ROUTING_SOLVERS_CONFIG", str(broken))
    reload_solvers()
    assert get_solver("pyvrp").name == "pyvrp"

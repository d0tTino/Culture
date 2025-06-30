import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


sys.modules.setdefault("neo4j", DummyNeo4j())

from src import app  # noqa: E402

pytestmark = pytest.mark.unit


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--agents", "2", "--steps", "3"])
    args = app.parse_args()
    assert args.agents == 2
    assert args.steps == 3


def test_main_invokes_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--agents", "2", "--steps", "3"])
    dummy_sim = SimpleNamespace(async_run=MagicMock(return_value="coro"))

    create_sim = MagicMock(return_value=dummy_sim)
    monkeypatch.setattr(app, "create_simulation", create_sim)
    run_mock = MagicMock()
    monkeypatch.setattr(app.asyncio, "run", run_mock)

    monkeypatch.setattr(app, "load_checkpoint", MagicMock(return_value=(dummy_sim, None)))
    monkeypatch.setattr(app, "save_checkpoint", MagicMock())
    monkeypatch.setattr(app, "restore_rng_state", MagicMock())
    monkeypatch.setattr(app, "restore_environment", MagicMock())

    app.main()

    create_sim.assert_called_once_with(
        num_agents=2,
        steps=3,
        scenario=app.DEFAULT_SCENARIO,
        use_discord=False,
        use_vector_store=False,
        vector_store_dir="./weaviate_data",
        db_file=None,
    )
    dummy_sim.async_run.assert_called_once_with(3)
    run_mock.assert_called_once_with(dummy_sim.async_run.return_value)


def test_load_scenario_from_file(tmp_path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text("""description: Test scenario\nsteps: 7\nagents: 4\n""")

    desc, steps, agents = app.load_scenario(str(path))

    assert desc == "Test scenario"
    assert steps == 7
    assert agents == 4


def test_main_uses_scenario_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    scenario = tmp_path / "demo.yaml"
    scenario.write_text("""description: File scenario\nsteps: 2\nagents: 1\n""")
    monkeypatch.setattr(sys, "argv", ["prog", "--scenario", str(scenario)])
    dummy_sim = SimpleNamespace(async_run=MagicMock(return_value="coro"))

    create_sim = MagicMock(return_value=dummy_sim)
    monkeypatch.setattr(app, "create_simulation", create_sim)
    run_mock = MagicMock()
    monkeypatch.setattr(app.asyncio, "run", run_mock)

    monkeypatch.setattr(app, "load_checkpoint", MagicMock(return_value=(dummy_sim, None)))
    monkeypatch.setattr(app, "save_checkpoint", MagicMock())
    monkeypatch.setattr(app, "restore_rng_state", MagicMock())
    monkeypatch.setattr(app, "restore_environment", MagicMock())

    app.main()

    create_sim.assert_called_once_with(
        num_agents=1,
        steps=2,
        scenario="File scenario",
        use_discord=False,
        use_vector_store=False,
        vector_store_dir="./weaviate_data",
        db_file=None,
    )
    dummy_sim.async_run.assert_called_once_with(2)
    run_mock.assert_called_once_with(dummy_sim.async_run.return_value)

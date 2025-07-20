import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import app


@pytest.mark.unit
def test_main_invokes_load_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])
    dummy_sim = SimpleNamespace(async_run=MagicMock(return_value="coro"))
    monkeypatch.setattr(app, "create_simulation", MagicMock(return_value=dummy_sim))
    monkeypatch.setattr(app.asyncio, "run", MagicMock())
    monkeypatch.setattr(app, "load_checkpoint", MagicMock(return_value=(dummy_sim, None)))
    monkeypatch.setattr(app, "save_checkpoint", MagicMock())
    monkeypatch.setattr(app, "restore_rng_state", MagicMock())
    monkeypatch.setattr(app, "restore_environment", MagicMock())

    load_plugins_mock = MagicMock()
    monkeypatch.setattr(app, "load_plugins", load_plugins_mock)

    app.main()

    load_plugins_mock.assert_called_once_with()

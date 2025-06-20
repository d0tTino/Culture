import asyncio
import os
import sys
import types

import pytest

from src.utils.loop_helper import use_uvloop_if_available


@pytest.mark.unit
def test_use_uvloop_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "posix", raising=False)
    fake_uvloop = types.SimpleNamespace(EventLoopPolicy=lambda: "policy")
    monkeypatch.setitem(sys.modules, "uvloop", fake_uvloop)
    called = []

    def _set_policy(policy: object) -> None:
        called.append(policy)

    monkeypatch.setattr(asyncio, "set_event_loop_policy", _set_policy)

    use_uvloop_if_available()

    assert called and called[0] == "policy"


@pytest.mark.unit
def test_use_uvloop_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt", raising=False)
    called = False

    def _set_policy(policy: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(asyncio, "set_event_loop_policy", _set_policy)

    use_uvloop_if_available()

    assert not called

import asyncio
import builtins
import importlib
import sys
import types

import pytest

pytestmark = pytest.mark.unit


def reload_sitecustomize(monkeypatch: pytest.MonkeyPatch, platform: str, fake_uvloop: types.SimpleNamespace | None) -> list:
    monkeypatch.setattr(sys, "platform", platform, raising=False)
    if fake_uvloop is not None:
        monkeypatch.setitem(sys.modules, "uvloop", fake_uvloop)
    else:
        sys.modules.pop("uvloop", None)
        orig_import = __import__

        def _fake_import(name: str, *args: object, **kwargs: object):
            if name == "uvloop":
                raise ImportError
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
    called: list = []

    def _set_policy(policy: object) -> None:
        called.append(policy)

    monkeypatch.setattr(asyncio, "set_event_loop_policy", _set_policy)
    if "sitecustomize" in sys.modules:
        importlib.reload(sys.modules["sitecustomize"])  # type: ignore
    else:
        importlib.import_module("sitecustomize")
    return called


def test_sitecustomize_uses_uvloop_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_uvloop = types.SimpleNamespace(EventLoopPolicy=lambda: "policy")
    called = reload_sitecustomize(monkeypatch, "linux", fake_uvloop)
    assert called and called[0] == "policy"


def test_sitecustomize_handles_missing_uvloop(monkeypatch: pytest.MonkeyPatch) -> None:
    called = reload_sitecustomize(monkeypatch, "linux", None)
    assert not called


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_sitecustomize_skips_uvloop_on_non_linux(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    called = reload_sitecustomize(monkeypatch, platform, None)
    assert not called

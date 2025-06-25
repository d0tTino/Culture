import sys
import types

import pytest

from src.sim.version_vector import VersionVector


@pytest.fixture(autouse=True)
def stub_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal in-memory ledger to avoid sqlite side effects."""

    ledger_stub = types.ModuleType("src.infra.ledger")

    class Ledger:
        def __init__(self, *_, **__):
            pass

        def log_change(self, *_, **__):
            pass

        def register_hook(self, *_, **__):
            pass

    ledger_stub.Ledger = Ledger
    ledger_stub.ledger = Ledger()
    ledger_stub.__all__ = ["Ledger", "ledger"]

    monkeypatch.setitem(sys.modules, "src.infra.ledger", ledger_stub)


pytestmark = pytest.mark.unit


def test_increment_and_to_dict() -> None:
    vv = VersionVector()
    vv.increment("A")
    vv.increment("A")
    assert vv.to_dict() == {"A": 2}
    result = vv.to_dict()
    assert result == {"A": 2}
    assert result is not vv.clock


def test_merge_updates_counters() -> None:
    left = VersionVector({"A": 1})
    right = VersionVector({"A": 2, "B": 1})
    left.merge(right)
    assert left.to_dict() == {"A": 2, "B": 1}
    other = VersionVector({"A": 2, "B": 1})
    other.merge(VersionVector({"A": 1}))
    assert other.to_dict() == {"A": 2, "B": 1}


def test_compare_relations() -> None:
    a = VersionVector({"A": 1})
    b = VersionVector({"A": 2})
    assert a.compare(b) == -1
    assert b.compare(a) == 1
    c = VersionVector({"A": 2, "B": 1})
    d = VersionVector({"A": 2, "B": 1})
    assert c.compare(d) == 0


def test_compare_concurrent_cases() -> None:
    v1 = VersionVector({"A": 1})
    v2 = VersionVector({"B": 1})
    assert v1.compare(v2) == 2
    v3 = VersionVector({"A": 1, "B": 0})
    v4 = VersionVector({"A": 0, "B": 1})
    assert v3.compare(v4) == 2

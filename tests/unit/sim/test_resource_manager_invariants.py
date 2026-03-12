from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from src.sim.resource_manager import ResourceManager

pytestmark = pytest.mark.unit


def test_charge_is_atomic_under_concurrent_callers() -> None:
    rm = ResourceManager(5.0, 5.0)
    rm.set_du_budget("agent", 1.0)

    def worker() -> bool:
        try:
            rm.charge("agent", 0.4)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _i: worker(), range(4)))

    assert sum(results) == 2
    assert rm.get_du_budget("agent") == pytest.approx(0.0)


def test_budget_check_and_charge_paths_preserve_non_negative_budget() -> None:
    rm = ResourceManager(5.0, 5.0)
    rm.set_du_budget("agent", 1.0)

    def command_path() -> bool:
        try:
            rm.budget_check("agent", 0.2)
            rm.charge("agent", 0.2)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: command_path(), range(8)))

    assert sum(results) <= 5
    assert rm.get_du_budget("agent") >= 0.0
    assert rm.get_du_budget("agent") == pytest.approx(0.0)

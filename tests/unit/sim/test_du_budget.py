import pytest

from src.sim.resource_manager import ResourceManager

pytestmark = pytest.mark.unit


def test_charge_du_budget_decrements() -> None:
    rm = ResourceManager(5.0, 5.0)
    rm.set_du_budget("agent", 0.5)
    rm.charge_du("agent", 0.1)
    assert rm.get_du_budget("agent") == pytest.approx(0.4)


def test_charge_du_budget_overuse_raises() -> None:
    rm = ResourceManager(5.0, 5.0)
    rm.set_du_budget("agent", 0.1)
    with pytest.raises(RuntimeError):
        rm.charge_du("agent", 0.2)

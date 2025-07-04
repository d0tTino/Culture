import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.agents.core import ResourceManager

pytestmark = pytest.mark.unit


class Dummy:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.xfail(reason="Hypothesis SimpleNamespace hash bug")
@given(
    start_ip=st.floats(min_value=0, max_value=100),
    start_du=st.floats(min_value=0, max_value=100),
    ip_gain=st.floats(min_value=0, max_value=50),
    du_gain=st.floats(min_value=0, max_value=50),
)
def test_cap_tick(start_ip: float, start_du: float, ip_gain: float, du_gain: float) -> None:
    mgr = ResourceManager(5.0, 7.0)
    d = Dummy()
    d.ip = start_ip + ip_gain
    d.du = start_du + du_gain

    mgr.cap_tick(ip_start=start_ip, du_start=start_du, obj=d)
    assert d.ip - start_ip <= 5.0 + 1e-6
    assert d.du - start_du <= 7.0 + 1e-6

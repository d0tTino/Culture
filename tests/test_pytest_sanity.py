import pytest


@pytest.mark.unit
@pytest.mark.critical_path
def test_pytest_sanity() -> None:
    assert True

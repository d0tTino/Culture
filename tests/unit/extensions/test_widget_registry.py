import pytest

from src.interfaces.widget_registry import WidgetRegistry


@pytest.mark.unit
def test_register_and_get() -> None:
    reg = WidgetRegistry()
    reg.register("test", {"foo": "bar"})
    assert reg.get("test") == {"foo": "bar"}
    assert reg.get("missing") is None


@pytest.mark.unit
def test_list_sorted() -> None:
    reg = WidgetRegistry()
    reg.register("b", {"meta": 2})
    reg.register("a", {"meta": 1})
    assert reg.list() == [
        {"name": "a", "meta": 1},
        {"name": "b", "meta": 2},
    ]

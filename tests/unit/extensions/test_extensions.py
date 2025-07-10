import pytest

from src.extensions import BEHAVIOR_REGISTRY, register_agent_behavior


@pytest.mark.unit
def test_behavior_registry_runs() -> None:
    calls: list[dict[str, object]] = []

    def plugin(agent: object, output: dict[str, object]) -> None:
        calls.append(output)

    BEHAVIOR_REGISTRY._behaviors.clear()
    register_agent_behavior(plugin)
    BEHAVIOR_REGISTRY.run(object(), {"msg": "hi"})
    assert calls and calls[0]["msg"] == "hi"

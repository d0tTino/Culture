import pytest

from src.interfaces.interaction_commands import InteractionContext
from src.interfaces.interaction_schema import ControlEnvelope, HumanMessageEnvelope
from src.sim.simulation import Simulation


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 10.0
        self.du = 10.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyAgentState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: DummyAgentState) -> None:
        self.state = state

    async def run_turn(self, *args: object, **kwargs: object) -> dict:
        return {}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "web", "api"])
async def test_human_message_envelope_has_same_result_across_transports(source: str) -> None:
    sim = Simulation([DummyAgent("alpha"), DummyAgent("beta")])
    envelope = HumanMessageEnvelope(
        text="transport invariance",
        routing={"sender_id": "user-1", "source": source},
        correlation_id="corr-transport-1",
    )
    context = InteractionContext(sender_id="user-1", source=source)

    try:
        if source == "discord":
            result = await sim.command_bus.dispatch(envelope, context=context)
        elif source == "web":
            result = await sim.command_bus.dispatch_payload(envelope.model_dump(), context=context)
        else:
            result = await sim.interaction_service.execute_from_payload(
                envelope.model_dump(),
                context=context,
            )

        assert result.status == "ok"
        assert result.reason_code == "message_dispatched"
        assert result.correlation_id == "corr-transport-1"
    finally:
        await sim.stop_event_listener()
        sim.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_envelope_dispatches_consistently_for_web_and_api() -> None:
    sim = Simulation([DummyAgent("alpha")])
    envelope = ControlEnvelope(
        action="set_speed",
        value=2.5,
        routing={"sender_id": "moderator-1", "source": "web"},
        auth={"permissions": {"admin"}},
        correlation_id="corr-control-1",
    )
    context = InteractionContext(sender_id="moderator-1", source="web", permissions={"admin"})

    try:
        web_result = await sim.command_bus.dispatch_payload(envelope.model_dump(), context=context)
        api_result = await sim.interaction_service.execute_from_payload(
            envelope.model_dump(),
            context=context,
        )

        assert web_result.status == api_result.status == "ok"
        assert web_result.reason_code == api_result.reason_code == "moderation_applied"
        assert web_result.correlation_id == "corr-control-1"
        assert api_result.correlation_id == "corr-control-1"
    finally:
        await sim.stop_event_listener()
        sim.close()

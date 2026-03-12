import pytest

from src.interfaces import dashboard_backend as db
from src.interfaces import discord_bot
from src.interfaces.interaction_commands import InteractionContext
from src.interfaces.interaction_policy import discord_message_to_intent_payload
from src.interfaces.interaction_schema import HumanMessageEnvelope
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
async def test_equivalent_human_message_has_identical_policy_outcome() -> None:
    sim = Simulation([DummyAgent("alpha"), DummyAgent("beta")])
    try:
        envelope = HumanMessageEnvelope(
            text="hello kernel",
            routing={"sender_id": "user-1", "source": "discord"},
            metadata={"request_id": "equiv-1"},
        )

        discord_result = await sim.command_bus.dispatch(
            envelope,
            context=InteractionContext(sender_id="user-1", source="discord"),
        )
        dashboard_result = await sim.command_bus.dispatch_payload(
            {
                "command_type": "human_message",
                "text": "hello kernel",
                "sender_id": "user-1",
                "source": "dashboard",
                "request_id": "equiv-1",
            },
            context=InteractionContext(sender_id="user-1", source="dashboard"),
        )

        assert discord_result.status == dashboard_result.status == "ok"
        assert discord_result.reason_code == dashboard_result.reason_code == "message_dispatched"
        assert discord_result.decision_provenance == dashboard_result.decision_provenance
    finally:
        await sim.stop_event_listener()
        sim.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_equivalent_control_request_has_identical_policy_outcome() -> None:
    sim = Simulation([DummyAgent("alpha")])
    try:
        db.DEFAULT_CONTEXT.sim_state["simulation"] = sim

        bus_result = await sim.command_bus.dispatch_payload(
            {
                "command": "set_speed",
                "value": 2.0,
                "sender_id": "moderator-1",
                "source": "discord",
                "permissions": ["admin"],
            },
            context=InteractionContext(
                sender_id="moderator-1",
                source="discord",
                permissions={"admin"},
            ),
        )
        endpoint_result = await db.handle_control_command(
            {"command": "set_speed", "value": 2.0},
            ctx=db.DEFAULT_CONTEXT,
        )

        assert bus_result.status == endpoint_result["status"] == "ok"
        assert bus_result.reason_code == endpoint_result["reason_code"] == "moderation_applied"
        assert (
            bus_result.decision_provenance.model_dump()
            == endpoint_result["decision_provenance"]
        )
    finally:
        db.DEFAULT_CONTEXT.sim_state.pop("simulation", None)
        await sim.stop_event_listener()
        sim.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversational_input_routes_to_world_effect_and_acknowledgement() -> None:
    sim = Simulation([DummyAgent("alpha")])
    try:
        payload, err = discord_message_to_intent_payload(
            content="/broadcast Coordinate relief effort",
            sender_agent_id="alpha",
            fallback_agent_id="alpha",
            raw_metadata={"channel_id": 1, "user_id": 2},
            mode="participant",
        )
        assert err is None
        assert payload is not None
        assert payload["intent"] == "broadcast"

        result = await sim.command_bus.dispatch_payload(
            payload,
            context=InteractionContext(sender_id="user-1", source="discord"),
        )
        assert result.status == "ok"
        assert result.reason_code == "message_dispatched"

        async with sim._msg_lock:
            pending = sim.pending_messages_for_next_round[-1]
        assert pending["content"] == "Coordinate relief effort"

        acknowledgement = discord_bot.format_explainable_acknowledgement(result)
        assert "Outcome:" in acknowledgement
        assert "Why:" in acknowledgement
    finally:
        await sim.stop_event_listener()
        sim.close()

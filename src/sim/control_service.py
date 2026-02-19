from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from src.agents.core.agent_state import AgentLifecycleState, PersonalityTraits
from src.agents.core.roles import ensure_profile, get_role_trait_template
from src.interfaces.command_bus import parse_bus_command
from src.interfaces.dashboard_backend import SimulationEvent, emit_event
from src.shared.typing import SimulationMessage
from src.sim.knowledge_board import BoardEntry

logger = logging.getLogger(__name__)


class SimulationControlService:
    """Control-plane command handling for simulation command/control logic."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation

    async def handle_control_command(self, cmd: Mapping[str, Any]) -> dict[str, Any] | None:
        sim = self.simulation
        parsed: Any = None
        try:
            parsed = parse_bus_command(cmd)
        except ValidationError:
            logger.warning("Invalid control command payload; falling back to raw command: %s", cmd)
        action = getattr(parsed, "action", None) or str(cmd.get("command", ""))
        if action == "pause":
            sim.paused = True
        elif action == "resume":
            sim.paused = False
        elif action == "pause_all":
            sim.paused = True
            kernel = getattr(sim, "event_kernel", None)
            if kernel is not None and hasattr(kernel, "pause"):
                try:
                    kernel.pause()
                except Exception:  # pragma: no cover
                    logger.debug("Kernel pause failed", exc_info=True)
        elif action == "start":
            sim.paused = False
        elif action == "stop":
            sim.simulation_complete = True
            await sim.stop_event_listener()
        elif action == "spawn":
            agent_id = getattr(parsed, "agent_id", None) or cmd.get("agent_id")
            if agent_id:
                normalized_agent_id = str(agent_id)
                if any(agent.agent_id == normalized_agent_id for agent in sim.agents):
                    await emit_event(
                        SimulationEvent(
                            type="spawn_rejected",
                            data={
                                "reason": "duplicate_agent_id",
                                "agent_id": normalized_agent_id,
                                "step": sim.current_step,
                            },
                        )
                    )
                    return None
                try:
                    from src.agents.core.base_agent import Agent

                    role_value = getattr(parsed, "role", None) if hasattr(parsed, "role") else None
                    role_profile = ensure_profile(role_value) if role_value is not None else None
                    initial_state: dict[str, Any] = {}
                    if role_profile is not None:
                        initial_state["current_role"] = role_profile
                    traits_payload = getattr(parsed, "traits", None) if hasattr(parsed, "traits") else None
                    if isinstance(traits_payload, dict):
                        merged_traits = get_role_trait_template(
                            role_profile.name if role_profile is not None else "Innovator"
                        )
                        for trait_name, raw_value in traits_payload.items():
                            merged_traits[str(trait_name)] = float(raw_value)
                        initial_state["traits"] = PersonalityTraits(**merged_traits)

                    new_agent = Agent(
                        agent_id=normalized_agent_id,
                        name=normalized_agent_id,
                        initial_state=initial_state or None,
                    )
                    await sim.spawn_agent(new_agent)
                except Exception:
                    logger.error("Failed to spawn agent %s", agent_id, exc_info=True)
        elif action == "kill_agent":
            agent_id = getattr(parsed, "agent_id", None) or cmd.get("agent_id")
            if agent_id:
                agent = next((a for a in sim.agents if a.agent_id == str(agent_id)), None)
                if agent is not None:
                    await sim.retire_agent(
                        agent,
                        remove_from_simulation=True,
                        lifecycle_state=AgentLifecycleState.DECEASED,
                        reason="kill_agent_command",
                    )
        elif action == "set_speed":
            try:
                sim.speed = float(getattr(parsed, "value", None) or cmd.get("value", 1))
            except (TypeError, ValueError):
                pass
        elif action == "post_kb":
            text = getattr(parsed, "text", None) or cmd.get("text")
            author = getattr(parsed, "author", None) or cmd.get("author", "human")
            if text and sim.knowledge_board:
                async with sim.knowledge_board.lock:
                    sim.knowledge_board.add_entry(
                        BoardEntry(
                            content_full=text,
                            entry_type="human_message",
                            tags=["human", "moderation"],
                        ),
                        str(author),
                        sim.current_step,
                        sim.vector.to_dict(),
                    )
                await emit_event(
                    SimulationEvent(
                        type="knowledge_board",
                        data={"agent_id": str(author), "content": text, "step": sim.current_step},
                    )
                )
        elif action == "inject_event":
            text = str(getattr(parsed, "text", None) or cmd.get("text", "")).strip()
            if not text:
                return None
            author = str(getattr(parsed, "author", None) or cmd.get("author", "human"))
            scope = str(getattr(parsed, "scope", None) or cmd.get("scope", "global"))
            event_payload = {
                "type": "world_event",
                "author": author,
                "step": sim.current_step,
                "timestamp": time.time(),
                "scope": scope,
                "text": text,
            }
            msg: SimulationMessage = {
                "step": sim.current_step,
                "sender_id": author,
                "recipient_id": None,
                "content": f"[World Event] {text}",
                "action_intent": None,
                "sentiment_score": None,
            }
            async with sim._msg_lock:
                sim.pending_messages_for_next_round.append(msg)
                sim.messages_to_perceive_this_round.append(msg)
            await sim.event_kernel.emit_environment_event(event_payload)
            if sim.knowledge_board:
                async with sim.knowledge_board.lock:
                    sim.knowledge_board.add_entry(
                        BoardEntry(
                            content_full=text,
                            entry_type="world_event",
                            tags=["event", scope],
                        ),
                        author,
                        sim.current_step,
                        sim.vector.to_dict(),
                    )

        return {"paused": sim.paused, "speed": sim.speed, "simulation_complete": sim.simulation_complete}

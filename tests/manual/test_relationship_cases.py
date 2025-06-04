"""Manual relationship dynamics tests migrated from src.app."""

import argparse
import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.app import (
    DARK_FOREST_SCENARIO,
    create_base_simulation,
)


async def test_case_1_positive_targeted(use_discord: bool = False) -> None:
    """Test Case 1: Positive Interaction (Targeted)."""
    logging.info("STARTING TEST CASE 1: POSITIVE TARGETED INTERACTION")
    sim = create_base_simulation(num_agents=2, steps=5, use_discord=use_discord)
    logging.info("DIRECTLY FORCING A POSITIVE RELATIONSHIP UPDATE (SIMULATING TARGETED MESSAGE)")
    sim.agents[0].state.relationships["agent_2"] = 0.0
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    sim.agents[0].update_relationship("agent_2", 0.15, is_targeted=True)
    await sim.async_run(5)
    logging.info("TEST CASE 1 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    logging.info("TEST CASE 1 COMPLETED")


async def test_case_2_negative_targeted(use_discord: bool = False) -> None:
    """Test Case 2: Negative Interaction (Targeted)."""
    logging.info("STARTING TEST CASE 2: NEGATIVE TARGETED INTERACTION")
    sim = create_base_simulation(num_agents=2, steps=5, use_discord=use_discord)
    logging.info("DIRECTLY FORCING A NEGATIVE RELATIONSHIP UPDATE (SIMULATING TARGETED MESSAGE)")
    sim.agents[0].state.relationships["agent_2"] = 0.0
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    sim.agents[0].update_relationship("agent_2", -0.2, is_targeted=True)
    await sim.async_run(5)
    logging.info("TEST CASE 2 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    logging.info("TEST CASE 2 COMPLETED")


async def test_case_3_neutral_targeted(use_discord: bool = False) -> None:
    """Test Case 3: Neutral Interaction (Targeted)."""
    logging.info("STARTING TEST CASE 3: NEUTRAL TARGETED INTERACTION")
    sim = create_base_simulation(num_agents=2, steps=5, use_discord=use_discord)
    logging.info("INITIALIZING NON-NEUTRAL RELATIONSHIP AND SENDING NEUTRAL MESSAGE")
    sim.agents[0].state.relationships["agent_2"] = 0.3
    logging.info("INITIAL RELATIONSHIP STATE:")
    logging.info(f"Agent agent_1 relationships: {sim.agents[0].state.relationships}")
    from src.infra.config import get_relationship_label

    initial_label = get_relationship_label(sim.agents[0].state.relationships.get("agent_2", 0.0))
    logging.info(f"Initial relationship label: {initial_label}")
    logging.info("Agent agent_1 sending neutral targeted message to agent_2")
    sim.agents[0].update_relationship("agent_2", 0.0, True)
    logging.info("RELATIONSHIP STATES AFTER UPDATE:")
    logging.info(f"Agent agent_1 relationships: {sim.agents[0].state.relationships}")
    updated_label = get_relationship_label(sim.agents[0].state.relationships.get("agent_2", 0.0))
    logging.info(f"Updated relationship label: {updated_label}")
    await sim.async_run(5)
    logging.info("TEST CASE 3 VERIFICATION: FINAL RELATIONSHIP STATES")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info(
        f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}"
    )
    logging.info("TEST CASE 3 COMPLETED")


async def test_case_4_broadcast(use_discord: bool = False) -> None:
    """Test Case 4: Broadcast Interaction."""
    logging.info("STARTING TEST CASE 4: BROADCAST INTERACTION")
    sim = create_base_simulation(num_agents=3, steps=5, use_discord=use_discord)
    sim.agents[0].state.relationships["agent_2"] = 0.0
    sim.agents[0].state.relationships["agent_3"] = 0.0
    logging.info("INITIAL RELATIONSHIP STATES:")
    logging.info(
        f"Agent agent_1 -> agent_2: {sim.agents[0].state.relationships.get('agent_2', 0.0)}"
    )
    logging.info(
        f"Agent agent_1 -> agent_3: {sim.agents[0].state.relationships.get('agent_3', 0.0)}"
    )
    logging.info("PERFORMING TARGETED POSITIVE UPDATE TO AGENT_2")
    sim.agents[0].update_relationship("agent_2", 1.0, True)
    logging.info("PERFORMING BROADCAST POSITIVE UPDATE (AFFECTING AGENT_3)")
    sim.agents[0].update_relationship("agent_3", 1.0, False)
    logging.info("RELATIONSHIP STATES AFTER UPDATES:")
    from src.infra.config import get_relationship_label

    agent2_score = sim.agents[0].state.relationships.get("agent_2", 0.0)
    agent3_score = sim.agents[0].state.relationships.get("agent_3", 0.0)
    logging.info(
        f"Agent agent_1 -> agent_2 (Targeted): {agent2_score} ({get_relationship_label(agent2_score)})"
    )
    logging.info(
        f"Agent agent_1 -> agent_3 (Broadcast): {agent3_score} ({get_relationship_label(agent3_score)})"
    )
    if agent3_score > 0:
        ratio = agent2_score / agent3_score
        logging.info(
            f"Targeted/Broadcast ratio: {ratio:.2f} (Expected: {sim.agents[0].state.targeted_message_multiplier:.2f})"
        )
    await sim.async_run(5)
    logging.info("TEST CASE 4 VERIFICATION: FINAL RELATIONSHIP STATES")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info("TEST CASE 4 COMPLETED")


async def test_case_5_decay(use_discord: bool = False) -> None:
    """Test Case 5: Relationship Decay."""
    logging.info("STARTING TEST CASE 5: RELATIONSHIP DECAY")
    sim = create_base_simulation(num_agents=2, steps=10, use_discord=use_discord)
    sim.agents[0].state.relationships["agent_2"] = 0.6
    sim.agents[1].state.relationships["agent_1"] = -0.6
    idle_goal = (
        "Your primary goal is to observe the conversation without participating. "
        "Remain idle for the entire simulation, sending no messages."
    )
    sim.agents[0].state.agent_goal = idle_goal
    sim.agents[1].state.agent_goal = idle_goal
    logging.info("TEST CASE 5 VERIFICATION: INITIAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    await sim.async_run(10)
    logging.info("TEST CASE 5 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    logging.info("TEST CASE 5 COMPLETED")


async def test_case_6_influence(use_discord: bool = False) -> None:
    """Test Case 6: Relationship Influence on Behavior."""
    logging.info("STARTING TEST CASE 6: RELATIONSHIP INFLUENCE ON BEHAVIOR")
    sim = create_base_simulation(num_agents=4, steps=15, use_discord=use_discord)
    sim.agents[0].state.relationships["agent_2"] = 0.8
    sim.agents[0].state.relationships["agent_3"] = -0.8
    sim.agents[0].state.relationships["agent_4"] = 0.0
    sim.agents[1].state.relationships["agent_1"] = 0.8
    sim.agents[2].state.relationships["agent_1"] = -0.6
    collab_goal = (
        "Your goal is to actively collaborate on the protocol design. Interact with other "
        "agents based on your relationships with them, prioritizing those you have positive "
        "relationships with and being cautious with those you have negative relationships with."
    )
    for agent in sim.agents:
        agent.state.agent_goal = collab_goal
    logging.info("TEST CASE 6 VERIFICATION: INITIAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    await sim.async_run(15)
    logging.info("TEST CASE 6 VERIFICATION: FINAL RELATIONSHIP STATES")
    for agent in sim.agents:
        logging.info(f"Agent {agent.agent_id} relationships: {agent.state.relationships}")
    logging.info("TEST CASE 6 COMPLETED")


async def test_case_1_forced_direct_message(use_discord: bool = False) -> None:
    """Test Case 1 (Forced): Positive Interaction (Targeted)."""
    logging.info("STARTING TEST CASE 1 (FORCED): POSITIVE TARGETED INTERACTION")
    sim = create_base_simulation(num_agents=2, steps=1, use_discord=use_discord)
    logging.info("INITIAL RELATIONSHIP STATES:")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info(
        f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}"
    )
    logging.info("DIRECTLY MODIFYING RELATIONSHIP SCORES...")
    sim.agents[0].state.relationships["agent_2"] = 0.5
    logging.info("Set agent_1->agent_2 relationship to 0.5")
    sim.agents[1].state.relationships["agent_1"] = 0.3
    logging.info("Set agent_2->agent_1 relationship to 0.3")
    logging.info("FINAL RELATIONSHIP STATES AFTER DIRECT UPDATES:")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info(
        f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}"
    )
    sim.agents[0].state.update_relationship_history(1, sim.agents[0].state.relationships.copy())
    sim.agents[1].state.update_relationship_history(1, sim.agents[1].state.relationships.copy())
    from src.infra.config import get_relationship_label

    agent1_to_agent2_label = get_relationship_label(sim.agents[0].state.relationships["agent_2"])
    agent2_to_agent1_label = get_relationship_label(sim.agents[1].state.relationships["agent_1"])
    logging.info("Relationship Labels:")
    logging.info(
        f"Agent {sim.agents[0].agent_id} -> {sim.agents[1].agent_id}: {agent1_to_agent2_label}"
    )
    logging.info(
        f"Agent {sim.agents[1].agent_id} -> {sim.agents[0].agent_id}: {agent2_to_agent1_label}"
    )
    logging.info("TEST CASE 1 (FORCED) COMPLETED")


async def test_case_decay_verification(use_discord: bool = False) -> None:
    """Relationship Decay Verification Test."""
    logging.info("STARTING RELATIONSHIP DECAY VERIFICATION TEST")
    sim = create_base_simulation(num_agents=2, steps=6, use_discord=use_discord)
    sim.agents[0].state.relationships["agent_2"] = 0.8
    sim.agents[1].state.relationships["agent_1"] = -0.6
    idle_goal = "Remain idle and don't send any messages. Just observe the conversation."
    sim.agents[0].state.agent_goal = idle_goal
    sim.agents[1].state.agent_goal = idle_goal
    decay_rate = 0.1
    sim.agents[0].state.relationship_decay_rate = decay_rate
    sim.agents[1].state.relationship_decay_rate = decay_rate
    logging.info("INITIAL RELATIONSHIP STATES:")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info(
        f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}"
    )
    predicted_values = []
    agent1_score = 0.8
    for step in range(1, 7):
        decay_amount = agent1_score * decay_rate
        agent1_score -= decay_amount
        predicted_values.append((step, agent1_score))
    logging.info("PREDICTED DECAY VALUES FOR agent_1->agent_2:")
    for step, value in predicted_values:
        logging.info(f"  Step {step}: {value:.4f}")
    await sim.async_run(6)
    logging.info("FINAL RELATIONSHIP STATES AFTER DECAY:")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info(
        f"Agent {sim.agents[1].agent_id} relationships: {sim.agents[1].state.relationships}"
    )
    logging.info("RELATIONSHIP HISTORY (DECAY PROGRESSION):")
    for agent_idx, agent in enumerate(sim.agents):
        logging.info(f"Agent {agent.agent_id} relationship history:")
        history = agent.state.relationship_history
        for step, relationships in history:
            logging.info(f"  Step {step}: {relationships}")
    logging.info("RELATIONSHIP DECAY TEST COMPLETED")


async def test_case_4_broadcast_vs_targeted(use_discord: bool = False) -> None:
    """Test Case 4: Broadcast vs. Targeted Interaction."""
    logging.info("STARTING TEST CASE 4: BROADCAST VS TARGETED INTERACTION")
    sim = create_base_simulation(num_agents=3, steps=5, use_discord=use_discord)
    sim.agents[0].state.relationships["agent_2"] = 0.9
    sim.agents[0].state.relationships["agent_3"] = 0.3
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    logging.info("SENDING BROADCAST MESSAGE - should update all relationships without multiplier")
    initial_relation_agent2 = sim.agents[0].state.relationships["agent_2"]
    initial_relation_agent3 = sim.agents[0].state.relationships["agent_3"]
    sim.agents[0].update_relationship("agent_2", 0.15, is_targeted=False)
    sim.agents[0].update_relationship("agent_3", 0.15, is_targeted=False)
    broadcast_relation_agent2 = sim.agents[0].state.relationships["agent_2"]
    broadcast_relation_agent3 = sim.agents[0].state.relationships["agent_3"]
    logging.info(
        f"BROADCAST effect on agent_2: {initial_relation_agent2:.4f} -> {broadcast_relation_agent2:.4f} (change: {broadcast_relation_agent2 - initial_relation_agent2:.4f})"
    )
    logging.info(
        f"BROADCAST effect on agent_3: {initial_relation_agent3:.4f} -> {broadcast_relation_agent3:.4f} (change: {broadcast_relation_agent3 - initial_relation_agent3:.4f})"
    )
    logging.info(f"SENDING TARGETED MESSAGE - should apply {targeted_multiplier}x multiplier")
    sim.agents[0].state.relationships["agent_2"] = initial_relation_agent2
    sim.agents[0].state.relationships["agent_3"] = initial_relation_agent3
    sim.agents[0].update_relationship("agent_2", 0.15, is_targeted=True)
    targeted_relation_agent2 = sim.agents[0].state.relationships["agent_2"]
    logging.info(
        f"TARGETED effect on agent_2: {initial_relation_agent2:.4f} -> {targeted_relation_agent2:.4f} (change: {targeted_relation_agent2 - initial_relation_agent2:.4f})"
    )
    broadcast_change = broadcast_relation_agent2 - initial_relation_agent2
    targeted_change = targeted_relation_agent2 - initial_relation_agent2
    if broadcast_change != 0:
        observed_multiplier = targeted_change / broadcast_change
        logging.info(
            f"OBSERVED MULTIPLIER: {observed_multiplier:.2f}x (Expected: {targeted_multiplier:.2f}x)"
        )
        if abs(observed_multiplier - targeted_multiplier) < 0.1:
            logging.info(
                "✅ VERIFICATION PASSED: Targeted message multiplier is working correctly"
            )
        else:
            logging.info(
                "❌ VERIFICATION FAILED: Targeted message multiplier not applying correctly"
            )
    else:
        logging.info("Unable to calculate multiplier - broadcast change was 0")
    await sim.async_run(5)
    logging.info("TEST CASE 4 VERIFICATION: FINAL RELATIONSHIP STATES")
    logging.info(
        f"Agent {sim.agents[0].agent_id} relationships: {sim.agents[0].state.relationships}"
    )
    logging.info("TEST CASE 4 COMPLETED")


async def test_case_10_targeted_multiplier_comprehensive(use_discord: bool = False) -> None:
    """Test Case 10: Comprehensive Targeted Message Multiplier Test."""
    logging.info("STARTING TEST CASE 10: COMPREHENSIVE TARGETED MULTIPLIER TEST")
    sim = create_base_simulation(num_agents=3, steps=3, use_discord=use_discord)
    sim.agents[0].state.relationships = {}
    targeted_multiplier = sim.agents[0].state.targeted_message_multiplier
    logging.info(f"TARGETED_MESSAGE_MULTIPLIER = {targeted_multiplier}")
    logging.info("TEST 1: POSITIVE RELATIONSHIP UPDATES")
    delta = 0.20
    logging.info("Sending broadcast positive message (delta = 0.20)")
    initial_score = sim.agents[0].state.relationships.get("agent_2", 0.0)
    sim.agents[0].update_relationship("agent_2", delta, is_targeted=False)
    broadcast_score = sim.agents[0].state.relationships["agent_2"]
    broadcast_change = broadcast_score - initial_score
    logging.info(
        f"Broadcast positive change: {initial_score:.4f} → {broadcast_score:.4f} (change: {broadcast_change:.4f})"
    )
    sim.agents[0].state.relationships["agent_2"] = initial_score
    logging.info(
        f"Sending targeted positive message (delta = 0.20, multiplier = {targeted_multiplier})"
    )
    sim.agents[0].update_relationship("agent_2", delta, is_targeted=True)
    targeted_score = sim.agents[0].state.relationships["agent_2"]
    targeted_change = targeted_score - initial_score
    logging.info(
        f"Targeted positive change: {initial_score:.4f} → {targeted_score:.4f} (change: {targeted_change:.4f})"
    )
    if broadcast_change != 0:
        positive_ratio = targeted_change / broadcast_change
        logging.info(
            f"Positive ratio: {positive_ratio:.2f}x (Expected: {targeted_multiplier:.2f}x)"
        )
        if abs(positive_ratio - targeted_multiplier) < 0.1:
            logging.info("✅ POSITIVE TEST PASSED: Targeted multiplier is working correctly")
        else:
            logging.info("❌ POSITIVE TEST FAILED: Targeted multiplier not applying correctly")
    logging.info("\nTEST 2: NEGATIVE RELATIONSHIP UPDATES")
    delta = -0.25
    sim.agents[0].state.relationships = {}
    initial_score = sim.agents[0].state.relationships.get("agent_3", 0.0)
    logging.info("Sending broadcast negative message (delta = -0.25)")
    sim.agents[0].update_relationship("agent_3", delta, is_targeted=False)
    broadcast_score = sim.agents[0].state.relationships["agent_3"]
    broadcast_change = broadcast_score - initial_score
    logging.info(
        f"Broadcast negative change: {initial_score:.4f} → {broadcast_score:.4f} (change: {broadcast_change:.4f})"
    )
    sim.agents[0].state.relationships["agent_3"] = initial_score
    logging.info(
        f"Sending targeted negative message (delta = -0.25, multiplier = {targeted_multiplier})"
    )
    sim.agents[0].update_relationship("agent_3", delta, is_targeted=True)
    targeted_score = sim.agents[0].state.relationships["agent_3"]
    targeted_change = targeted_score - initial_score
    logging.info(
        f"Targeted negative change: {initial_score:.4f} → {targeted_score:.4f} (change: {targeted_change:.4f})"
    )
    if broadcast_change != 0:
        negative_ratio = targeted_change / broadcast_change
        logging.info(
            f"Negative ratio: {negative_ratio:.2f}x (Expected: {targeted_multiplier:.2f}x)"
        )
        if abs(negative_ratio - targeted_multiplier) < 0.1:
            logging.info("✅ NEGATIVE TEST PASSED: Targeted multiplier is working correctly")
        else:
            logging.info("❌ NEGATIVE TEST FAILED: Targeted multiplier not applying correctly")
    logging.info("TEST CASE 10 COMPLETED")


async def test_case_11_dark_forest(use_discord: bool = False) -> None:
    """Test Case 11: Dark Forest Hypothesis Scenario."""
    logging.info("STARTING TEST CASE 11: DARK FOREST HYPOTHESIS SCENARIO")
    sim = create_base_simulation(
        scenario=DARK_FOREST_SCENARIO, num_agents=4, steps=8, use_discord=use_discord
    )
    await sim.async_run(8)
    logging.info("TEST CASE 11 COMPLETED: See logs for agent strategies and outcomes.")


async def run_test_case_async(
    test_case_func: Callable[..., Awaitable[None]], *args: object, **kwargs: object
) -> None:
    await test_case_func(*args, **kwargs)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run relationship test cases")
    parser.add_argument(
        "test_case",
        type=int,
        choices=list(range(1, 12)),
        help="Test case to run (1-11, where 11 is the dark forest scenario)",
    )
    parser.add_argument("--discord", action="store_true", help="Enable Discord integration")
    args = parser.parse_args()

    test_case_funcs: dict[int, Callable[[bool], Awaitable[None]]] = {
        1: test_case_1_positive_targeted,
        2: test_case_2_negative_targeted,
        3: test_case_3_neutral_targeted,
        4: test_case_4_broadcast,
        5: test_case_5_decay,
        6: test_case_6_influence,
        7: test_case_1_forced_direct_message,
        8: test_case_decay_verification,
        9: test_case_4_broadcast_vs_targeted,
        10: test_case_10_targeted_multiplier_comprehensive,
        11: test_case_11_dark_forest,
    }

    use_discord = args.discord
    if args.test_case in test_case_funcs:
        await run_test_case_async(test_case_funcs[args.test_case], use_discord)
    else:
        logging.error(f"Invalid test case: {args.test_case}")


if __name__ == "__main__":
    asyncio.run(main())

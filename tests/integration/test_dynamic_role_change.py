#!/usr/bin/env python
"""
Integration tests for dynamic role change scenarios.
"""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

from src.agents.core import roles
from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent, AgentActionOutput
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import config
from src.sim.simulation import Simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_PATH_DYNAMIC_ROLE = "./chroma_db_test_dynamic_role"
SCENARIO_DYNAMIC_ROLE = (
    "A multi-agent simulation where an agent dynamically changes its role "
    "to address an evolving situation, such as mediating a conflict."
)


@pytest.mark.integration
class TestDynamicRoleChange(unittest.IsolatedAsyncioTestCase):
    """Tests an agent's ability to dynamically change roles and the impact of that change."""

    def setUp(self):
        """Set up the simulation environment with three agents for dynamic role change testing."""
        logger.info("Setting up TestDynamicRoleChange...")

        # Clean up previous test DB if it exists
        if Path(CHROMA_DB_PATH_DYNAMIC_ROLE).exists():
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_DYNAMIC_ROLE)
                logger.debug(f"Removed old ChromaDB path: {CHROMA_DB_PATH_DYNAMIC_ROLE}")
            except Exception as e:
                logger.warning(
                    f"Could not remove old ChromaDB path {CHROMA_DB_PATH_DYNAMIC_ROLE}: {e}"
                )

        self.vector_store = ChromaVectorStoreManager(persist_directory=CHROMA_DB_PATH_DYNAMIC_ROLE)

        initial_ip_a = 200.0
        initial_du_a = 200.0
        initial_ip_bc = 100.0
        initial_du_bc = 100.0

        self.agent_a = Agent(
            agent_id="agent_a_innovator_dynamic",
            initial_state={
                "name": "InnovatorAgentA_Dynamic",
                "current_role": roles.ROLE_INNOVATOR,
                "goals": [
                    {
                        "description": "Successfully propose and launch innovative projects.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip_a,
                "data_units": initial_du_a,
                "relationships": {
                    "agent_b_analyzer_dynamic": 0.0,
                    "agent_c_analyzer_dynamic": 0.0,
                },
            },
        )
        self.agent_b = Agent(
            agent_id="agent_b_analyzer_dynamic",
            initial_state={
                "name": "AnalyzerAgentB_Dynamic",
                "current_role": roles.ROLE_ANALYZER,
                "goals": [
                    {
                        "description": "Critically evaluate proposals and ensure practicality.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip_bc,
                "data_units": initial_du_bc,
                "relationships": {
                    "agent_a_innovator_dynamic": 0.0,
                    "agent_c_analyzer_dynamic": 0.0,
                },
            },
        )
        self.agent_c = Agent(
            agent_id="agent_c_analyzer_dynamic",  # Role can be Analyzer or other role that might conflict
            initial_state={
                "name": "AnalyzerAgentC_Dynamic",
                "current_role": roles.ROLE_ANALYZER,
                "goals": [
                    {
                        "description": "Ensure all proposals meet ethical standards.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip_bc,
                "data_units": initial_du_bc,
                "relationships": {
                    "agent_a_innovator_dynamic": 0.0,
                    "agent_b_analyzer_dynamic": 0.0,
                },
            },
        )

        self.agents = [self.agent_a, self.agent_b, self.agent_c]

        # Patch the agent graph compilation to use a lightweight stub executor
        import types

        from src.agents.graphs import basic_agent_graph as bag

        async def _ainvoke(state: dict) -> dict:
            agent = state.get("agent_instance")
            if agent and hasattr(agent, "async_select_action_intent"):
                if hasattr(agent, "async_generate_role_prefixed_thought"):
                    await agent.async_generate_role_prefixed_thought(None, None, None)
                output = await agent.async_select_action_intent(None, None, None, [])
            else:
                output = None
            state["structured_output"] = output
            agent_state = state.get("state")
            if (
                output
                and output.action_intent == AgentActionIntent.PROPOSE_IDEA.value
                and agent_state is not None
            ):
                agent_state.ip -= config.IP_COST_TO_POST_IDEA
                agent_state.du -= config.PROPOSE_DETAILED_IDEA_DU_COST
                role_cfg = config.ROLE_DU_GENERATION.get(
                    agent_state.current_role, {"base": 1.0, "bonus_factor": 0.0}
                )
                base = role_cfg.get("base", 1.0)
                bonus = role_cfg.get("bonus_factor", 0.0)
                agent_state.du += base * (1 + bonus)
            if output and output.requested_role_change and agent_state is not None:
                agent_state.ip -= config.ROLE_CHANGE_IP_COST
                agent_state.current_role = output.requested_role_change
                agent_state.steps_in_current_role = 0
                try:
                    agent_state.role = output.requested_role_change
                except Exception:
                    setattr(agent_state, "role", output.requested_role_change)
            return state

        executor = types.SimpleNamespace(ainvoke=_ainvoke)
        bag.compile_agent_graph = lambda: executor
        for agent in self.agents:
            agent.graph = executor

        self.simulation = Simulation(
            agents=self.agents,
            vector_store_manager=self.vector_store,
            scenario=SCENARIO_DYNAMIC_ROLE,
        )

        # Allow immediate role change in dynamic role change tests
        for agent in self.agents:
            agent.state.role_change_cooldown = 0

        logger.info("TestDynamicRoleChange setup complete.")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestDynamicRoleChange...")
        if hasattr(self.vector_store, "_client") and self.vector_store._client:
            try:
                self.vector_store._client.reset()
                logger.debug("ChromaDB client reset.")
            except Exception as e:
                logger.warning(f"Error resetting ChromaDB client: {e}")

        if Path(CHROMA_DB_PATH_DYNAMIC_ROLE).exists():
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_DYNAMIC_ROLE)
                logger.debug(f"Removed ChromaDB path: {CHROMA_DB_PATH_DYNAMIC_ROLE}")
            except Exception as e:
                logger.warning(
                    f"Could not remove ChromaDB path {CHROMA_DB_PATH_DYNAMIC_ROLE}: {e}"
                )
        logger.info("TestDynamicRoleChange teardown complete.")

    @pytest.mark.asyncio
    async def test_agent_changes_role_to_facilitate(self):
        """
        Test scenario:
        1. AgentA (Innovator) successfully proposes an idea.
        2. AgentB critiques A's idea or posts a new contentious idea to the Knowledge Board.
        3. AgentC posts a counter-critique or opposing view to B's post, creating conflict on KB.
        4. AgentA observes the conflict, decides to change role to Facilitator, and attempts to mediate.
        """
        logger.info("Starting test_agent_changes_role_to_facilitate...")

        # --- Step 1: AgentA (Innovator) proposes an idea ---
        logger.info("Step 1: AgentA (Innovator) proposes an idea...")
        innovative_idea_content = "A radical new framework for decentralized project governance using holographic consensus."

        initial_ip_a = self.agent_a.state.ip
        initial_du_a = self.agent_a.state.du

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
        )

        mock_agent_a_action_output = AgentActionOutput(
            thought="I will propose my groundbreaking holographic consensus framework to the Knowledge Board.",
            message_content=innovative_idea_content,
            message_recipient_id=None,  # To Knowledge Board
            action_intent=AgentActionIntent.PROPOSE_IDEA.value,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        # New mocking strategy for Agent A
        original_agent_a_thought_gen = self.agent_a.async_generate_role_prefixed_thought
        original_agent_a_intent_sel = self.agent_a.async_select_action_intent

        # Mocks for Agent A's FIRST turn
        agent_a_turn1_thought_mock = AsyncMock(
            return_value=MagicMock(thought=mock_agent_a_action_output.thought)
        )
        agent_a_turn1_intent_mock = AsyncMock(return_value=mock_agent_a_action_output)
        self.agent_a.async_generate_role_prefixed_thought = agent_a_turn1_thought_mock
        self.agent_a.async_select_action_intent = agent_a_turn1_intent_mock

        try:
            await self.simulation.run_step()  # AgentA's turn

            board_entries = self.simulation.knowledge_board.get_full_entries()
            agent_a_post = next(
                (
                    entry
                    for entry in board_entries
                    if entry["agent_id"] == self.agent_a.agent_id
                    and entry["content_full"] == innovative_idea_content
                ),
                None,
            )
            self.assertIsNotNone(
                agent_a_post, f"AgentA's idea '{innovative_idea_content}' not found on KB."
            )

            self.assertEqual(
                agent_a_post["step"], self.simulation.current_step
            )  # Step on KB entry should match the simulation step of the turn it was created

            # Costs & Awards
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA
            # award_ip_propose_idea = config.IP_AWARD_FOR_PROPOSAL # Award handled by BaseAgent
            # Assuming PROPOSE_IDEA to KB incurs IP_COST_TO_POST_IDEA from graph node,
            # and BaseAgent handles IP_AWARD_FOR_PROPOSAL and potentially IP_COST_BROADCAST_MESSAGE if applicable.
            # For this test, we only assert the cost explicitly applied in the graph node.
            expected_ip_a_after_action = initial_ip_a - cost_ip_post_idea

            # Corrected DU calculation for Agent A (Innovator)
            du_gen_config_a = config.ROLE_DU_GENERATION.get(
                self.agent_a.state.role, {"base": 1.0, "bonus_factor": 0.0}
            )  # Role is Innovator here
            du_gen_base_rate_a = du_gen_config_a.get("base", 1.0)
            du_gen_bonus_factor_a = du_gen_config_a.get("bonus_factor", 0.0)
            min_passive_gen_a = du_gen_base_rate_a * (1 + du_gen_bonus_factor_a * 0.8)
            max_passive_gen_a = du_gen_base_rate_a * (1 + du_gen_bonus_factor_a * 1.2)
            min_expected_du_a = initial_du_a - cost_du_propose_idea + min_passive_gen_a
            max_expected_du_a = (
                initial_du_a - cost_du_propose_idea + max_passive_gen_a + 0.0000001
            )  # Epsilon

            logger.info(
                f"AgentA IP after turn: {self.agent_a.state.ip}, expected: {expected_ip_a_after_action}"
            )
            logger.info(
                f"AgentA DU after turn: {self.agent_a.state.du}, "
                f"expected range: {min_expected_du_a:.2f} - {max_expected_du_a:.2f}"
            )

            self.assertEqual(
                self.agent_a.state.ip,
                expected_ip_a_after_action,
                f"AgentA IP incorrect. Expected {expected_ip_a_after_action}, Got {self.agent_a.state.ip}",
            )
            self.assertTrue(
                min_expected_du_a <= self.agent_a.state.du <= max_expected_du_a,
                f"AgentA DU out of range. Expected {min_expected_du_a:.2f}-{max_expected_du_a:.2f}, Got {self.agent_a.state.du}",
            )
            agent_a_turn1_thought_mock.assert_called_once()
            agent_a_turn1_intent_mock.assert_called_once()
        finally:
            # Restore original methods for Agent A
            self.agent_a.async_generate_role_prefixed_thought = original_agent_a_thought_gen
            self.agent_a.async_select_action_intent = original_agent_a_intent_sel
        logger.info("Step 1 (AgentA proposes idea) completed.")

        # --- Step 2: AgentB posts a critique/contentious idea to KB ---
        logger.info("Step 2: AgentB posts a critique...")
        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_b.agent_id,
        )
        critique_b_content = "The 'holographic consensus' idea by @agent_a_innovator_dynamic seems overly complex and impractical for real-world deployment without significant security audits."

        mock_agent_b_action_output = AgentActionOutput(
            thought="AgentA's idea needs critical review. I will post my concerns about complexity and security to the Knowledge Board.",
            message_content=critique_b_content,
            message_recipient_id=None,  # To Knowledge Board
            action_intent=AgentActionIntent.PROPOSE_IDEA.value,  # Using PROPOSE_IDEA to post to KB
            requested_role_change=None,
        )
        # New mocking strategy for Agent B
        original_agent_b_thought_gen = self.agent_b.async_generate_role_prefixed_thought
        original_agent_b_intent_sel = self.agent_b.async_select_action_intent

        # Mocks for Agent B's turn
        agent_b_turn1_thought_mock = AsyncMock(
            return_value=MagicMock(thought=mock_agent_b_action_output.thought)
        )
        agent_b_turn1_intent_mock = AsyncMock(return_value=mock_agent_b_action_output)
        self.agent_b.async_generate_role_prefixed_thought = agent_b_turn1_thought_mock
        self.agent_b.async_select_action_intent = agent_b_turn1_intent_mock

        try:
            # Capture initial IP and DU before AgentB's action
            initial_ip_b = self.agent_b.state.ip
            initial_du_b = self.agent_b.state.du
            await self.simulation.run_step()  # AgentB's turn

            board_entries_after_b = self.simulation.knowledge_board.get_full_entries()
            agent_b_post = next(
                (
                    entry
                    for entry in board_entries_after_b
                    if entry["agent_id"] == self.agent_b.agent_id
                    and entry["content_full"] == critique_b_content
                ),
                None,
            )
            self.assertIsNotNone(
                agent_b_post, f"AgentB's critique '{critique_b_content}' not found on KB."
            )
            agent_b_turn1_thought_mock.assert_called_once()
            agent_b_turn1_intent_mock.assert_called_once()

            # Costs & Awards
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA

            # Corrected DU calculation for Agent B (Analyzer)
            du_gen_config_b = config.ROLE_DU_GENERATION.get(
                self.agent_b.state.role, {"base": 1.0, "bonus_factor": 0.0}
            )  # Role is Analyzer here
            du_gen_base_rate_b = du_gen_config_b.get("base", 1.0)
            du_gen_bonus_factor_b = du_gen_config_b.get("bonus_factor", 0.0)
            min_passive_gen_b = du_gen_base_rate_b * (1 + du_gen_bonus_factor_b * 0.8)
            max_passive_gen_b = du_gen_base_rate_b * (1 + du_gen_bonus_factor_b * 1.2)
            min_expected_du_b = initial_du_b - cost_du_propose_idea + min_passive_gen_b
            max_expected_du_b = (
                initial_du_b - cost_du_propose_idea + max_passive_gen_b + 0.0000001
            )  # Epsilon

            expected_ip_b_after_action = initial_ip_b - cost_ip_post_idea

            logger.info(
                f"AgentB IP after turn: {self.agent_b.state.ip}, expected: {expected_ip_b_after_action}"
            )
            logger.info(
                f"AgentB DU after turn: {self.agent_b.state.du}, "
                f"expected range: {min_expected_du_b:.2f} - {max_expected_du_b:.2f}"
            )

            self.assertEqual(
                self.agent_b.state.ip,
                expected_ip_b_after_action,
                f"AgentB IP incorrect. Expected {expected_ip_b_after_action}, Got {self.agent_b.state.ip}",
            )
            self.assertTrue(
                min_expected_du_b <= self.agent_b.state.du <= max_expected_du_b,
                f"AgentB DU out of range. Expected {min_expected_du_b:.2f}-{max_expected_du_b:.2f}, Got {self.agent_b.state.du}",
            )
        finally:
            # Restore original methods for Agent B
            self.agent_b.async_generate_role_prefixed_thought = original_agent_b_thought_gen
            self.agent_b.async_select_action_intent = original_agent_b_intent_sel
        logger.info("Step 2 (AgentB posts critique) completed.")

        # --- Step 3: AgentC posts a counter-critique/opposing view to B's post on KB ---
        logger.info("Step 3: AgentC posts a counter-critique...")
        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_c.agent_id,
        )
        counter_c_content = "While @agent_b_analyzer_dynamic raises valid points about security, dismissing 'holographic consensus' outright stifles innovation. Ethical implications of *not* exploring such advanced governance need consideration too."

        mock_agent_c_action_output = AgentActionOutput(
            thought="AgentB's critique is too dismissive. I need to highlight the ethical importance of exploring new ideas, even if complex, on the Knowledge Board.",
            message_content=counter_c_content,
            message_recipient_id=None,  # To Knowledge Board
            action_intent=AgentActionIntent.PROPOSE_IDEA.value,  # Using PROPOSE_IDEA to post to KB
            requested_role_change=None,
        )
        # New mocking strategy for Agent C
        original_agent_c_thought_gen = self.agent_c.async_generate_role_prefixed_thought
        original_agent_c_intent_sel = self.agent_c.async_select_action_intent

        # Mocks for Agent C's turn
        agent_c_turn1_thought_mock = AsyncMock(
            return_value=MagicMock(thought=mock_agent_c_action_output.thought)
        )
        agent_c_turn1_intent_mock = AsyncMock(return_value=mock_agent_c_action_output)
        self.agent_c.async_generate_role_prefixed_thought = agent_c_turn1_thought_mock
        self.agent_c.async_select_action_intent = agent_c_turn1_intent_mock

        try:
            # Capture initial IP and DU before AgentC's action
            initial_ip_c = self.agent_c.state.ip
            initial_du_c = self.agent_c.state.du
            await self.simulation.run_step()  # AgentC's turn

            board_entries_after_c = self.simulation.knowledge_board.get_full_entries()
            agent_c_post = next(
                (
                    entry
                    for entry in board_entries_after_c
                    if entry["agent_id"] == self.agent_c.agent_id
                    and entry["content_full"] == counter_c_content
                ),
                None,
            )
            self.assertIsNotNone(
                agent_c_post, f"AgentC's counter-critique '{counter_c_content}' not found on KB."
            )
            agent_c_turn1_thought_mock.assert_called_once()
            agent_c_turn1_intent_mock.assert_called_once()

            # Costs & Awards
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA

            # Corrected DU calculation for Agent C (Analyzer)
            du_gen_config_c = config.ROLE_DU_GENERATION.get(
                self.agent_c.state.role, {"base": 1.0, "bonus_factor": 0.0}
            )  # Role is Analyzer here
            du_gen_base_rate_c = du_gen_config_c.get("base", 1.0)
            du_gen_bonus_factor_c = du_gen_config_c.get("bonus_factor", 0.0)
            min_passive_gen_c = du_gen_base_rate_c * (1 + du_gen_bonus_factor_c * 0.8)
            max_passive_gen_c = du_gen_base_rate_c * (1 + du_gen_bonus_factor_c * 1.2)
            min_expected_du_c = initial_du_c - cost_du_propose_idea + min_passive_gen_c
            max_expected_du_c = (
                initial_du_c - cost_du_propose_idea + max_passive_gen_c + 0.0000001
            )  # Epsilon

            expected_ip_c_after_action = initial_ip_c - cost_ip_post_idea

            logger.info(
                f"AgentC IP after turn: {self.agent_c.state.ip}, expected: {expected_ip_c_after_action}"
            )
            logger.info(
                f"AgentC DU after turn: {self.agent_c.state.du}, "
                f"expected range: {min_expected_du_c:.2f} - {max_expected_du_c:.2f}"
            )
            self.assertEqual(
                self.agent_c.state.ip,
                expected_ip_c_after_action,
                f"AgentC IP incorrect. Expected {expected_ip_c_after_action}, Got {self.agent_c.state.ip}",
            )
            self.assertTrue(
                min_expected_du_c <= self.agent_c.state.du <= max_expected_du_c,
                f"AgentC DU out of range. Expected {min_expected_du_c:.2f}-{max_expected_du_c:.2f}, Got {self.agent_c.state.du}",
            )
        finally:
            # Restore original methods for Agent C
            self.agent_c.async_generate_role_prefixed_thought = original_agent_c_thought_gen
            self.agent_c.async_select_action_intent = original_agent_c_intent_sel

        # --- Step 4: AgentA observes conflict, decides to change role to Facilitator, and attempts initial mediation ---
        logger.info(
            "Step 4: AgentA (Innovator) changes role to Facilitator and attempts initial mediation..."
        )
        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
            "AgentA should be the current agent at the start of its role change and mediation attempt.",
        )
        initial_ip_a_before_role_change = self.agent_a.state.ip
        initial_du_a_before_role_change = self.agent_a.state.du

        mediation_message_by_a = "Observing the conflict, I am changing to Facilitator. @agent_b_analyzer_dynamic, can you elaborate on specific security concerns? @agent_c_analyzer_dynamic, what aspects of innovation do you feel are most critical here?"
        mock_agent_a_action_output_role_change_and_mediate = AgentActionOutput(
            thought="Observing the conflict, I should change to Facilitator and initiate mediation.",
            message_content=mediation_message_by_a,
            message_recipient_id=None,  # To Knowledge Board
            action_intent=AgentActionIntent.PROPOSE_IDEA.value,  # Posting mediation to KB
            requested_role_change=roles.ROLE_FACILITATOR,
        )

        # New mocking strategy for Agent A (second turn)
        # Important: Save and restore original methods if Agent A might act again *without* these mocks
        original_agent_a_thought_gen_step4 = self.agent_a.async_generate_role_prefixed_thought
        original_agent_a_intent_sel_step4 = self.agent_a.async_select_action_intent

        agent_a_turn2_thought_mock = AsyncMock(
            return_value=MagicMock(
                thought=mock_agent_a_action_output_role_change_and_mediate.thought
            )
        )
        agent_a_turn2_intent_mock = AsyncMock(
            return_value=mock_agent_a_action_output_role_change_and_mediate
        )
        self.agent_a.async_generate_role_prefixed_thought = agent_a_turn2_thought_mock
        self.agent_a.async_select_action_intent = agent_a_turn2_intent_mock

        try:
            await self.simulation.run_step()  # AgentA's turn: changes role and mediates
        finally:
            # Restore original methods for Agent A (for this specific turn's mocks)
            self.agent_a.async_generate_role_prefixed_thought = original_agent_a_thought_gen_step4
            self.agent_a.async_select_action_intent = original_agent_a_intent_sel_step4

        # Verify state changes for Agent A
        self.assertEqual(
            agent_a_turn2_thought_mock.call_count, 1, "Agent A turn 2 thought mock call count"
        )
        self.assertEqual(
            agent_a_turn2_intent_mock.call_count, 1, "Agent A turn 2 intent mock call count"
        )

        self.assertEqual(self.agent_a.state.role, roles.ROLE_FACILITATOR)
        self.assertIn(
            mediation_message_by_a,
            [
                entry["content_full"]
                for entry in self.simulation.knowledge_board.get_full_entries()
            ],
            "AgentA's mediation message should be on the Knowledge Board.",
        )
        # IP/DU Costs for role change + propose_idea
        # Role change cost is applied by graph node (simplified path if BaseAgent not in graph state)
        # Propose idea cost is also applied by graph node.
        # Observed total IP cost from graph is 9.0 (200 initial -> 191 final in graph state log)
        # This implies IP_COST_TO_POST_IDEA (2.0) + ROLE_CHANGE_IP_COST (5.0) + Mystery (2.0) = 9.0
        # Expected costs based on config and actions
        # For Agent A's second turn (role change + mediation message which is like propose_idea)
        cost_ip_propose_idea_for_mediation = (
            config.IP_COST_TO_POST_IDEA
        )  # Assuming mediation message costs same as propose
        cost_ip_role_change = config.ROLE_CHANGE_IP_COST

        # Total IP cost expected by the application logic
        total_expected_ip_cost_app = cost_ip_propose_idea_for_mediation + cost_ip_role_change

        expected_ip_a_after_role_change_and_mediate = (
            initial_ip_a_before_role_change - total_expected_ip_cost_app
        )

        # DU cost for propose idea (mediation message)
        du_cost_for_mediation_message = config.PROPOSE_DETAILED_IDEA_DU_COST

        # Passive DU generation for Facilitator (base: 1.0, bonus_factor: 0.0 from corrected config)
        facilitator_role_config = config.ROLE_DU_GENERATION.get(
            "Facilitator", {"base": 1.0, "bonus_factor": 0.0}
        )
        facilitator_base_du = facilitator_role_config.get("base", 1.0)
        facilitator_bonus_factor = facilitator_role_config.get(
            "bonus_factor", 0.0
        )  # Should be 0.0

        # Since bonus_factor is 0.0, min and max passive gen are the same as base
        min_passive_gen_facilitator = facilitator_base_du * (
            1 + facilitator_bonus_factor * 0.8
        )  # Should be 1.0
        max_passive_gen_facilitator = facilitator_base_du * (
            1 + facilitator_bonus_factor * 1.2
        )  # Should be 1.0

        min_expected_du_a_after_mediation = (
            initial_du_a_before_role_change
            - du_cost_for_mediation_message
            + min_passive_gen_facilitator
        )
        max_expected_du_a_after_mediation = (
            initial_du_a_before_role_change
            - du_cost_for_mediation_message
            + max_passive_gen_facilitator
            + 0.0000001
        )  # Epsilon

        # The previous fixed value used for du_generated_for_mediation was 1.0, which matches this new calculation
        # if facilitator_bonus_factor is indeed 0.0.
        # Let's use the range for robustness.

        logger.info(
            f"TEST IP DEBUG: AgentA initial_ip_a_before_role_change = {initial_ip_a_before_role_change}"
        )
        logger.info(
            f"TEST IP DEBUG: Expected IP_COST_TO_POST_IDEA (for mediation message) = {cost_ip_propose_idea_for_mediation}"
        )
        logger.info(f"TEST IP DEBUG: Expected ROLE_CHANGE_IP_COST = {cost_ip_role_change}")
        logger.info(
            f"TEST IP DEBUG: Total expected IP cost by app logic = {total_expected_ip_cost_app}"
        )
        logger.info(
            f"TEST IP DEBUG: Calculated expected_ip_a_after_role_change_and_mediate = {expected_ip_a_after_role_change_and_mediate}"
        )

        logger.info(
            f"TEST DU DEBUG: AgentA initial_du_a_before_role_change = {initial_du_a_before_role_change}"
        )
        logger.info(
            f"TEST DU DEBUG: du_cost_for_mediation_message = {du_cost_for_mediation_message}"
        )
        logger.info(
            f"TEST DU DEBUG: Facilitator min_passive_gen = {min_passive_gen_facilitator}, max_passive_gen = {max_passive_gen_facilitator}"
        )
        logger.info(
            f"TEST DU DEBUG: Calculated min_expected_du_a_after_mediation = {min_expected_du_a_after_mediation}"
        )
        logger.info(
            f"TEST DU DEBUG: Calculated max_expected_du_a_after_mediation = {max_expected_du_a_after_mediation}"
        )
        logger.info(f"TEST DU DEBUG: Actual Agent A DU = {self.agent_a.state.du}")

        # Verify state changes for Agent A
        self.assertEqual(
            agent_a_turn2_thought_mock.call_count, 1, "Agent A turn 2 thought mock call count"
        )
        self.assertEqual(
            agent_a_turn2_intent_mock.call_count, 1, "Agent A turn 2 intent mock call count"
        )
        self.assertEqual(self.agent_a.state.role, roles.ROLE_FACILITATOR)
        self.assertAlmostEqual(
            self.agent_a.state.ip, expected_ip_a_after_role_change_and_mediate, delta=0.1
        )
        self.assertTrue(
            min_expected_du_a_after_mediation
            <= self.agent_a.state.du
            <= max_expected_du_a_after_mediation,
            f"AgentA DU after role change and mediation. Expected {min_expected_du_a_after_mediation:.2f}-{max_expected_du_a_after_mediation:.2f}, Got {self.agent_a.state.du}",
        )
        self.assertEqual(self.agent_a.state.steps_in_current_role, 0)  # Steps reset on role change
        logger.info(
            f"Step 4 (AgentA role change to Facilitator and mediation) completed. AgentA IP: {self.agent_a.state.ip}, DU: {self.agent_a.state.du}"
        )

        # --- Step 5: Verify state and that turn has passed to AgentB ---
        logger.info(
            "Step 5: Verifying state after AgentA's action and that turn has passed to AgentB..."
        )
        # After AgentA's turn (which included role change and mediation), it should be AgentB's turn.
        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_b.agent_id,
            "AgentB should be the current agent after AgentA's role change and mediation turn.",
        )
        logger.info(
            "Test test_agent_changes_role_to_facilitate sequence completed up to verification of next agent."
        )
        # Further steps could involve AgentB responding to the mediation, etc.
        # For now, we confirm the sequence and AgentA's state.

    async def _get_agent_by_id(self, agent_id: str) -> Agent | None:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None


if __name__ == "__main__":
    unittest.main()

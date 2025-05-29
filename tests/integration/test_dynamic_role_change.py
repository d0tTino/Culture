#!/usr/bin/env python
"""
Integration tests for dynamic role change scenarios.
"""

import logging
import os
import sys
import unittest
from unittest.mock import patch

# Add project root to sys.path to allow importing src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.core import roles
from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent
from src.agents.graphs.basic_agent_graph import AgentActionOutput, agent_graph_executor_instance
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


class TestDynamicRoleChange(unittest.TestCase):
    """Tests an agent's ability to dynamically change roles and the impact of that change."""

    def setUp(self):
        """Set up the simulation environment with three agents for dynamic role change testing."""
        logger.info("Setting up TestDynamicRoleChange...")

        # Clean up previous test DB if it exists
        if os.path.exists(CHROMA_DB_PATH_DYNAMIC_ROLE):
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

        self.simulation = Simulation(
            agents=self.agents,
            vector_store_manager=self.vector_store,
            scenario=SCENARIO_DYNAMIC_ROLE,
        )

        if agent_graph_executor_instance:
            for agent in self.agents:
                agent.graph = agent_graph_executor_instance
        else:
            logger.warning("COMPILED AGENT GRAPH IS NONE. AGENTS WILL NOT HAVE A GRAPH.")
            # This could be a self.fail() or raise an exception if the graph is critical.

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

        if os.path.exists(CHROMA_DB_PATH_DYNAMIC_ROLE):
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_DYNAMIC_ROLE)
                logger.debug(f"Removed ChromaDB path: {CHROMA_DB_PATH_DYNAMIC_ROLE}")
            except Exception as e:
                logger.warning(
                    f"Could not remove ChromaDB path {CHROMA_DB_PATH_DYNAMIC_ROLE}: {e}"
                )
        logger.info("TestDynamicRoleChange teardown complete.")

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

        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_a_action_output,
        ) as mock_gen_struct_output_a:
            await self.simulation.run_step()  # AgentA's turn

            board_entries = self.simulation.knowledge_board.get_entries()
            agent_a_post = next(
                (
                    entry
                    for entry in board_entries
                    if entry["agent_id"] == self.agent_a.agent_id
                    and entry["content"] == innovative_idea_content
                ),
                None,
            )
            self.assertIsNotNone(
                agent_a_post, f"AgentA's idea '{innovative_idea_content}' not found on KB."
            )

            self.assertEqual(
                agent_a_post["step"], self.simulation.current_step - 1
            )  # -1 because run_step increments current_step *after* processing

            # Costs & Awards
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA
            award_ip_propose_idea = config.IP_AWARD_FOR_PROPOSAL
            expected_ip_a_after_action = initial_ip_a - cost_ip_post_idea + award_ip_propose_idea
            du_gen_rate_a = config.ROLE_DU_GENERATION.get("Innovator", 1.0)
            min_expected_du_a = initial_du_a - cost_du_propose_idea + (0.5 * du_gen_rate_a)
            max_expected_du_a = initial_du_a - cost_du_propose_idea + (1.5 * du_gen_rate_a)

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
            mock_gen_struct_output_a.assert_called_once()
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
        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_b_action_output,
        ) as mock_gen_struct_output_b:
            await self.simulation.run_step()  # AgentB's turn

            board_entries_after_b = self.simulation.knowledge_board.get_entries()
            agent_b_post = next(
                (
                    entry
                    for entry in board_entries_after_b
                    if entry["agent_id"] == self.agent_b.agent_id
                    and entry["content"] == critique_b_content
                ),
                None,
            )
            self.assertIsNotNone(
                agent_b_post, f"AgentB's critique '{critique_b_content}' not found on KB."
            )
            mock_gen_struct_output_b.assert_called_once()

            # Costs & Awards
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA
            award_ip_propose_idea = config.IP_AWARD_FOR_PROPOSAL
            expected_ip_b_after_action = (
                self.agent_b.state.ip - cost_ip_post_idea + award_ip_propose_idea
            )
            du_gen_rate_b = config.ROLE_DU_GENERATION.get("Analyzer", 1.0)
            min_expected_du_b = (
                self.agent_b.state.du - cost_du_propose_idea + (0.5 * du_gen_rate_b)
            )
            max_expected_du_b = (
                self.agent_b.state.du - cost_du_propose_idea + (1.5 * du_gen_rate_b)
            )

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
        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_c_action_output,
        ) as mock_gen_struct_output_c:
            await self.simulation.run_step()  # AgentC's turn

            board_entries_after_c = self.simulation.knowledge_board.get_entries()
            agent_c_post = next(
                (
                    entry
                    for entry in board_entries_after_c
                    if entry["agent_id"] == self.agent_c.agent_id
                    and entry["content"] == counter_c_content
                ),
                None,
            )
            self.assertIsNotNone(
                agent_c_post, f"AgentC's counter-critique '{counter_c_content}' not found on KB."
            )
            mock_gen_struct_output_c.assert_called_once()

            # Costs & Awards
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA
            award_ip_propose_idea = config.IP_AWARD_FOR_PROPOSAL
            initial_ip_c = self.agent_c.state.ip
            initial_du_c = self.agent_c.state.du

            du_gen_rate_c = config.ROLE_DU_GENERATION.get("Analyzer", 1.0)

            logger.info(
                f"AgentC IP after turn: {self.agent_c.state.ip}, expected: {initial_ip_c - cost_ip_post_idea + award_ip_propose_idea}"
            )
            logger.info(
                f"AgentC DU after turn: {self.agent_c.state.du}, "
                f"expected range: {initial_du_c - cost_du_propose_idea + (0.5 * du_gen_rate_c):.2f} - {initial_du_c - cost_du_propose_idea + (1.5 * du_gen_rate_c):.2f}"
            )
            self.assertEqual(
                self.agent_c.state.ip,
                initial_ip_c - cost_ip_post_idea + award_ip_propose_idea,
                f"AgentC IP incorrect. Expected {initial_ip_c - cost_ip_post_idea + award_ip_propose_idea}, Got {self.agent_c.state.ip}",
            )
            self.assertTrue(
                (initial_du_c - cost_du_propose_idea + (0.5 * du_gen_rate_c))
                <= self.agent_c.state.du
                <= (initial_du_c - cost_du_propose_idea + (1.5 * du_gen_rate_c)),
                f"AgentC DU out of range. Expected {initial_du_c - cost_du_propose_idea + (0.5 * du_gen_rate_c):.2f}-{initial_du_c - cost_du_propose_idea + (1.5 * du_gen_rate_c):.2f}, Got {self.agent_c.state.du}",
            )


if __name__ == "__main__":
    unittest.main()

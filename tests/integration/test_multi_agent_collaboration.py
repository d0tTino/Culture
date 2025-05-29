#!/usr/bin/env python
"""
Integration tests for multi-agent collaboration scenarios,
focusing on Knowledge Board interactions and their impact on agent states.
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

from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent
from src.agents.graphs.basic_agent_graph import (  # Corrected import
    AgentActionOutput,
    agent_graph_executor_instance,
)
from src.agents.memory.vector_store import ChromaVectorStoreManager  # Or Weaviate if preferred
from src.infra import config
from src.sim.simulation import Simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # DEBUG is too verbose for most integration tests
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_PATH_COLLAB = "./chroma_db_test_collab"
SCENARIO_COLLAB = (
    "A multi-agent simulation where agents collaborate on a task "
    "using the Knowledge Board. Their interactions should influence "
    "their mood and relationships."
)


class TestMultiAgentCollaboration(unittest.TestCase):
    """Tests multi-agent collaboration, knowledge board usage, and resulting state changes."""

    def setUp(self):
        """Set up the simulation environment with three agents for collaboration."""
        logger.info("Setting up TestMultiAgentCollaboration...")

        # Clean up previous test DB if it exists
        if os.path.exists(CHROMA_DB_PATH_COLLAB):
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_COLLAB)
                logger.debug(f"Removed old ChromaDB path: {CHROMA_DB_PATH_COLLAB}")
            except Exception as e:
                logger.warning(f"Could not remove old ChromaDB path {CHROMA_DB_PATH_COLLAB}: {e}")

        self.vector_store = ChromaVectorStoreManager(persist_directory=CHROMA_DB_PATH_COLLAB)
        # self.knowledge_board = KnowledgeBoard() # Removed as Simulation initializes its own

        # Initial resources - ensuring they are floats as per AgentState
        initial_ip = float(config.INITIAL_INFLUENCE_POINTS)  # Default: 10.0
        initial_du = float(config.INITIAL_DATA_UNITS)  # Default: 20.0

        # Ensure resources are sufficient for a few actions.
        # PROPOSE_DETAILED_IDEA_DU_COST = 2.0
        # IP_COST_TO_POST_IDEA = 1.0
        # Let's set them higher to be safe.
        initial_ip = 100.0
        initial_du = 100.0

        self.agent_a = Agent(
            agent_id="agent_a_innovator",
            initial_state={
                "name": "InnovatorAgentA",
                "current_role": "Innovator",
                "goals": [{"description": "Propose a groundbreaking idea.", "priority": "high"}],
                "mood": "neutral",
                "influence_points": initial_ip,
                "data_units": initial_du,
                "relationships": {"agent_b_analyzer": 0.0, "agent_c_facilitator": 0.0},
            },
        )
        self.agent_b = Agent(
            agent_id="agent_b_analyzer",
            initial_state={
                "name": "AnalyzerAgentB",
                "current_role": "Analyzer",
                "goals": [
                    {"description": "Analyze ideas on the Knowledge Board.", "priority": "high"}
                ],
                "mood": "neutral",
                "influence_points": initial_ip,
                "data_units": initial_du,
                "relationships": {"agent_a_innovator": 0.0, "agent_c_facilitator": 0.0},
            },
        )
        self.agent_c = Agent(
            agent_id="agent_c_facilitator",
            initial_state={
                "name": "FacilitatorAgentC",
                "current_role": "Facilitator",
                "goals": [
                    {
                        "description": "Facilitate collaboration and summarize discussions.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip,
                "data_units": initial_du,
                "relationships": {"agent_a_innovator": 0.0, "agent_b_analyzer": 0.0},
            },
        )

        self.agents = [self.agent_a, self.agent_b, self.agent_c]

        self.simulation = Simulation(
            agents=self.agents,
            vector_store_manager=self.vector_store,
            scenario=SCENARIO_COLLAB,
            # knowledge_board=self.knowledge_board # Removed argument
        )
        # Assign the compiled graph to each agent
        # The actual compiled graph should be imported from basic_agent_graph.py
        # For this example, assuming 'agent_graph_executor' is the one.
        if agent_graph_executor_instance:
            for agent in self.agents:
                agent.graph = agent_graph_executor_instance
        else:
            logger.warning("COMPILED AGENT GRAPH IS NONE. AGENTS WILL NOT HAVE A GRAPH.")

        logger.info("TestMultiAgentCollaboration setup complete.")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestMultiAgentCollaboration...")
        # Clean up ChromaDB
        if hasattr(self.vector_store, "_client") and self.vector_store._client:
            try:
                self.vector_store._client.reset()  # Resets the collection
                # self.vector_store._client.stop() # This might not be available or needed for http client
                logger.debug("ChromaDB client reset.")
            except Exception as e:
                logger.warning(f"Error resetting ChromaDB client: {e}")

        if os.path.exists(CHROMA_DB_PATH_COLLAB):
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_COLLAB)
                logger.debug(f"Removed ChromaDB path: {CHROMA_DB_PATH_COLLAB}")
            except Exception as e:
                logger.warning(f"Could not remove ChromaDB path {CHROMA_DB_PATH_COLLAB}: {e}")
        logger.info("TestMultiAgentCollaboration teardown complete.")

    async def test_collaborative_knowledge_board_interaction(self):
        """Test a multi-turn collaboration scenario involving the Knowledge Board."""
        logger.info("Starting test_collaborative_knowledge_board_interaction...")

        # --- Step 1: AgentA proposes an idea ---
        idea_content = "A novel approach to decentralized AI collaboration using semantic routing."

        initial_ip_a = self.agent_a.state.ip
        initial_du_a = self.agent_a.state.du

        logger.info(f"AgentA initial IP: {initial_ip_a}, DU: {initial_du_a}")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
        )

        mock_agent_a_action_output = AgentActionOutput(
            thought="I should propose my novel idea about decentralized AI collaboration.",
            message_content=idea_content,  # This content should go to KB
            message_recipient_id=None,
            action_intent=AgentActionIntent.PROPOSE_IDEA.value,  # Use enum value
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
            logger.info(f"Knowledge Board entries after AgentA's turn: {board_entries}")

            self.assertTrue(len(board_entries) > 0, "Knowledge Board should not be empty.")

            agent_a_post = None
            for entry in board_entries:
                if entry["agent_id"] == self.agent_a.agent_id and entry["content"] == idea_content:
                    agent_a_post = entry
                    break

            self.assertIsNotNone(
                agent_a_post, f"AgentA's idea '{idea_content}' not found on the Knowledge Board."
            )
            if agent_a_post:
                self.assertEqual(agent_a_post["content"], idea_content)
                self.assertEqual(agent_a_post["agent_id"], self.agent_a.agent_id)
                self.assertEqual(agent_a_post["step"], self.simulation.current_step - 1)

            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA
            award_ip_propose_idea = config.IP_AWARD_FOR_PROPOSAL

            expected_ip_after_action = initial_ip_a - cost_ip_post_idea + award_ip_propose_idea

            logger.info(
                f"AgentA IP after turn: {self.agent_a.state.ip}, expected around: {expected_ip_after_action}"
            )
            logger.info(
                f"AgentA DU after turn: {self.agent_a.state.du}, expected change around: -{cost_du_propose_idea} + passive_gen"
            )

            self.assertEqual(
                self.agent_a.state.ip,
                expected_ip_after_action,
                f"AgentA IP incorrect. Expected ~{expected_ip_after_action}, Got {self.agent_a.state.ip}",
            )

            self.assertLess(
                self.agent_a.state.du,
                initial_du_a,
                "AgentA DU should have decreased after proposing an idea (net of passive generation).",
            )

            mock_gen_struct_output_a.assert_called_once()

        logger.info("Step 1 (AgentA proposes idea) and assertions completed.")

        # --- Step 2: AgentB Perceives AgentA's Idea and Posts Feedback ---
        logger.info("Starting Step 2: AgentB perceives and provides feedback...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_b.agent_id,
            "Simulation did not advance to AgentB's turn.",
        )

        feedback_content = "An insightful idea by @agent_a_innovator! How do you envision handling potential semantic drift in long-running decentralized systems?"

        initial_ip_b = self.agent_b.state.ip
        initial_du_b = self.agent_b.state.du
        initial_mood_b_val = self.agent_b.state.mood_value

        initial_mood_a_val_before_b_turn = self.agent_a.state.mood_value
        initial_relationship_a_to_b_before_b_turn = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )

        logger.info(
            f"AgentB initial IP: {initial_ip_b}, DU: {initial_du_b}, MoodValue: {initial_mood_b_val}"
        )
        logger.info(
            f"AgentA initial MoodValue (before B's turn effects A): {initial_mood_a_val_before_b_turn}, Initial Relationship A->B (before B's turn effects A): {initial_relationship_a_to_b_before_b_turn}"
        )

        mock_agent_b_action_output = AgentActionOutput(
            thought="AgentA's idea is interesting. I should ask a clarifying question to foster discussion and show engagement.",
            message_content=feedback_content,
            message_recipient_id=self.agent_a.agent_id,
            action_intent=AgentActionIntent.ASK_CLARIFICATION.value,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_b_action_output,
        ) as mock_gen_struct_output_b:
            await self.simulation.run_step()  # AgentB's turn

            mock_gen_struct_output_b.assert_called_once()

            cost_du_detailed_clarification = config.DU_COST_REQUEST_DETAILED_CLARIFICATION
            du_gen_rate_b = config.ROLE_DU_GENERATION.get("Analyzer", 1.0)

            logger.info(f"AgentB IP after turn: {self.agent_b.state.ip}")
            logger.info(
                f"AgentB DU after turn: {self.agent_b.state.du}, expected change: -{cost_du_detailed_clarification} + passive_gen_b"
            )

            self.assertAlmostEqual(
                self.agent_b.state.ip,
                initial_ip_b,
                delta=0.1,
                msg="AgentB IP should not significantly change for asking clarification.",
            )

            min_expected_du_b = (
                initial_du_b - cost_du_detailed_clarification + (0.5 * du_gen_rate_b)
            )
            max_expected_du_b = (
                initial_du_b - cost_du_detailed_clarification + (1.5 * du_gen_rate_b)
            )

            self.assertTrue(
                min_expected_du_b <= self.agent_b.state.du <= max_expected_du_b,
                f"AgentB DU out of expected range. Got {self.agent_b.state.du}, expected between {min_expected_du_b:.2f} and {max_expected_du_b:.2f}",
            )

            self.assertAlmostEqual(
                self.agent_a.state.mood_value,
                initial_mood_a_val_before_b_turn,
                delta=0.01,
                msg="AgentA's mood should not change before its turn to process B's message.",
            )
            self.assertAlmostEqual(
                self.agent_a.state.relationships.get(self.agent_b.agent_id, 0.0),
                initial_relationship_a_to_b_before_b_turn,
                delta=0.01,
                msg="AgentA's relationship towards B should not change before A processes B's message.",
            )

        logger.info("Step 2 (AgentB provides feedback) and assertions completed.")

        # --- Step 3: AgentC (Facilitator) Perceives Interaction and Posts Encouragement/Summary ---
        logger.info("Starting Step 3: AgentC perceives and broadcasts encouragement...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_c.agent_id,
            "Simulation did not advance to AgentC's turn.",
        )

        initial_ip_c = self.agent_c.state.ip
        initial_du_c = self.agent_c.state.du
        logger.info(f"AgentC initial IP: {initial_ip_c}, DU: {initial_du_c}")

        facilitator_message_content = (
            "Great discussion between AgentA and AgentB! Let's keep the ideas flowing."
        )

        mock_agent_c_action_output = AgentActionOutput(
            thought="The discussion is productive. I should encourage continued collaboration and summarize key points if needed later.",
            message_content=facilitator_message_content,
            message_recipient_id=None,  # Broadcast
            action_intent=AgentActionIntent.CONTINUE_COLLABORATION.value,  # Results in broadcast
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_c_action_output,
        ) as mock_gen_struct_output_c:
            await self.simulation.run_step()  # AgentC's turn

            # Assertion 6: AgentC's Message Broadcast
            mock_gen_struct_output_c.assert_called_once()
            logger.info("AgentC broadcasted its message.")

            # Assertion 7: AgentC IP/DU Debit
            # IP cost for general message/continue_collaboration: Typically none explicit in basic_graph handlers beyond AgentState defaults which are not directly applied often.
            # DU cost: Similarly, passive generation is the main factor.
            du_gen_rate_c = config.ROLE_DU_GENERATION.get("Facilitator", 1.0)

            logger.info(f"AgentC IP after turn: {self.agent_c.state.ip}")
            logger.info(
                f"AgentC DU after turn: {self.agent_c.state.du}, expected change: + passive_gen_c"
            )

            # IP should be stable if no explicit cost for CONTINUE_COLLABORATION intent message
            self.assertAlmostEqual(
                self.agent_c.state.ip,
                initial_ip_c,
                delta=0.1,
                msg="AgentC IP should not significantly change for broadcasting encouragement.",
            )

            # DU should increase due to passive generation (assuming action_intent != idle)
            # Action intent is CONTINUE_COLLABORATION, so passive DU generation applies.
            min_expected_du_c = initial_du_c + (0.5 * du_gen_rate_c)
            max_expected_du_c = initial_du_c + (1.5 * du_gen_rate_c)
            # If initial_du_c was the value before this turn, and it made a non-idle action,
            # it should only increase or stay same if generation is low and some minor cost applied (unlikely here)
            self.assertTrue(
                self.agent_c.state.du
                >= initial_du_c,  # Should at least gain a little or stay same if costs perfectly offset
                f"AgentC DU should have increased or stayed similar due to passive generation. Got {self.agent_c.state.du}, initial {initial_du_c}",
            )
            self.assertTrue(
                min_expected_du_c <= self.agent_c.state.du <= max_expected_du_c,
                f"AgentC DU out of expected range after passive generation. Got {self.agent_c.state.du}, expected {min_expected_du_c:.2f}-{max_expected_du_c:.2f}",
            )

        logger.info("Step 3 (AgentC broadcasts encouragement) and assertions completed.")

        # --- Step 4: AgentA's Second Turn - Processes AgentB's Feedback ---
        logger.info("Starting Step 4: AgentA's second turn, processes B's feedback...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
            "Simulation did not advance to AgentA's second turn.",
        )

        # Record AgentA's mood and relationship with B *before* this turn's processing
        # These values are from *after* AgentB's turn but *before* AgentA processes B's message.
        mood_a_before_processing_b_msg = self.agent_a.state.mood_value
        relationship_a_to_b_before_processing_b_msg = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )

        logger.info(
            f"AgentA MoodValue before processing B's message: {mood_a_before_processing_b_msg}"
        )
        logger.info(
            f"AgentA->B Relationship before processing B's message: {relationship_a_to_b_before_processing_b_msg}"
        )

        # For this turn, we let AgentA generate its own action. We are interested in the state changes
        # from *receiving* AgentB's message.
        # The `analyze_perception_sentiment_node` in AgentA's graph will process `feedback_content` from AgentB.
        # `feedback_content` was positive.
        await self.simulation.run_step()  # AgentA's second turn

        # Assertion 8: AgentA's Mood Update from AgentB's Message
        mood_a_after_processing_b_msg = self.agent_a.state.mood_value
        logger.info(
            f"AgentA MoodValue after processing B's message: {mood_a_after_processing_b_msg}"
        )
        self.assertGreater(
            mood_a_after_processing_b_msg,
            mood_a_before_processing_b_msg,
            "AgentA's mood should have improved after processing positive feedback from AgentB.",
        )

        # Assertion 9: AgentA's Relationship with AgentB Update
        # This assertion might fail if receiving a message doesn't directly trigger relationship update for the receiver.
        # It typically updates when AgentA *sends* a targeted message to B.
        relationship_a_to_b_after_processing_b_msg = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )
        logger.info(
            f"AgentA->B Relationship after processing B's message: {relationship_a_to_b_after_processing_b_msg}"
        )
        # Based on instructions, we expect an increase. If this fails, it highlights a characteristic of the current graph.
        self.assertGreater(
            relationship_a_to_b_after_processing_b_msg,
            relationship_a_to_b_before_processing_b_msg,
            "AgentA's relationship with AgentB should improve after processing positive feedback. (Note: May depend on graph logic for received messages)",
        )

        # Assertion 10 (AgentA's Message Queue) - Optional, skipping for now as no explicit state for this.

        logger.info("Step 4 (AgentA processes B's feedback) and assertions completed.")


# To run this test specifically (requires pytest-asyncio):
# python -m pytest -m integration tests/integration/test_multi_agent_collaboration.py
if __name__ == "__main__":
    print(
        "Please run this test using 'python -m pytest tests/integration/test_multi_agent_collaboration.py'"
    )

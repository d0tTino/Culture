#!/usr/bin/env python
"""
Integration tests for multi-agent collaboration scenarios,
focusing on Knowledge Board interactions and their impact on agent states.
"""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

# Add project root to sys.path to allow importing src modules
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent, AgentActionOutput
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


class TestMultiAgentCollaboration(unittest.IsolatedAsyncioTestCase):
    """Tests multi-agent collaboration, knowledge board usage, and resulting state changes."""

    def setUp(self):
        """Set up the simulation environment with three agents for collaboration."""
        logger.info("Setting up TestMultiAgentCollaboration...")

        # Clean up previous test DB if it exists
        if Path(CHROMA_DB_PATH_COLLAB).exists():
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

        if Path(CHROMA_DB_PATH_COLLAB).exists():
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_COLLAB)
                logger.debug(f"Removed ChromaDB path: {CHROMA_DB_PATH_COLLAB}")
            except Exception as e:
                logger.warning(f"Could not remove ChromaDB path {CHROMA_DB_PATH_COLLAB}: {e}")
        logger.info("TestMultiAgentCollaboration teardown complete.")

    @pytest.mark.asyncio
    async def test_collaborative_knowledge_board_interaction(self):
        logger.info("Starting test_collaborative_knowledge_board_interaction...")

        # --- GLOBAL ROUND 1 ---
        logger.info("--- Starting GLOBAL ROUND 1 ---")
        self.simulation.start_new_round()  # Simulation current_step becomes 1

        # --- AgentA proposes an idea (Turn 1 in Round 1) ---
        logger.info("Round 1, Turn 1: AgentA proposes an idea...")
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

        # Old patch removed
        # with patch(
        #     "src.agents.graphs.basic_agent_graph.generate_structured_output",
        #     return_value=mock_agent_a_action_output,
        # ) as mock_gen_struct_output_a:

        # New mocking strategy
        # The graph's generate_thought_and_message_node calls agent.async_generate_role_prefixed_thought
        # and agent.async_select_action_intent.
        # We mock these directly on the agent instance.
        original_agent_a_thought_gen = self.agent_a.async_generate_role_prefixed_thought
        original_agent_a_intent_sel = self.agent_a.async_select_action_intent

        self.agent_a.async_generate_role_prefixed_thought = AsyncMock(
            return_value=MagicMock(thought=mock_agent_a_action_output.thought)
        )
        # async_select_action_intent in base_agent.py is expected to return an object that
        # the graph node then wraps into AgentActionOutput, or it can return AgentActionOutput directly.
        # For testing, it's simpler if our mock returns the fully formed AgentActionOutput.
        # The graph node `generate_thought_and_message_node` will use this.
        self.agent_a.async_select_action_intent = AsyncMock(
            return_value=mock_agent_a_action_output
        )

        try:
            await self.simulation.run_step()  # AgentA's turn

            board_entries = self.simulation.knowledge_board.get_full_entries()
            logger.info(f"Knowledge Board entries after AgentA's turn: {board_entries}")

            self.assertTrue(len(board_entries) > 0, "Knowledge Board should not be empty.")

            agent_a_post = None
            for entry in board_entries:
                if (
                    entry["agent_id"] == self.agent_a.agent_id
                    and entry["content_full"] == idea_content
                ):
                    agent_a_post = entry
                    break

            self.assertIsNotNone(
                agent_a_post, f"AgentA's idea '{idea_content}' not found on the Knowledge Board."
            )
            if agent_a_post:
                self.assertEqual(agent_a_post["content_full"], idea_content)
                self.assertEqual(agent_a_post["agent_id"], self.agent_a.agent_id)
                self.assertEqual(agent_a_post["step"], self.simulation.current_step)

            # self.assertLess( # This assertion might be too strict if DU generation is variable
            #     self.agent_a.state.du,
            #     initial_du_a,
            #     "AgentA DU should have decreased after proposing an idea (net of passive generation).",
            # )
            # Instead, check if DU changed reasonably, considering cost and generation.
            # Exact DU check is hard due to random factor in generation.
            # Cost is PROPOSE_DETAILED_IDEA_DU_COST = 2.0
            # Generation is Innovator base = 1.0, factor 0.5-1.5. So, gen = 0.5 to 1.5.
            # Net change = (-2.0) + (0.5 to 1.5) = -1.5 to -0.5
            # So, DU should decrease.
            self.assertTrue(
                self.agent_a.state.du < initial_du_a,
                f"AgentA DU should have decreased. Initial: {initial_du_a}, Final: {self.agent_a.state.du}",
            )

            # mock_gen_struct_output_a.assert_called_once() # Old assertion based on patch
            self.agent_a.async_generate_role_prefixed_thought.assert_called_once()
            self.agent_a.async_select_action_intent.assert_called_once()

        finally:
            # Restore original methods
            self.agent_a.async_generate_role_prefixed_thought = original_agent_a_thought_gen
            self.agent_a.async_select_action_intent = original_agent_a_intent_sel

        logger.info("Round 1, Turn 1 (AgentA proposes idea) completed.")

        # --- AgentB perceives AgentA's idea and provides positive feedback (Turn 2 in Round 1) ---
        logger.info("Round 1, Turn 2: AgentB perceives and provides feedback...")

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

        # New mocking strategy for Agent B
        original_agent_b_thought_gen = self.agent_b.async_generate_role_prefixed_thought
        original_agent_b_intent_sel = self.agent_b.async_select_action_intent
        self.agent_b.async_generate_role_prefixed_thought = AsyncMock(
            return_value=MagicMock(thought=mock_agent_b_action_output.thought)
        )
        self.agent_b.async_select_action_intent = AsyncMock(
            return_value=mock_agent_b_action_output
        )

        try:
            await self.simulation.run_step()  # AgentB's turn

            # mock_gen_struct_output_b.assert_called_once() # Old assertion
            self.agent_b.async_generate_role_prefixed_thought.assert_called_once()
            self.agent_b.async_select_action_intent.assert_called_once()

            cost_du_detailed_clarification = config.DU_COST_REQUEST_DETAILED_CLARIFICATION
            # du_gen_rate_b is now a dictionary, e.g., {"base": 1.0, "bonus_factor": 0.5}
            du_gen_config_b = config.ROLE_DU_GENERATION.get(
                "Analyzer", {"base": 1.0, "bonus_factor": 0.5}
            )
            du_gen_base_rate_b = du_gen_config_b.get("base", 1.0)
            du_gen_bonus_factor_b = du_gen_config_b.get("bonus_factor", 0.0)

            min_passive_gen_b = du_gen_base_rate_b * (1 + du_gen_bonus_factor_b * 0.8)
            max_passive_gen_b = du_gen_base_rate_b * (1 + du_gen_bonus_factor_b * 1.2)

            min_expected_du_b = initial_du_b - cost_du_detailed_clarification + min_passive_gen_b
            max_expected_du_b = (
                initial_du_b - cost_du_detailed_clarification + max_passive_gen_b + 0.0000001
            )  # Epsilon

            logger.info(
                f"TEST DEBUG: cost_du_detailed_clarification from config = {config.DU_COST_REQUEST_DETAILED_CLARIFICATION}"
            )
            logger.info(f"TEST DEBUG: initial_du_b = {initial_du_b}")
            logger.info(f"TEST DEBUG: du_gen_base_rate_b = {du_gen_base_rate_b}")
            logger.info(f"TEST DEBUG: du_gen_bonus_factor_b = {du_gen_bonus_factor_b}")
            logger.info(f"TEST DEBUG: Calculated min_expected_du_b = {min_expected_du_b}")
            logger.info(f"TEST DEBUG: Calculated max_expected_du_b = {max_expected_du_b}")
            logger.info(f"TEST DEBUG: Actual Agent B DU = {self.agent_b.state.du}")

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
        finally:
            self.agent_b.async_generate_role_prefixed_thought = original_agent_b_thought_gen
            self.agent_b.async_select_action_intent = original_agent_b_intent_sel

        logger.info("Round 1, Turn 2 (AgentB provides feedback) completed.")

        # --- AgentC perceives messages and broadcasts encouragement (Turn 3 in Round 1) ---
        logger.info("Round 1, Turn 3: AgentC perceives and broadcasts encouragement...")
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

        # New mocking strategy for Agent C
        original_agent_c_thought_gen = self.agent_c.async_generate_role_prefixed_thought
        original_agent_c_intent_sel = self.agent_c.async_select_action_intent
        self.agent_c.async_generate_role_prefixed_thought = AsyncMock(
            return_value=MagicMock(thought=mock_agent_c_action_output.thought)
        )
        self.agent_c.async_select_action_intent = AsyncMock(
            return_value=mock_agent_c_action_output
        )

        try:
            await self.simulation.run_step()  # AgentC's turn

            # Assertion 6: AgentC's Message Broadcast
            # mock_gen_struct_output_c.assert_called_once() # Old assertion
            self.agent_c.async_generate_role_prefixed_thought.assert_called_once()
            self.agent_c.async_select_action_intent.assert_called_once()
            logger.info("AgentC broadcasted its message.")

            # Assertion 7: AgentC IP/DU Debit
            # IP cost for general message/continue_collaboration: Handled by BaseAgent.
            # DU cost: Similarly, passive generation is the main factor.
            du_gen_config_c = config.ROLE_DU_GENERATION.get(
                "Facilitator", {"base": 1.0, "bonus_factor": 0.3}
            )
            du_gen_base_rate_c = du_gen_config_c.get("base", 1.0)
            du_gen_bonus_factor_c = du_gen_config_c.get("bonus_factor", 0.0)

            min_passive_gen_c = du_gen_base_rate_c * (1 + du_gen_bonus_factor_c * 0.8)
            max_passive_gen_c = du_gen_base_rate_c * (1 + du_gen_bonus_factor_c * 1.2)

            min_expected_du_c = initial_du_c + min_passive_gen_c
            max_expected_du_c = initial_du_c + max_passive_gen_c + 0.0000001  # Epsilon

            self.assertTrue(
                self.agent_c.state.du
                >= initial_du_c,  # Should at least gain a little or stay same if costs perfectly offset
                f"AgentC DU should have increased or stayed similar due to passive generation. Got {self.agent_c.state.du}, initial {initial_du_c}",
            )
            self.assertTrue(
                min_expected_du_c <= self.agent_c.state.du <= max_expected_du_c,
                f"AgentC DU out of expected range after passive generation. Got {self.agent_c.state.du}, expected {min_expected_du_c:.2f}-{max_expected_du_c:.2f}",
            )
        finally:
            self.agent_c.async_generate_role_prefixed_thought = original_agent_c_thought_gen
            self.agent_c.async_select_action_intent = original_agent_c_intent_sel

        logger.info("Round 1, Turn 3 (AgentC broadcasts encouragement) completed.")

        # --- GLOBAL ROUND 2 ---
        logger.info("--- Starting GLOBAL ROUND 2 ---")
        self.simulation.start_new_round()  # Simulation current_step becomes 2
        # Now, messages from AgentA (Turn1), B(Turn2), C(Turn3) of Round 1 are in messages_to_perceive_this_round

        # --- AgentA's second turn, processes B's feedback and C's encouragement (Turn 1 in Round 2) ---
        logger.info("Round 2, Turn 1: AgentA's second turn, processes feedback...")
        initial_mood_value_a_before_round_2_turn = self.agent_a.state.mood_value

        mock_response_agent_a_round2 = AgentActionOutput(
            thought="I've received feedback, I will acknowledge it.",
            message_content="Thanks for the feedback everyone!",
            message_recipient_id=None,
            action_intent=AgentActionIntent.CONTINUE_COLLABORATION.value,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        original_agent_a_r2_thought_gen = self.agent_a.async_generate_role_prefixed_thought
        original_agent_a_r2_intent_sel = self.agent_a.async_select_action_intent

        self.agent_a.async_generate_role_prefixed_thought = AsyncMock(
            return_value=MagicMock(thought=mock_response_agent_a_round2.thought)
        )
        self.agent_a.async_select_action_intent = AsyncMock(
            return_value=mock_response_agent_a_round2
        )

        # Patch sentiment analysis where it's used in basic_agent_graph
        with patch(
            "src.agents.graphs.basic_agent_graph.analyze_sentiment", return_value="0.5"
        ) as mock_sentiment_analysis_r2:
            try:
                await self.simulation.run_step()  # AgentA's turn in Round 2

                # Assertions for Agent A's second turn
                self.agent_a.async_generate_role_prefixed_thought.assert_called_once()
                self.agent_a.async_select_action_intent.assert_called_once()
                # It should be called for Agent B's message and Agent C's message perceived by Agent A
                self.assertEqual(mock_sentiment_analysis_r2.call_count, 2)

                # Verify mood update based on the (mocked) positive sentiment
                # initial_mood_value_a_before_round_2_turn was 0.0
                logger.info(
                    f"AgentA MoodValue after processing Round 1 messages: {self.agent_a.state.mood_value:.2f} (was {initial_mood_value_a_before_round_2_turn:.2f})"
                )
                self.assertTrue(
                    self.agent_a.state.mood_value > initial_mood_value_a_before_round_2_turn,
                    f"AgentA's mood should improve after processing positive feedback from B and C. Mood: {self.agent_a.state.mood_value:.2f}, Initial: {initial_mood_value_a_before_round_2_turn:.2f}",
                )

            finally:
                # Restore original methods for Agent A
                self.agent_a.async_generate_role_prefixed_thought = original_agent_a_r2_thought_gen
                self.agent_a.async_select_action_intent = original_agent_a_r2_intent_sel

        logger.info("Round 2, Turn 1 (AgentA processes feedback) and assertions completed.")

        # ... (Optional: Add more rounds/assertions if needed) ...


# To run this test specifically (requires pytest-asyncio):
# python -m pytest -m integration tests/integration/test_multi_agent_collaboration.py

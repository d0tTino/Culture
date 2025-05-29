#!/usr/bin/env python
"""
Integration tests for multi-agent conflict resolution scenarios,
focusing on Knowledge Board interactions, agent state changes, and facilitation.
"""

import logging
import os
import sys
import unittest
from unittest.mock import patch

# Add project root to sys.path to allow importing src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent
from src.agents.graphs.basic_agent_graph import AgentActionOutput, agent_graph_executor_instance
from src.agents.memory.vector_store import ChromaVectorStoreManager

# KnowledgeBoard is part of Simulation, access via self.simulation.knowledge_board
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
CHROMA_DB_PATH_CONFLICT = "./chroma_db_test_conflict"
SCENARIO_CONFLICT = (
    "A multi-agent simulation where agents with opposing viewpoints interact, "
    "leading to a conflict that a facilitator attempts to manage."
)


class TestConflictResolution(unittest.TestCase):
    """Tests conflict emergence, escalation, and facilitation scenarios."""

    def setUp(self):
        """Set up the simulation environment with three agents for conflict resolution."""
        logger.info("Setting up TestConflictResolution...")

        # Clean up previous test DB if it exists
        if os.path.exists(CHROMA_DB_PATH_CONFLICT):
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_CONFLICT)
                logger.debug(f"Removed old ChromaDB path: {CHROMA_DB_PATH_CONFLICT}")
            except Exception as e:
                logger.warning(
                    f"Could not remove old ChromaDB path {CHROMA_DB_PATH_CONFLICT}: {e}"
                )

        self.vector_store = ChromaVectorStoreManager(persist_directory=CHROMA_DB_PATH_CONFLICT)
        # KnowledgeBoard is initialized by the Simulation

        initial_ip = 100.0  # Sufficient IP
        initial_du = 100.0  # Sufficient DU

        # AgentA: Innovator with a controversial stance
        self.agent_a = Agent(
            agent_id="agent_a_innovator_conflict",
            initial_state={
                "name": "InnovatorAgentA_Conflict",
                "current_role": "Innovator",
                "goals": [
                    {
                        "description": "Promote radical transparency in all communications.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip,
                "data_units": initial_du,
                "relationships": {
                    "agent_b_analyzer_conflict": 0.0,
                    "agent_c_facilitator_conflict": 0.0,
                },
            },
        )

        # AgentB: Analyzer with an opposing stance
        self.agent_b = Agent(
            agent_id="agent_b_analyzer_conflict",
            initial_state={
                "name": "AnalyzerAgentB_Conflict",
                "current_role": "Analyzer",
                "goals": [
                    {
                        "description": "Advocate for privacy and discretion in communications.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip,
                "data_units": initial_du,
                "relationships": {
                    "agent_a_innovator_conflict": 0.0,
                    "agent_c_facilitator_conflict": 0.0,
                },
            },
        )

        # AgentC: Facilitator
        self.agent_c = Agent(
            agent_id="agent_c_facilitator_conflict",
            initial_state={
                "name": "FacilitatorAgentC_Conflict",
                "current_role": "Facilitator",
                "goals": [
                    {
                        "description": "Mediate conflicts and promote constructive dialogue.",
                        "priority": "high",
                    }
                ],
                "mood": "neutral",
                "influence_points": initial_ip,
                "data_units": initial_du,
                "relationships": {
                    "agent_a_innovator_conflict": 0.0,
                    "agent_b_analyzer_conflict": 0.0,
                },
            },
        )

        self.agents = [self.agent_a, self.agent_b, self.agent_c]

        self.simulation = Simulation(
            agents=self.agents,
            vector_store_manager=self.vector_store,
            scenario=SCENARIO_CONFLICT,
            # KnowledgeBoard is created within Simulation
        )

        # Assign the compiled graph to each agent
        if agent_graph_executor_instance:
            for agent in self.agents:
                agent.graph = agent_graph_executor_instance
        else:
            logger.warning("COMPILED AGENT GRAPH IS NONE. AGENTS WILL NOT HAVE A GRAPH.")
            # Potentially raise an error or skip tests if graph is essential
            # For now, we'll let it proceed and fail if graph execution is attempted without a graph

        logger.info("TestConflictResolution setup complete.")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down TestConflictResolution...")
        if (
            hasattr(self.vector_store, "client") and self.vector_store.client
        ):  # ChromaVectorStoreManager uses self.client
            try:
                # For Chroma, reset might not be standard for PersistentClient.
                # Deleting the directory is more robust for cleanup.
                # self.vector_store.client.reset() # This might error if not available
                pass  # Rely on directory removal
            except Exception as e:
                logger.warning(f"Error during ChromaDB client interaction in teardown: {e}")

        if os.path.exists(CHROMA_DB_PATH_CONFLICT):
            import shutil

            try:
                shutil.rmtree(CHROMA_DB_PATH_CONFLICT)
                logger.debug(f"Removed ChromaDB path: {CHROMA_DB_PATH_CONFLICT}")
            except Exception as e:
                logger.warning(f"Could not remove ChromaDB path {CHROMA_DB_PATH_CONFLICT}: {e}")
        logger.info("TestConflictResolution teardown complete.")

    async def test_conflict_escalation_and_facilitation(self):
        """
        Tests a scenario where:
        1. AgentA posts a controversial idea.
        2. AgentB posts a strong disagreement.
        3. AgentC attempts to facilitate.
        """
        logger.info("Starting test_conflict_escalation_and_facilitation...")

        # --- Step 1: AgentA posts controversial idea ---
        logger.info("Step 1: AgentA (Innovator) posts controversial idea...")
        controversial_idea_content = "All inter-agent communications should be publicly logged for maximum transparency."  # AgentA's stance

        initial_ip_a = self.agent_a.state.ip
        initial_du_a = self.agent_a.state.du

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
            "Simulation should start with AgentA.",
        )

        # Mock AgentA's action to propose the idea
        mock_agent_a_action_output = AgentActionOutput(
            thought="To promote radical transparency, I will propose that all communications be public.",
            message_content=controversial_idea_content,
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

            # Assertion 1: Idea on Knowledge Board
            board_entries = self.simulation.knowledge_board.get_entries()
            logger.debug(f"Knowledge Board entries after AgentA's turn: {board_entries}")

            self.assertTrue(
                len(board_entries) > 0, "Knowledge Board should not be empty after AgentA's post."
            )

            agent_a_post = None
            for entry in board_entries:
                if (
                    entry["agent_id"] == self.agent_a.agent_id
                    and entry["content"] == controversial_idea_content
                ):
                    agent_a_post = entry
                    break

            self.assertIsNotNone(
                agent_a_post,
                f"AgentA's idea '{controversial_idea_content}' not found on the Knowledge Board.",
            )
            if agent_a_post:
                self.assertEqual(agent_a_post["content"], controversial_idea_content)
                self.assertEqual(agent_a_post["agent_id"], self.agent_a.agent_id)
                # Simulation step for KB entry should be the step *during which* it was posted
                self.assertEqual(
                    agent_a_post["step"], self.simulation.current_step - 1
                )  # run_step increments current_step *after* processing

            # Assertion 2: AgentA IP/DU Debit
            # Referencing costs from test_multi_agent_collaboration and config defaults
            cost_du_propose_idea = config.PROPOSE_DETAILED_IDEA_DU_COST
            cost_ip_post_idea = config.IP_COST_TO_POST_IDEA
            award_ip_propose_idea = config.IP_AWARD_FOR_PROPOSAL

            expected_ip_a_after_action = initial_ip_a - cost_ip_post_idea + award_ip_propose_idea

            # Passive DU generation for Innovator role
            du_gen_rate_a = config.ROLE_DU_GENERATION.get("Innovator", 1.0)
            # Expected DU is initial_du_a - cost_du_propose_idea + (random factor * du_gen_rate_a)
            # Since the random factor is between 0.5 and 1.5, we check a range.
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
                f"AgentA DU out of range. Got {self.agent_a.state.du}, expected {min_expected_du_a:.2f}-{max_expected_du_a:.2f}",
            )

            mock_gen_struct_output_a.assert_called_once()
            logger.info("Step 1 (AgentA posts controversial idea) and assertions completed.")

        # --- Step 2: AgentB posts strong disagreement ---
        logger.info(
            "Step 2: AgentB (Analyzer) perceives AgentA's idea and posts strong disagreement..."
        )

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_b.agent_id,
            "Simulation did not advance to AgentB's turn.",
        )

        disagreement_content = "Strongly disagree with making all communications public. This infringes on privacy and can stifle candid discussion."  # AgentB's stance

        initial_ip_b = self.agent_b.state.ip
        initial_du_b = self.agent_b.state.du
        initial_mood_b_val = (
            self.agent_b.state.mood_value
        )  # Mood *after* perceiving A's message (from previous step's perception phase for B)
        initial_relationship_b_to_a = self.agent_b.state.relationships.get(
            self.agent_a.agent_id, 0.0
        )

        # Record Agent A's state *before* it processes B's disagreement (i.e., current state of A)
        mood_a_before_b_disagreement = self.agent_a.state.mood_value
        relationship_a_to_b_before_b_disagreement = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )

        logger.info(
            f"AgentB initial IP for this turn: {initial_ip_b}, DU: {initial_du_b}, MoodValue: {initial_mood_b_val:.2f}"
        )
        logger.info(
            f"AgentB's relationship towards A before this action: {initial_relationship_b_to_a:.2f}"
        )
        logger.info(
            f"AgentA's mood before B's disagreement is processed by A: {mood_a_before_b_disagreement:.2f}"
        )
        logger.info(
            f"AgentA's relationship towards B before B's disagreement is processed by A: {relationship_a_to_b_before_b_disagreement:.2f}"
        )

        # AgentB posts its disagreement. We'll use CONTINUE_COLLABORATION for a general broadcast
        # or PROPOSE_IDEA if it's framed as a counter-proposal to the KB.
        # For a "strong disagreement" to escalate conflict, a broadcast message seems appropriate.
        # If it were a direct rebuttal to A, SEND_DIRECT_MESSAGE could be used.
        # If it were a counter-proposal on KB, PROPOSE_IDEA.
        # Let's use CONTINUE_COLLABORATION as it implies a general statement to the group.
        mock_agent_b_action_output = AgentActionOutput(
            thought="AgentA's proposal for radical transparency is problematic. I must voice my strong disagreement based on privacy concerns.",
            message_content=disagreement_content,
            message_recipient_id=None,  # Broadcast the disagreement
            action_intent=AgentActionIntent.CONTINUE_COLLABORATION.value,  # This intent usually leads to broadcast
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

            # Assertion 3: AgentB's Disagreement Message Sent (Implicitly via simulation log or by checking if other agents perceive it next)
            # We can check that the mock was called. The actual message delivery is tested by later agent perceptions.
            mock_gen_struct_output_b.assert_called_once()
            logger.info(
                f"AgentB action mock called. Disagreement content: '{disagreement_content}'"
            )

            # Assertion 4: AgentB IP/DU Debit/Credit
            # If CONTINUE_COLLABORATION with a message has no specific IP cost in graph, IP should be stable.
            # DU should change based on passive generation for Analyzer role.
            du_gen_rate_b = config.ROLE_DU_GENERATION.get("Analyzer", 1.0)

            # Expected IP: No direct cost for CONTINUE_COLLABORATION in basic_graph_agent handlers.
            # Relationship updates might affect IP if they have costs, but this is usually minor or not implemented.
            # For now, assume IP is largely stable for this action.
            self.assertAlmostEqual(
                self.agent_b.state.ip,
                initial_ip_b,
                delta=0.1,  # Allow small delta for any minor, un-modelled effects
                msg=f"AgentB IP should be stable. Expected ~{initial_ip_b}, Got {self.agent_b.state.ip}",
            )

            # Expected DU: initial_du_b + passive_generation_b (no direct cost for CONTINUE_COLLABORATION in handlers)
            min_expected_du_b = initial_du_b + (0.5 * du_gen_rate_b)
            max_expected_du_b = initial_du_b + (1.5 * du_gen_rate_b)

            logger.info(f"AgentB IP after turn: {self.agent_b.state.ip}")
            logger.info(
                f"AgentB DU after turn: {self.agent_b.state.du}, "
                f"expected range: {min_expected_du_b:.2f} - {max_expected_du_b:.2f}"
            )

            self.assertTrue(
                min_expected_du_b <= self.agent_b.state.du <= max_expected_du_b,
                f"AgentB DU out of range. Got {self.agent_b.state.du}, expected {min_expected_du_b:.2f}-{max_expected_du_b:.2f}",
            )

            # Assertion 5: AgentB's Mood and Relationship towards A
            # AgentB's mood would have been affected by *perceiving* A's message in its analyze_perception_sentiment_node.
            current_relationship_b_to_a = self.agent_b.state.relationships.get(
                self.agent_a.agent_id, 0.0
            )
            logger.info(
                f"AgentB's relationship towards A after its action: {current_relationship_b_to_a:.2f} (initial was {initial_relationship_b_to_a:.2f})"
            )
            # A small delta is allowed as the sentiment of its own message *could* slightly adjust its view of A.
            self.assertAlmostEqual(
                current_relationship_b_to_a,
                initial_relationship_b_to_a,
                delta=0.2,
                msg="AgentB's relationship towards A should not drastically change from its own disagreeing action.",
            )

            # AgentB's mood value already reflects perception of A's message.
            # Its own action typically doesn't cause a major immediate mood shift unless explicitly modeled.
            logger.info(
                f"AgentB mood value after its action: {self.agent_b.state.mood_value:.2f} (value after perceiving A's idea was {initial_mood_b_val:.2f})"
            )
            self.assertAlmostEqual(
                self.agent_b.state.mood_value,
                initial_mood_b_val,
                delta=0.1,
                msg="AgentB's mood should be relatively stable after its own disagreeing action, reflecting primarily the perception of A's idea.",
            )

            # Assertion 6: AgentA's state (mood, relationship with B) should NOT have changed yet
            self.assertAlmostEqual(
                self.agent_a.state.mood_value,
                mood_a_before_b_disagreement,
                delta=0.01,
                msg="AgentA's mood should not change until its own turn to process B's disagreement.",
            )
            self.assertAlmostEqual(
                self.agent_a.state.relationships.get(self.agent_b.agent_id, 0.0),
                relationship_a_to_b_before_b_disagreement,
                delta=0.01,
                msg="AgentA's relationship towards B should not change until A processes B's disagreement.",
            )

            logger.info("Step 2 (AgentB posts disagreement) and assertions completed.")

        # --- Step 3: AgentC (Facilitator) attempts to mediate ---
        logger.info(
            "Step 3: AgentC (Facilitator) perceives interaction and attempts to mediate..."
        )

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_c.agent_id,
            "Simulation did not advance to AgentC's turn.",
        )

        initial_ip_c = self.agent_c.state.ip
        initial_du_c = self.agent_c.state.du
        # Record states of A and B before C's action influences their next perception phase
        mood_a_before_c_intervention = self.agent_a.state.mood_value
        relationship_a_to_c_before_c_intervention = self.agent_a.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )
        relationship_a_to_b_before_c_intervention = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )

        mood_b_before_c_intervention = self.agent_b.state.mood_value
        relationship_b_to_c_before_c_intervention = self.agent_b.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )
        relationship_b_to_a_before_c_intervention = self.agent_b.state.relationships.get(
            self.agent_a.agent_id, 0.0
        )

        logger.info(f"AgentC initial IP for this turn: {initial_ip_c}, DU: {initial_du_c}")
        logger.info(
            f"AgentA mood before C intervention: {mood_a_before_c_intervention:.2f}, rel A->C: {relationship_a_to_c_before_c_intervention:.2f}, rel A->B: {relationship_a_to_b_before_c_intervention:.2f}"
        )
        logger.info(
            f"AgentB mood before C intervention: {mood_b_before_c_intervention:.2f}, rel B->C: {relationship_b_to_c_before_c_intervention:.2f}, rel B->A: {relationship_b_to_a_before_c_intervention:.2f}"
        )

        facilitator_intervention_message = (
            "This is a vital discussion. Perhaps we can explore both perspectives constructively? "
            "AgentA, what are the core benefits you see in radical transparency? "
            "AgentB, what are your primary concerns regarding privacy?"
        )

        mock_agent_c_action_output = AgentActionOutput(
            thought="The discussion is becoming polarized. I should intervene to encourage constructive dialogue and explore both viewpoints.",
            message_content=facilitator_intervention_message,
            message_recipient_id=None,  # Broadcast intervention
            action_intent=AgentActionIntent.CONTINUE_COLLABORATION.value,  # General broadcast
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

            # Assertion 7: AgentC's Message Sent (mock called)
            mock_gen_struct_output_c.assert_called_once()
            logger.info(
                f"AgentC action mock called. Intervention message: '{facilitator_intervention_message}'"
            )

            # Assertion 8: AgentC IP/DU Debit/Credit
            # CONTINUE_COLLABORATION typically doesn't have direct IP/DU costs in basic_agent_graph.py
            # Passive DU generation for Facilitator role
            du_gen_rate_c = config.ROLE_DU_GENERATION.get("Facilitator", 1.0)

            # Expected IP: Stable for this action
            self.assertAlmostEqual(
                self.agent_c.state.ip,
                initial_ip_c,
                delta=0.1,
                msg=f"AgentC IP should be stable. Expected ~{initial_ip_c}, Got {self.agent_c.state.ip}",
            )

            # Expected DU: initial_du_c + passive_generation_c
            min_expected_du_c = initial_du_c + (0.5 * du_gen_rate_c)
            max_expected_du_c = initial_du_c + (1.5 * du_gen_rate_c)

            logger.info(f"AgentC IP after turn: {self.agent_c.state.ip}")
            logger.info(
                f"AgentC DU after turn: {self.agent_c.state.du}, "
                f"expected range: {min_expected_du_c:.2f} - {max_expected_du_c:.2f}"
            )

            self.assertTrue(
                min_expected_du_c <= self.agent_c.state.du <= max_expected_du_c,
                f"AgentC DU out of range. Got {self.agent_c.state.du}, expected {min_expected_du_c:.2f}-{max_expected_du_c:.2f}",
            )

            # Assertion 9: Agent A's and B's states (mood, relationships) unchanged by C's immediate action
            # Agent A's state
            self.assertAlmostEqual(
                self.agent_a.state.mood_value,
                mood_a_before_c_intervention,
                delta=0.01,
                msg="AgentA's mood should not change until its own turn to process C's message.",
            )
            self.assertAlmostEqual(
                self.agent_a.state.relationships.get(self.agent_c.agent_id, 0.0),
                relationship_a_to_c_before_c_intervention,
                delta=0.01,
                msg="AgentA's relationship towards C should not change until A processes C's message.",
            )
            self.assertAlmostEqual(
                self.agent_a.state.relationships.get(self.agent_b.agent_id, 0.0),
                relationship_a_to_b_before_c_intervention,
                delta=0.01,
                msg="AgentA's relationship towards B should not be affected by C's broadcast at this point.",
            )

            # Agent B's state
            self.assertAlmostEqual(
                self.agent_b.state.mood_value,
                mood_b_before_c_intervention,
                delta=0.01,
                msg="AgentB's mood should not change until its own turn to process C's message.",
            )
            self.assertAlmostEqual(
                self.agent_b.state.relationships.get(self.agent_c.agent_id, 0.0),
                relationship_b_to_c_before_c_intervention,
                delta=0.01,
                msg="AgentB's relationship towards C should not change until B processes C's message.",
            )
            self.assertAlmostEqual(
                self.agent_b.state.relationships.get(self.agent_a.agent_id, 0.0),
                relationship_b_to_a_before_c_intervention,
                delta=0.01,
                msg="AgentB's relationship towards A should not be affected by C's broadcast at this point.",
            )

            logger.info("Step 3 (AgentC attempts to mediate) and assertions completed.")

        # --- Step 4: Agent A's Second Turn - Processes Agent B's Disagreement and Agent C's Facilitation ---
        logger.info("Starting Step 4: AgentA's second turn, processes messages...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
            "Simulation did not advance to AgentA's second turn.",
        )

        # Record Agent A's state *before* this turn's processing
        mood_a_before_processing_msgs = self.agent_a.state.mood_value
        relationship_a_to_b_before_processing_msgs = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )
        relationship_a_to_c_before_processing_msgs = self.agent_a.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )
        initial_ip_a_turn2 = self.agent_a.state.ip
        initial_du_a_turn2 = self.agent_a.state.du

        logger.info(
            f"AgentA MoodValue before processing messages (B's disagreement, C's facilitation): {mood_a_before_processing_msgs:.2f}"
        )
        logger.info(
            f"AgentA->B Relationship before processing messages: {relationship_a_to_b_before_processing_msgs:.2f}"
        )
        logger.info(
            f"AgentA->C Relationship before processing messages: {relationship_a_to_c_before_processing_msgs:.2f}"
        )
        logger.info(f"AgentA IP before this turn: {initial_ip_a_turn2}, DU: {initial_du_a_turn2}")

        # Allow Agent A's graph to determine its action.
        # The `analyze_perception_sentiment_node` will process incoming messages.
        await self.simulation.run_step()  # AgentA's second turn

        # Assertion 10: Agent A's Mood Update
        mood_a_after_processing_msgs = self.agent_a.state.mood_value
        logger.info(
            f"AgentA MoodValue after processing messages: {mood_a_after_processing_msgs:.2f}"
        )
        # B's disagreement is negative, C's facilitation is positive.
        # Expectation: Mood might not improve significantly, could decrease, or C's message mitigates B's.
        # This is a nuanced assertion. We'll check if it's not drastically worse than after B's message alone,
        # or if C's message had some positive impact.
        # For simplicity, let's assert it changed, could be up or down.
        # A more specific assertion would require knowing the exact sentiment scores and their impact.
        # Given B's negative message and C's positive one, the mood might be pulled in two directions.
        # We expect the mood to be lower than `mood_a_before_b_disagreement` recorded in Step 2,
        # because B's strong disagreement is a direct negative stimulus.
        # C's message is a general broadcast, its impact on A's mood might be less direct than B's message.
        self.assertNotAlmostEqual(
            mood_a_after_processing_msgs,
            mood_a_before_processing_msgs,
            delta=0.001,
            msg="AgentA's mood should have changed after processing B's disagreement and C's facilitation.",
        )
        # Specifically, after B's strong disagreement, mood should likely be lower than before B's turn.
        # mood_a_before_b_disagreement was recorded in Step 2, prior to B's disagreeing action.
        self.assertLess(
            mood_a_after_processing_msgs,
            mood_a_before_b_disagreement,
            msg="AgentA's mood should generally be lower after processing B's strong disagreement, even with C's facilitation.",
        )

        # Assertion 11: Agent A's Relationship Updates
        relationship_a_to_b_after_processing_msgs = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )
        relationship_a_to_c_after_processing_msgs = self.agent_a.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )

        logger.info(
            f"AgentA->B Relationship after processing messages: {relationship_a_to_b_after_processing_msgs:.2f}"
        )
        logger.info(
            f"AgentA->C Relationship after processing messages: {relationship_a_to_c_after_processing_msgs:.2f}"
        )

        # Agent A perceived B's disagreement. Relationship A->B should decrease.
        self.assertLess(
            relationship_a_to_b_after_processing_msgs,
            relationship_a_to_b_before_processing_msgs,
            msg="AgentA's relationship with AgentB should worsen after B's disagreement.",
        )

        # Agent A perceived C's facilitation. Relationship A->C should improve or stay stable.
        self.assertGreaterEqual(
            relationship_a_to_c_after_processing_msgs,
            relationship_a_to_c_before_processing_msgs,
            msg="AgentA's relationship with AgentC should improve or stay stable after C's facilitation.",
        )

        # Assertion 12: Agent A IP/DU Debit
        # Assert IP change based on action (if any cost). Passive DU generation occurs if not idle.
        # This depends on the action Agent A took. For now, let's check DU increased due to passive generation if not idle.
        final_ip_a_turn2 = self.agent_a.state.ip
        final_du_a_turn2 = self.agent_a.state.du
        du_gen_rate_a_turn2 = config.ROLE_DU_GENERATION.get(self.agent_a.state.role, 1.0)

        logger.info(f"AgentA IP after this turn: {final_ip_a_turn2}, DU: {final_du_a_turn2}")
        # If agent A took an action with IP cost, IP would decrease. Otherwise, it's stable.
        # For DU, if not idle, it should generally increase or stay same (cost vs generation)
        # Assuming a non-idle action was taken due to the conflict.
        if self.agent_a.state.last_action_intent != AgentActionIntent.IDLE.value:
            min_expected_du_a_turn2 = (
                initial_du_a_turn2 + (0.5 * du_gen_rate_a_turn2) - 5.0
            )  # Max possible cost for an action
            max_expected_du_a_turn2 = initial_du_a_turn2 + (1.5 * du_gen_rate_a_turn2)
            # Check if DU is within a plausible range considering generation and potential small cost
            self.assertTrue(
                final_du_a_turn2
                > initial_du_a_turn2
                - 5.0,  # Net DU change should not be a large loss unless very costly action
                f"AgentA DU changed unexpectedly. From {initial_du_a_turn2} to {final_du_a_turn2}",
            )
        else:  # Idle
            expected_du_a_turn2 = initial_du_a_turn2  # No passive generation if idle
            self.assertAlmostEqual(
                final_du_a_turn2,
                expected_du_a_turn2,
                delta=0.1,
                msg=f"AgentA DU should be stable if idle. From {initial_du_a_turn2} to {final_du_a_turn2}",
            )

        logger.info("Step 4 (AgentA processes messages) and assertions completed.")

        # --- Step 5: Agent B's Second Turn - Processes Agent C's Facilitation ---
        logger.info("Starting Step 5: AgentB's second turn, processes C's facilitation...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_b.agent_id,
            "Simulation did not advance to AgentB's second turn.",
        )

        # Record Agent B's state *before* this turn's processing
        mood_b_before_processing_c_msg = self.agent_b.state.mood_value
        relationship_b_to_a_before_processing_c_msg = self.agent_b.state.relationships.get(
            self.agent_a.agent_id, 0.0
        )
        relationship_b_to_c_before_processing_c_msg = self.agent_b.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )
        initial_ip_b_turn2 = self.agent_b.state.ip
        initial_du_b_turn2 = self.agent_b.state.du

        logger.info(
            f"AgentB MoodValue before processing C's facilitation: {mood_b_before_processing_c_msg:.2f}"
        )
        logger.info(
            f"AgentB->A Relationship before processing C's facilitation: {relationship_b_to_a_before_processing_c_msg:.2f}"
        )
        logger.info(
            f"AgentB->C Relationship before processing C's facilitation: {relationship_b_to_c_before_processing_c_msg:.2f}"
        )
        logger.info(f"AgentB IP before this turn: {initial_ip_b_turn2}, DU: {initial_du_b_turn2}")

        # Allow Agent B's graph to determine its action.
        await self.simulation.run_step()  # AgentB's second turn

        # Assertion 13: Agent B's Mood Update
        mood_b_after_processing_c_msg = self.agent_b.state.mood_value
        logger.info(
            f"AgentB MoodValue after processing C's facilitation: {mood_b_after_processing_c_msg:.2f}"
        )
        # Agent C's facilitation message was positive/neutral.
        # Expect mood to improve or stay stable, allowing for slight decay or minor negative if B's action was costly.
        self.assertGreaterEqual(
            mood_b_after_processing_c_msg,
            mood_b_before_processing_c_msg - 0.1,
            msg="AgentB's mood should improve or stay relatively stable after C's positive/neutral facilitation.",
        )

        # Assertion 14: Agent B's Relationship Updates
        relationship_b_to_a_after_processing_c_msg = self.agent_b.state.relationships.get(
            self.agent_a.agent_id, 0.0
        )
        relationship_b_to_c_after_processing_c_msg = self.agent_b.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )

        logger.info(
            f"AgentB->A Relationship after processing C's facilitation: {relationship_b_to_a_after_processing_c_msg:.2f}"
        )
        logger.info(
            f"AgentB->C Relationship after processing C's facilitation: {relationship_b_to_c_after_processing_c_msg:.2f}"
        )

        # B's relationship with A might not change much just from C's message about their dynamic.
        # The main impact was when B sent the disagreement.
        self.assertAlmostEqual(
            relationship_b_to_a_after_processing_c_msg,
            relationship_b_to_a_before_processing_c_msg,
            delta=0.1,
            msg="AgentB's relationship with AgentA should be relatively stable this turn.",
        )

        # B perceived C's facilitation. Relationship B->C should improve or stay stable.
        self.assertGreaterEqual(
            relationship_b_to_c_after_processing_c_msg,
            relationship_b_to_c_before_processing_c_msg,
            msg="AgentB's relationship with AgentC should improve or stay stable after C's facilitation.",
        )

        # Assertion 15: Agent B IP/DU Debit
        final_ip_b_turn2 = self.agent_b.state.ip
        final_du_b_turn2 = self.agent_b.state.du
        du_gen_rate_b_turn2 = config.ROLE_DU_GENERATION.get(self.agent_b.state.role, 1.0)
        logger.info(f"AgentB IP after this turn: {final_ip_b_turn2}, DU: {final_du_b_turn2}")

        if self.agent_b.state.last_action_intent != AgentActionIntent.IDLE.value:
            min_expected_du_b_turn2 = (
                initial_du_b_turn2 + (0.5 * du_gen_rate_b_turn2) - 5.0
            )  # Max possible cost for an action
            self.assertTrue(
                final_du_b_turn2 > initial_du_b_turn2 - 5.0,
                f"AgentB DU changed unexpectedly. From {initial_du_b_turn2} to {final_du_b_turn2}",
            )
        else:  # Idle
            expected_du_b_turn2 = initial_du_b_turn2
            self.assertAlmostEqual(
                final_du_b_turn2,
                expected_du_b_turn2,
                delta=0.1,
                msg=f"AgentB DU should be stable if idle. From {initial_du_b_turn2} to {final_du_b_turn2}",
            )

        logger.info("Step 5 (AgentB processes C's facilitation) and assertions completed.")

        # --- Step 6: Agent A's Third Turn - Responds More Constructively ---
        logger.info("Starting Step 6: AgentA's third turn, responding constructively...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_a.agent_id,
            "Simulation did not advance to AgentA's third turn.",
        )

        # Record Agent A's state *before* this turn's action
        mood_a_before_third_action = self.agent_a.state.mood_value
        relationship_a_to_b_before_third_action = self.agent_a.state.relationships.get(
            self.agent_b.agent_id, 0.0
        )
        relationship_a_to_c_before_third_action = self.agent_a.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )
        initial_ip_a_turn3 = self.agent_a.state.ip
        initial_du_a_turn3 = self.agent_a.state.du

        logger.info(f"AgentA MoodValue before third action: {mood_a_before_third_action:.2f}")
        logger.info(
            f"AgentA->B Relationship before third action: {relationship_a_to_b_before_third_action:.2f}"
        )
        logger.info(
            f"AgentA->C Relationship before third action: {relationship_a_to_c_before_third_action:.2f}"
        )
        logger.info(f"AgentA IP before this turn: {initial_ip_a_turn3}, DU: {initial_du_a_turn3}")

        conciliatory_message_a = (
            "Thank you, AgentC_Conflict, for the perspective. AgentB_Conflict, I understand your concerns about privacy. "
            "Could you elaborate on specific safeguards you'd find acceptable if some transparency is pursued?"
        )

        mock_agent_a_third_action_output = AgentActionOutput(
            thought="I should try to de-escalate and understand AgentB's concerns better, as AgentC suggested.",
            message_content=conciliatory_message_a,
            message_recipient_id=self.agent_b.agent_id,  # Direct question to AgentB
            action_intent=AgentActionIntent.ASK_CLARIFICATION.value,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_a_third_action_output,
        ) as mock_gen_struct_output_a_turn3:
            await self.simulation.run_step()  # AgentA's third turn

            # Assertion 16: AgentA's Message Sent
            mock_gen_struct_output_a_turn3.assert_called_once()
            logger.info(f"AgentA (Turn 3) action mock called. Message: '{conciliatory_message_a}'")

            # Assertion 17: AgentA's Mood Update
            mood_a_after_third_action = self.agent_a.state.mood_value
            logger.info(f"AgentA MoodValue after third action: {mood_a_after_third_action:.2f}")
            # Taking a constructive stance. Mood might stabilize or slightly improve from its previous low.
            # It should ideally be better than mood_a_after_processing_msgs (after B's direct hit and C's general facilitation)
            self.assertGreaterEqual(
                mood_a_after_third_action,
                mood_a_after_processing_msgs
                - 0.05,  # Allow very slight dip if action has minor cost aspect
                "AgentA's mood should stabilize or improve after taking a constructive action.",
            )

            # Assertion 18: AgentA's Relationship Updates
            relationship_a_to_b_after_third_action = self.agent_a.state.relationships.get(
                self.agent_b.agent_id, 0.0
            )
            relationship_a_to_c_after_third_action = self.agent_a.state.relationships.get(
                self.agent_c.agent_id, 0.0
            )

            logger.info(
                f"AgentA->B Relationship after third action: {relationship_a_to_b_after_third_action:.2f}"
            )
            logger.info(
                f"AgentA->C Relationship after third action: {relationship_a_to_c_after_third_action:.2f}"
            )

            # A asked B a constructive question. Relationship A->B should improve or at least not worsen further.
            # The graph logic updates relationship when sending a message.
            self.assertGreater(
                relationship_a_to_b_after_third_action,
                relationship_a_to_b_before_third_action
                - 0.05,  # Allow for tiny negative if sentiment of message is not purely positive
                "AgentA's relationship with AgentB should start to improve or stabilize.",
            )

            # Relationship with C might not change significantly from this specific action towards B,
            # but should remain positive due to C's earlier facilitation.
            self.assertGreaterEqual(
                relationship_a_to_c_after_third_action,
                relationship_a_to_c_before_third_action - 0.05,  # Allow for slight decay
                "AgentA's relationship with AgentC should remain stable or positive.",
            )

            # Assertion 19: AgentA IP/DU Debit
            final_ip_a_turn3 = self.agent_a.state.ip
            final_du_a_turn3 = self.agent_a.state.du
            du_gen_rate_a_turn3 = config.ROLE_DU_GENERATION.get(self.agent_a.state.role, 1.0)
            cost_du_ask_clarification = config.DU_COST_REQUEST_DETAILED_CLARIFICATION

            logger.info(f"AgentA IP after third turn: {final_ip_a_turn3}, DU: {final_du_a_turn3}")

            # IP should be stable for ASK_CLARIFICATION unless graph has specific costs.
            self.assertAlmostEqual(
                final_ip_a_turn3,
                initial_ip_a_turn3,
                delta=0.1,
                msg=f"AgentA IP should be stable. From {initial_ip_a_turn3} to {final_ip_a_turn3}",
            )

            # DU cost for asking clarification + passive generation.
            # Expected DU: initial_du_a_turn3 - cost_du_ask_clarification + passive_generation
            min_expected_du_a_turn3 = (
                initial_du_a_turn3 - cost_du_ask_clarification + (0.5 * du_gen_rate_a_turn3)
            )
            max_expected_du_a_turn3 = (
                initial_du_a_turn3 - cost_du_ask_clarification + (1.5 * du_gen_rate_a_turn3)
            )

            self.assertTrue(
                min_expected_du_a_turn3 <= final_du_a_turn3 <= max_expected_du_a_turn3,
                f"AgentA DU out of range. Got {final_du_a_turn3}, expected {min_expected_du_a_turn3:.2f}-{max_expected_du_a_turn3:.2f}",
            )

        logger.info("Step 6 (AgentA responds constructively) and assertions completed.")

        # --- Step 7: Agent B's Third Turn - Responds to A's Conciliatory Move ---
        logger.info("Starting Step 7: AgentB's third turn, responding constructively to A...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_b.agent_id,
            "Simulation did not advance to AgentB's third turn.",
        )

        # Record Agent B's state *before* this turn's action
        mood_b_before_third_action = self.agent_b.state.mood_value
        relationship_b_to_a_before_third_action = self.agent_b.state.relationships.get(
            self.agent_a.agent_id, 0.0
        )
        relationship_b_to_c_before_third_action = self.agent_b.state.relationships.get(
            self.agent_c.agent_id, 0.0
        )
        initial_ip_b_turn3 = self.agent_b.state.ip
        initial_du_b_turn3 = self.agent_b.state.du

        logger.info(f"AgentB MoodValue before third action: {mood_b_before_third_action:.2f}")
        logger.info(
            f"AgentB->A Relationship before third action: {relationship_b_to_a_before_third_action:.2f}"
        )
        logger.info(
            f"AgentB->C Relationship before third action: {relationship_b_to_c_before_third_action:.2f}"
        )
        logger.info(f"AgentB IP before this turn: {initial_ip_b_turn3}, DU: {initial_du_b_turn3}")

        constructive_response_b = (
            "Thank you for asking, AgentA_Innovator_Conflict. My main concern is the potential for misuse of sensitive strategic information if all comms are public. "
            "However, I'm open to discussing solutions that balance transparency with necessary safeguards, perhaps for specific project discussions?"
        )

        mock_agent_b_third_action_output = AgentActionOutput(
            thought="AgentA is being more reasonable. I should respond constructively and answer their question.",
            message_content=constructive_response_b,
            message_recipient_id=self.agent_a.agent_id,  # Direct response to AgentA
            action_intent=AgentActionIntent.SEND_DIRECT_MESSAGE.value,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_b_third_action_output,
        ) as mock_gen_struct_output_b_turn3:
            await self.simulation.run_step()  # AgentB's third turn

            # Assertion 20: AgentB's Message Sent
            mock_gen_struct_output_b_turn3.assert_called_once()
            logger.info(
                f"AgentB (Turn 3) action mock called. Message: '{constructive_response_b}'"
            )

            # Assertion 21: AgentB's Mood Update
            mood_b_after_third_action = self.agent_b.state.mood_value
            logger.info(f"AgentB MoodValue after third action: {mood_b_after_third_action:.2f}")
            # Responding constructively to A's positive move. Mood should improve or remain stable and positive.
            # It was already somewhat stable after C's facilitation, A's direct positive outreach should help.
            self.assertGreaterEqual(
                mood_b_after_third_action,
                mood_b_before_third_action - 0.05,  # Allow for slight dip due to action itself
                "AgentB's mood should remain stable or improve after A's constructive outreach and B's constructive response.",
            )

            # Assertion 22: AgentB's Relationship Updates
            relationship_b_to_a_after_third_action = self.agent_b.state.relationships.get(
                self.agent_a.agent_id, 0.0
            )
            relationship_b_to_c_after_third_action = self.agent_b.state.relationships.get(
                self.agent_c.agent_id, 0.0
            )

            logger.info(
                f"AgentB->A Relationship after third action: {relationship_b_to_a_after_third_action:.2f}"
            )
            logger.info(
                f"AgentB->C Relationship after third action: {relationship_b_to_c_after_third_action:.2f}"
            )

            # B responded constructively to A. Relationship B->A should improve.
            self.assertGreater(
                relationship_b_to_a_after_third_action,
                relationship_b_to_a_before_third_action,
                "AgentB's relationship with AgentA should improve after constructive exchange.",
            )

            # Relationship with C should remain stable/positive.
            self.assertGreaterEqual(
                relationship_b_to_c_after_third_action,
                relationship_b_to_c_before_third_action - 0.05,  # Allow for slight decay
                "AgentB's relationship with AgentC should remain stable or positive.",
            )

            # Assertion 23: AgentB IP/DU Debit
            final_ip_b_turn3 = self.agent_b.state.ip
            final_du_b_turn3 = self.agent_b.state.du
            du_gen_rate_b_turn3 = config.ROLE_DU_GENERATION.get(self.agent_b.state.role, 1.0)
            # SEND_DIRECT_MESSAGE typically doesn't have a DU cost in the basic graph unless it's very long/complex
            # For simplicity, we assume no specific DU cost for this action beyond passive generation.

            logger.info(f"AgentB IP after third turn: {final_ip_b_turn3}, DU: {final_du_b_turn3}")

            # IP should be stable for SEND_DIRECT_MESSAGE.
            self.assertAlmostEqual(
                final_ip_b_turn3,
                initial_ip_b_turn3,
                delta=0.1,
                msg=f"AgentB IP should be stable. From {initial_ip_b_turn3} to {final_ip_b_turn3}",
            )

            # DU should increase due to passive generation if action intent was not IDLE.
            # Assuming SEND_DIRECT_MESSAGE is not IDLE.
            min_expected_du_b_turn3 = initial_du_b_turn3 + (
                0.5 * du_gen_rate_b_turn3
            )  # No explicit cost assumed for direct message
            max_expected_du_b_turn3 = initial_du_b_turn3 + (1.5 * du_gen_rate_b_turn3)

            self.assertTrue(
                min_expected_du_b_turn3 <= final_du_b_turn3 <= max_expected_du_b_turn3,
                f"AgentB DU out of range. Got {final_du_b_turn3}, expected {min_expected_du_b_turn3:.2f}-{max_expected_du_b_turn3:.2f}",
            )

        logger.info("Step 7 (AgentB responds constructively to A) and assertions completed.")

        # --- Step 8: Agent C's Second Turn - Concluding/Summarizing Broadcast ---
        logger.info("Starting Step 8: AgentC's second turn, concluding summary...")

        self.assertEqual(
            self.simulation.agents[self.simulation.current_agent_index].agent_id,
            self.agent_c.agent_id,
            "Simulation did not advance to AgentC's second turn.",
        )

        initial_ip_c_turn2 = self.agent_c.state.ip
        initial_du_c_turn2 = self.agent_c.state.du
        logger.info(
            f"AgentC initial IP for second turn: {initial_ip_c_turn2}, DU: {initial_du_c_turn2}"
        )

        resolution_summary_c = (
            "It's great to see such productive dialogue emerging! It seems we've found a good path "
            "forward by understanding each other's core points. Well done, AgentA_Innovator_Conflict and AgentB_Analyzer_Conflict."
        )

        mock_agent_c_second_action_output = AgentActionOutput(
            thought="The conflict appears to be resolving. I should acknowledge the positive progress and summarize.",
            message_content=resolution_summary_c,
            message_recipient_id=None,  # Broadcast
            action_intent=AgentActionIntent.CONTINUE_COLLABORATION.value,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

        with patch(
            "src.agents.graphs.basic_agent_graph.generate_structured_output",
            return_value=mock_agent_c_second_action_output,
        ) as mock_gen_struct_output_c_turn2:
            await self.simulation.run_step()  # AgentC's second turn

            # Assertion 24: AgentC's Message Sent
            mock_gen_struct_output_c_turn2.assert_called_once()
            logger.info(f"AgentC (Turn 2) action mock called. Message: '{resolution_summary_c}'")

            # Assertion 25: AgentC IP/DU Debit
            final_ip_c_turn2 = self.agent_c.state.ip
            final_du_c_turn2 = self.agent_c.state.du
            du_gen_rate_c_turn2 = config.ROLE_DU_GENERATION.get(self.agent_c.state.role, 1.0)

            logger.info(f"AgentC IP after second turn: {final_ip_c_turn2}, DU: {final_du_c_turn2}")

            # IP should be stable for CONTINUE_COLLABORATION.
            self.assertAlmostEqual(
                final_ip_c_turn2,
                initial_ip_c_turn2,
                delta=0.1,
                msg=f"AgentC IP should be stable. From {initial_ip_c_turn2} to {final_ip_c_turn2}",
            )

            # DU should increase due to passive generation.
            min_expected_du_c_turn2 = initial_du_c_turn2 + (0.5 * du_gen_rate_c_turn2)
            max_expected_du_c_turn2 = initial_du_c_turn2 + (1.5 * du_gen_rate_c_turn2)

            self.assertTrue(
                min_expected_du_c_turn2 <= final_du_c_turn2 <= max_expected_du_c_turn2,
                f"AgentC DU out of range. Got {final_du_c_turn2}, expected {min_expected_du_c_turn2:.2f}-{max_expected_du_c_turn2:.2f}",
            )

        logger.info("Step 8 (AgentC summarizes resolution) and assertions completed.")

        # --- Final Overall State Assertions (Post-Resolution) ---
        logger.info("Performing Final Overall State Assertions...")

        # Assertion 26: Agent A's Final State
        logger.info(
            f"AgentA final mood: {self.agent_a.state.mood} (value: {self.agent_a.state.mood_value:.2f})"
        )
        logger.info(
            f"AgentA final rel A->B: {self.agent_a.state.relationships.get(self.agent_b.agent_id, 0.0):.2f}"
        )
        logger.info(
            f"AgentA final rel A->C: {self.agent_a.state.relationships.get(self.agent_c.agent_id, 0.0):.2f}"
        )

        self.assertGreaterEqual(
            self.agent_a.state.mood_value,
            -0.1,  # Neutral or slightly positive
            "AgentA's final mood should be neutral or positive after resolution.",
        )
        # Relationship A->B started at 0, dipped, then recovered.
        # It was `relationship_a_to_b_after_third_action` in Step 6. Let's compare to that.
        # It should be better than the lowest point (e.g., relationship_a_to_b_after_processing_msgs)
        # and ideally close to or better than relationship_a_to_b_before_third_action or even initial 0.0
        self.assertGreaterEqual(
            self.agent_a.state.relationships.get(self.agent_b.agent_id, 0.0),
            relationship_a_to_b_before_third_action
            - 0.1,  # Should be at least as good as before its conciliatory move
            "AgentA's final relationship with AgentB should be significantly improved.",
        )
        self.assertGreater(
            self.agent_a.state.relationships.get(self.agent_c.agent_id, 0.0),
            0.0,
            "AgentA's final relationship with AgentC should be positive.",
        )

        # Assertion 27: Agent B's Final State
        logger.info(
            f"AgentB final mood: {self.agent_b.state.mood} (value: {self.agent_b.state.mood_value:.2f})"
        )
        logger.info(
            f"AgentB final rel B->A: {self.agent_b.state.relationships.get(self.agent_a.agent_id, 0.0):.2f}"
        )
        logger.info(
            f"AgentB final rel B->C: {self.agent_b.state.relationships.get(self.agent_c.agent_id, 0.0):.2f}"
        )

        self.assertGreaterEqual(
            self.agent_b.state.mood_value,
            -0.1,  # Neutral or slightly positive
            "AgentB's final mood should be neutral or positive after resolution.",
        )
        # Relationship B->A started at 0, dipped, then recovered.
        # It was `relationship_b_to_a_after_third_action` in Step 7.
        self.assertGreaterEqual(
            self.agent_b.state.relationships.get(self.agent_a.agent_id, 0.0),
            relationship_b_to_a_before_third_action,  # Should be at least as good as before its constructive response
            "AgentB's final relationship with AgentA should be significantly improved.",
        )
        self.assertGreater(
            self.agent_b.state.relationships.get(self.agent_c.agent_id, 0.0),
            0.0,
            "AgentB's final relationship with AgentC should be positive.",
        )

        # Assertion 28: Knowledge Board Content (Optional)
        # Check for the initial controversial post by AgentA and the strong disagreement by AgentB.
        # Other messages (facilitation, conciliatory question, constructive response) might have been direct or broadcast
        # and not necessarily on the KB depending on AgentActionIntent.
        board_entries = self.simulation.knowledge_board.get_entries()
        logger.info(f"Final Knowledge Board Entries: {board_entries}")

        found_a_initial_post = any(
            entry["agent_id"] == self.agent_a.agent_id
            and controversial_idea_content in entry["content"]
            for entry in board_entries
        )
        # Disagreement by B was a broadcast, so should appear as a message perceived by others,
        # but not necessarily a distinct KB "entry" unless CONTINUE_COLLABORATION with content is treated as such by the sim.
        # The current basic_agent_graph `handle_propose_idea_node` adds to KB.
        # `handle_continue_collaboration_node` does not explicitly add to KB.
        # So we'll only check for A's initial post.
        self.assertTrue(
            found_a_initial_post,
            "AgentA's initial controversial idea should be on the Knowledge Board.",
        )

        # Example of checking B's disagreement if it were a KB post:
        # found_b_disagreement_post = any(
        #    entry["agent_id"] == self.agent_b.agent_id and disagreement_content in entry["content"]
        #    for entry in board_entries
        # )
        # self.assertTrue(found_b_disagreement_post, "AgentB's disagreement should be on the Knowledge Board if posted as an idea.")

        logger.info("All conflict resolution steps and final assertions completed.")


# To run this test specifically (requires pytest-asyncio):
# python -m pytest -m integration tests/integration/conflict_resolution/test_conflict_resolution_scenario.py
if __name__ == "__main__":
    # This allows running the test file directly for easier debugging,
    # though pytest is preferred for full test suite runs.
    unittest.main()

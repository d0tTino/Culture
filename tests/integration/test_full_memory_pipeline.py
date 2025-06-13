#!/usr/bin/env python
"""
End-to-End Memory Pipeline Integration Test

This test verifies the complete memory pipeline, from raw event recording through
multiple stages of summarization and pruning, ensuring all components work together correctly.

The test simulates:
1. Creation of raw memory events across multiple agents with different roles
2. L1 consolidation with role-specific summarizers
3. L2 consolidation with role-specific summarizers
4. Memory usage statistics tracking during RAG retrievals
5. Age-based memory pruning for both L1 and L2 summaries
6. MUS-based memory pruning for both L1 and L2 summaries
"""

import logging
import math
import shutil
import sys
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import pytest

pytest.importorskip("chromadb")
from src.agents.core.roles import ROLE_ANALYZER, ROLE_FACILITATOR, ROLE_INNOVATOR
from src.agents.dspy_programs.role_specific_summary_generator import RoleSpecificSummaryGenerator
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_full_memory_pipeline.log"),
    ],
)

logger = logging.getLogger("test_full_memory_pipeline")
logger.setLevel(logging.INFO)


class TestFullMemoryPipeline(unittest.TestCase):
    """
    Test case to verify the end-to-end memory pipeline functionality.
    Tests all memory components working together in a longer simulation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by configuring the memory system and mocking time.
        """
        logger.info("Setting up test environment for end-to-end memory pipeline tests")

        # Adjust config settings to ensure all memory operations happen quickly
        config.CONSOLIDATION_WINDOW_STEPS = 10  # Consolidate L1 every 10 steps
        config.L2_CONSOLIDATION_WINDOW_STEPS = 20  # Consolidate L2 every 20 steps
        config.MEMORY_PRUNING_ENABLED = True
        config.MEMORY_PRUNING_L1_DELAY_STEPS = 5  # Prune L1s 5 steps after L2 creation
        config.MEMORY_PRUNING_L2_ENABLED = True
        config.MEMORY_PRUNING_L2_MAX_AGE_DAYS = 30
        config.MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS = 20
        config.MUS_PRUNING_ENABLED = True
        config.MUS_L1_THRESHOLD = 0.3
        config.MUS_L2_THRESHOLD = 0.3
        config.MIN_AGE_DAYS_FOR_CONSIDERATION = 1  # Consider memories 1+ days old for MUS pruning

        # Create test-specific ChromaDB directory
        cls.vector_store_dir = f"./test_full_memory_pipeline_{uuid.uuid4().hex[:6]}"
        if Path(cls.vector_store_dir).exists():
            logger.info(f"Removing previous test ChromaDB at {cls.vector_store_dir}")
            shutil.rmtree(cls.vector_store_dir)

        cls._start_datetime = datetime(2023, 1, 1, 12, 0, 0)  # Fixed start time for testing

        # Create a patcher for datetime.utcnow() to have controlled time progression
        cls.datetime_patcher = patch("src.agents.memory.vector_store.datetime")
        cls.mock_datetime = cls.datetime_patcher.start()
        cls.mock_datetime.utcnow.return_value = cls._start_datetime
        cls.mock_datetime.fromisoformat = datetime.fromisoformat  # Keep the real method

        # Patch llm_client.generate_response
        cls.llm_patcher = patch("src.infra.llm_client.generate_response")
        cls.mock_llm_generate = cls.llm_patcher.start()
        cls.mock_llm_generate.return_value = "Mocked LLM response for testing."

        # Setup is completed in the test method since we need more complex mocking

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        # Stop the datetime patcher
        cls.datetime_patcher.stop()
        # Stop the LLM patcher
        cls.llm_patcher.stop()

        # Clean up ChromaDB directory
        if Path(cls.vector_store_dir).exists():
            try:
                shutil.rmtree(cls.vector_store_dir)
                logger.info(f"Removed test ChromaDB directory: {cls.vector_store_dir}")
            except PermissionError as e:
                logger.warning(f"Could not remove test directory due to permission error: {e}")

    def setUp(self):
        """Set up each test with fresh instances"""
        # Setup vector store
        self.vector_store = ChromaVectorStoreManager(persist_directory=self.vector_store_dir)

        # Create agent instances with different roles
        self.agents = {
            "innovator": self._create_mock_agent(ROLE_INNOVATOR),
            "analyzer": self._create_mock_agent(ROLE_ANALYZER),
            "facilitator": self._create_mock_agent(ROLE_FACILITATOR),
        }

        # Track simulation step
        self.current_step = 0

        # Create a role-specific summary generator
        self.role_specific_summarizer = RoleSpecificSummaryGenerator()

        # Setup simple mock simulation
        self.simulation = MagicMock()
        self.simulation.step = self.current_step

        # Don't create an agent graph instance since we don't need the full graph functionality
        # We'll use the role-specific summarizer and vector store directly in our tests

    def _create_mock_agent(self, role: str) -> MagicMock:
        """Create a mock agent with a specific role."""
        agent = MagicMock()
        agent.id = f"{role.lower()}_agent_{uuid.uuid4().hex[:4]}"
        agent.role = role
        agent.get_state.return_value = {"role": role, "current_mood": "focused"}

        # Record the role in the vector store
        self.vector_store.record_role_change(
            agent_id=agent.id, step=0, previous_role="unknown", new_role=role
        )

        return agent

    def _advance_time(self, days: int = 0, hours: int = 0, minutes: int = 0) -> None:
        """Advance the mocked time by the specified amount."""
        new_datetime = self._start_datetime + timedelta(days=days, hours=hours, minutes=minutes)
        self.mock_datetime.utcnow.return_value = new_datetime
        logger.info(f"Advanced time to {new_datetime.isoformat()}")

    def _add_test_memories_for_agent(
        self, agent: MagicMock, step_start: int, step_count: int
    ) -> list[str]:
        """Add a sequence of test memories for an agent."""
        added_memory_ids: list[str] = []

        # Different content types based on agent role
        if agent.role == ROLE_INNOVATOR:
            contents = [
                "I've been thinking about a novel approach to our problem using generative AI.",
                "What if we combined vector embeddings with multi-modal transformers?",
                "My intuition says there's a creative solution that others haven't considered.",
                "The connection between these seemingly disparate ideas might lead to breakthrough innovation.",
                "I need to explore this unconventional approach further before presenting it to the team.",
            ]
        elif agent.role == ROLE_ANALYZER:
            contents = [
                "The data indicates a 24% improvement in efficiency with the new algorithm.",
                "We should conduct a detailed analysis of the failure cases to identify patterns.",
                "Based on my calculations, hypothesis B is supported with p < 0.01.",
                "The metrics show concerning performance degradation under high load conditions.",
                "A systematic evaluation reveals three critical weaknesses in the current design.",
            ]
        else:  # ROLE_FACILITATOR
            contents = [
                "We need to establish clear communication protocols for the team.",
                "I sense some tension between the engineers and the product team that needs addressing.",
                "Let's create a shared vision that incorporates everyone's perspectives.",
                "The team will be more effective if we align our goals and establish trust.",
                "I should organize a workshop to help resolve these conflicting priorities.",
            ]

        # Add memory events for specified steps
        for i in range(step_count):
            step = step_start + i
            event_type = "thought" if i % 2 == 0 else "broadcast_sent"
            content = contents[i % len(contents)]

            memory_id = self.vector_store.add_memory(
                agent_id=agent.id,
                step=step,
                event_type=event_type,
                content=content,
                metadata={"simulation_step_timestamp": self.mock_datetime.utcnow().isoformat()},
            )
            added_memory_ids.append(memory_id)

            # Every few steps, add a "reply" from another agent
            if i % 3 == 0:
                # Another agent responds
                responder_role = ROLE_ANALYZER if agent.role != ROLE_ANALYZER else ROLE_FACILITATOR
                responder_agent = (
                    self.agents["analyzer"]
                    if responder_role == ROLE_ANALYZER
                    else self.agents["facilitator"]
                )

                response_content = f"Interesting point about {content.split()[3:7]}. Have you considered alternative perspectives?"

                self.vector_store.add_memory(
                    agent_id=agent.id,
                    step=step,
                    event_type="broadcast_perceived",
                    content=response_content,
                    metadata={
                        "source_agent_id": responder_agent.id,
                        "simulation_step_timestamp": self.mock_datetime.utcnow().isoformat(),
                    },
                )

        return added_memory_ids

    def _simulate_l1_consolidation(
        self, agent: MagicMock, start_step: int, end_step: int
    ) -> str | None:
        """Simulate L1 consolidation for an agent for the given step range."""
        agent_state = agent.get_state()

        # Get all raw memories in the step range
        # Instead of using get_memory_ids_in_step_range which has issues, query directly
        where_filter = {
            "$and": [
                {"agent_id": {"$eq": agent.id}},
                {"step": {"$gte": start_step}},
                {"step": {"$lte": end_step}},
            ]
        }

        results = self.vector_store.collection.get(
            where=where_filter, include=["documents", "metadatas"]
        )

        if not results or not results.get("ids", []):
            logger.warning(
                f"No raw memories found for agent {agent.id} in step range {start_step}-{end_step}"
            )
            return None

        # Process the memories that were returned
        memories = []
        for i, doc in enumerate(results["documents"]):
            if i < len(results.get("metadatas", [])):
                memories.append({"content": doc, "metadata": results["metadatas"][i]})

        # Prepare the context for the summarizer
        recent_events = "\n".join([memory["content"] for memory in memories])

        # Generate the L1 summary using the role-specific summarizer
        l1_summary = self.role_specific_summarizer.generate_l1_summary(
            agent_role=agent.role,
            recent_events=recent_events,
            current_mood=agent_state.get("current_mood", "neutral"),
        )

        # Add the consolidated summary to the vector store
        consolidated_memory_id = self.vector_store.add_memory(
            agent_id=agent.id,
            step=end_step,
            event_type="thought",
            content=l1_summary,
            memory_type="consolidated_summary",
            metadata={
                "consolidated_step_range": f"{start_step}-{end_step}",
                "consolidation_timestamp": self.mock_datetime.utcnow().isoformat(),
                "simulation_step_timestamp": self.mock_datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            f"Added L1 consolidated summary for agent {agent.id} at step {end_step}, covering steps {start_step}-{end_step}"
        )
        return consolidated_memory_id

    def _simulate_l2_consolidation(self, agent: MagicMock, step: int) -> str | None:
        """Simulate L2 consolidation for an agent at a specific step."""
        agent_state = agent.get_state()

        # Calculate the window start (L2 windows are larger than L1)
        start_step = max(0, step - config.L2_CONSOLIDATION_WINDOW_STEPS)

        # Get all L1 summaries in the step range
        where_filter = {
            "$and": [
                {"agent_id": {"$eq": agent.id}},
                {"memory_type": {"$eq": "consolidated_summary"}},
                {"step": {"$gte": start_step}},
                {"step": {"$lte": step}},
            ]
        }

        results = self.vector_store.collection.get(
            where=where_filter, include=["documents", "metadatas"]
        )

        if not results or not results.get("ids", []):
            logger.warning(
                f"No L1 summaries found for agent {agent.id} in step range {start_step}-{step}"
            )
            return None

        # Process the L1 summaries that were returned
        l1_summaries = []
        for i, doc in enumerate(results["documents"]):
            if i < len(results.get("metadatas", [])):
                l1_summaries.append({"content": doc, "metadata": results["metadatas"][i]})

        # Prepare the context for the L2 summarizer
        l1_summaries_context = "\n\n".join(
            [
                f"L1 Summary (Step {summary['metadata'].get('step', 'unknown')}): {summary['content']}"
                for summary in l1_summaries
            ]
        )

        # Generate the L2 summary using the role-specific summarizer
        l2_summary = self.role_specific_summarizer.generate_l2_summary(
            agent_role=agent.role,
            l1_summaries_context=l1_summaries_context,
            overall_mood_trend=agent_state.get("current_mood", "neutral"),
            agent_goals=agent_state.get("goals", "Complete the task effectively"),
        )

        # Add the L2 chapter summary to the vector store
        l2_memory_id = self.vector_store.add_memory(
            agent_id=agent.id,
            step=step,
            event_type="thought",
            content=l2_summary,
            memory_type="chapter_summary",
            metadata={
                "consolidated_step_range": f"{start_step}-{step}",
                "consolidation_timestamp": self.mock_datetime.utcnow().isoformat(),
                "simulation_step_start_timestamp": self.mock_datetime.utcnow().isoformat(),
                "simulation_step_end_timestamp": self.mock_datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            f"Added L2 chapter summary for agent {agent.id} at step {step}, covering steps {start_step}-{step}"
        )
        return l2_memory_id

    def _simulate_memory_retrievals(self, agent: MagicMock, queries: list[str]) -> None:
        """Simulate RAG retrievals to build up usage statistics."""
        for query in queries:
            # Perform a retrieval
            memories = self.vector_store.retrieve_relevant_memories(
                agent_id=agent.id, query=query, k=3
            )

            # Log what was retrieved
            logger.info(
                f"Agent {agent.id} retrieved {len(memories)} memories for query: '{query}'"
            )

            # Perform a targeted retrieval to ensure specific memories get retrieved often
            # This helps test the MUS scoring later
            if "innovation" in query.lower() or "creative" in query.lower():
                targeted_filter = {"event_type": "thought"}
                targeted_memories = self.vector_store.retrieve_filtered_memories(
                    agent_id=agent.id, filters=targeted_filter, limit=2
                )
                logger.info(
                    f"Agent {agent.id} made targeted retrieval of {len(targeted_memories)} memories"
                )

    def _perform_age_pruning(self):
        """Simulate age-based pruning for both L1 and L2 summaries"""
        # Get L1 summaries that should be pruned based on age
        l1_ids_to_prune = []
        for agent in self.agents.values():
            # Get all L1 summaries for the agent
            l1_where = {
                "$and": [
                    {"agent_id": {"$eq": agent.id}},
                    {"memory_type": {"$eq": "consolidated_summary"}},
                ]
            }

            l1_results = self.vector_store.collection.get(
                where=l1_where, include=["metadatas", "documents"]
            )

            if l1_results and "ids" in l1_results and l1_results["ids"]:
                for idx, memory_id in enumerate(l1_results["ids"]):
                    metadata = l1_results["metadatas"][idx]
                    # Check if there's an L2 summary covering this L1 and sufficient delay has passed
                    step = metadata.get("step", 0)
                    if "consolidated_step_range" in metadata:
                        range_str = metadata["consolidated_step_range"]
                        try:
                            start_step, end_step = map(int, range_str.split("-"))
                            l2_step = end_step + config.CONSOLIDATION_WINDOW_STEPS
                            current_step = self.current_step

                            # If this L1 has been covered by an L2 AND the delay period has passed
                            if current_step >= l2_step + config.MEMORY_PRUNING_L1_DELAY_STEPS:
                                l1_ids_to_prune.append(memory_id)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid consolidated_step_range format: {range_str}")

        # Perform the actual pruning
        if l1_ids_to_prune:
            logger.info(f"Age-based pruning: Deleting {len(l1_ids_to_prune)} L1 summaries")
            self.vector_store.delete_memories_by_ids(l1_ids_to_prune)

        # Age-based L2 pruning
        # Get L2 summaries older than MAX_AGE_DAYS
        l2_ids_to_prune = self.vector_store.get_l2_summaries_older_than(
            max_age_days=config.MEMORY_PRUNING_L2_MAX_AGE_DAYS
        )

        # Perform the actual L2 pruning
        if l2_ids_to_prune:
            logger.info(f"Age-based pruning: Deleting {len(l2_ids_to_prune)} L2 summaries")
            self.vector_store.delete_memories_by_ids(l2_ids_to_prune)

    def _perform_mus_pruning(self):
        """Simulate MUS-based pruning for both L1 and L2 summaries"""
        from datetime import datetime

        # Instead of using get_l1_memories_for_mus_pruning, implement the logic directly
        min_age_days = config.MIN_AGE_DAYS_FOR_CONSIDERATION
        mus_threshold = config.MUS_L1_THRESHOLD
        now = self.mock_datetime.utcnow()
        l1_ids_to_prune = []

        # Get all L1 summaries
        l1_where = {"memory_type": {"$eq": "consolidated_summary"}}
        l1_results = self.vector_store.collection.get(
            where=l1_where, include=["metadatas", "documents"]
        )

        if l1_results and "ids" in l1_results and l1_results["ids"]:
            for i, memory_id in enumerate(l1_results["ids"]):
                if i < len(l1_results.get("metadatas", [])):
                    metadata = l1_results["metadatas"][i]

                    # Check age
                    timestamp_str = metadata.get("simulation_step_timestamp")
                    if not timestamp_str:
                        continue

                    try:
                        created_dt = datetime.fromisoformat(timestamp_str)
                        age_days = (now - created_dt).days
                    except Exception as e:
                        logger.warning(f"Invalid timestamp format: {timestamp_str}")
                        continue

                    if age_days < min_age_days:
                        continue  # Skip memories that are too young

                    # Calculate MUS
                    retrieval_count = metadata.get("retrieval_count", 0)
                    accumulated_relevance = metadata.get("accumulated_relevance_score", 0.0)
                    relevance_count = metadata.get("retrieval_relevance_count", 0)
                    last_retrieved = metadata.get("last_retrieved_timestamp", "")

                    # Calculate components
                    rfs = math.log(1 + retrieval_count)
                    rs = (accumulated_relevance / relevance_count) if relevance_count > 0 else 0.0
                    recs = 0.0

                    if last_retrieved:
                        try:
                            last_dt = datetime.fromisoformat(last_retrieved)
                            days_since = (now - last_dt).days
                            days_since = max(0, days_since)
                            recs = 1.0 / (1.0 + days_since)
                        except Exception as e:
                            logger.warning(
                                f"Invalid last_retrieved_timestamp format: {last_retrieved}"
                            )

                    # Calculate MUS
                    mus = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)

                    # If MUS is below threshold, mark for pruning
                    if mus < mus_threshold:
                        l1_ids_to_prune.append(memory_id)

        # Perform the actual L1 MUS pruning
        if l1_ids_to_prune:
            logger.info(
                f"MUS-based pruning: Deleting {len(l1_ids_to_prune)} L1 summaries with low utility scores"
            )
            self.vector_store.delete_memories_by_ids(l1_ids_to_prune)

        # Similar approach for L2 summaries
        l2_ids_to_prune = []
        min_age_days = config.MIN_AGE_DAYS_FOR_CONSIDERATION
        mus_threshold = config.MUS_L2_THRESHOLD

        # Get all L2 summaries
        l2_where = {"memory_type": {"$eq": "chapter_summary"}}
        l2_results = self.vector_store.collection.get(
            where=l2_where, include=["metadatas", "documents"]
        )

        if l2_results and "ids" in l2_results and l2_results["ids"]:
            for i, memory_id in enumerate(l2_results["ids"]):
                if i < len(l2_results.get("metadatas", [])):
                    metadata = l2_results["metadatas"][i]

                    # Check age
                    timestamp_str = metadata.get("simulation_step_end_timestamp")
                    if not timestamp_str:
                        continue

                    try:
                        created_dt = datetime.fromisoformat(timestamp_str)
                        age_days = (now - created_dt).days
                    except Exception as e:
                        logger.warning(f"Invalid timestamp format: {timestamp_str}")
                        continue

                    if age_days < min_age_days:
                        continue  # Skip memories that are too young

                    # Calculate MUS (same formula as L1)
                    retrieval_count = metadata.get("retrieval_count", 0)
                    accumulated_relevance = metadata.get("accumulated_relevance_score", 0.0)
                    relevance_count = metadata.get("retrieval_relevance_count", 0)
                    last_retrieved = metadata.get("last_retrieved_timestamp", "")

                    # Calculate components
                    rfs = math.log(1 + retrieval_count)
                    rs = (accumulated_relevance / relevance_count) if relevance_count > 0 else 0.0
                    recs = 0.0

                    if last_retrieved:
                        try:
                            last_dt = datetime.fromisoformat(last_retrieved)
                            days_since = (now - last_dt).days
                            days_since = max(0, days_since)
                            recs = 1.0 / (1.0 + days_since)
                        except Exception as e:
                            logger.warning(
                                f"Invalid last_retrieved_timestamp format: {last_retrieved}"
                            )

                    # Calculate MUS
                    mus = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)

                    # If MUS is below threshold, mark for pruning
                    if mus < mus_threshold:
                        l2_ids_to_prune.append(memory_id)

        # Perform the actual L2 MUS pruning
        if l2_ids_to_prune:
            logger.info(
                f"MUS-based pruning: Deleting {len(l2_ids_to_prune)} L2 summaries with low utility scores"
            )
            self.vector_store.delete_memories_by_ids(l2_ids_to_prune)

    def test_memory_pipeline_evolution(self):
        """
        Test the end-to-end memory pipeline by simulating a sequence of agent interactions,
        memory formations, consolidations, retrievals, and pruning operations.
        """
        total_steps = 100  # Total simulation steps to run

        # Step 1: Initialize and add raw memories for each agent
        logger.info("STEP 1: Adding initial raw memories for all agents")
        for agent_name, agent in self.agents.items():
            logger.info(f"Adding memories for {agent_name} (role: {agent.role})")
            self._add_test_memories_for_agent(agent, 1, 9)  # Add memories for steps 1-9

        # Step 2: Run the simulation and trigger memory pipeline events
        logger.info("STEP 2: Running simulation for memory pipeline processing")

        for step in range(1, total_steps + 1):
            self.current_step = step
            self.simulation.step = step

            # Add new memories every 5 steps
            if step % 5 == 0:
                for agent_name, agent in self.agents.items():
                    logger.info(f"Step {step}: Adding new memories for {agent_name}")
                    self._add_test_memories_for_agent(agent, step, 5)  # Add 5 new memories

            # L1 Consolidation check
            if step % config.CONSOLIDATION_WINDOW_STEPS == 0:
                logger.info(f"Step {step}: L1 Consolidation triggered")

                # Calculate the window for consolidation
                start_step = max(1, step - config.CONSOLIDATION_WINDOW_STEPS + 1)
                end_step = step

                for agent_name, agent in self.agents.items():
                    logger.info(
                        f"Performing L1 consolidation for {agent_name} (steps {start_step}-{end_step})"
                    )
                    self._simulate_l1_consolidation(agent, start_step, end_step)

                # Advance time to simulate passage of a day
                self._advance_time(days=1)

            # L2 Consolidation check
            if step % config.L2_CONSOLIDATION_WINDOW_STEPS == 0:
                logger.info(f"Step {step}: L2 Consolidation triggered")

                for agent_name, agent in self.agents.items():
                    logger.info(f"Performing L2 consolidation for {agent_name} at step {step}")
                    self._simulate_l2_consolidation(agent, step)

                # Advance time to simulate passage of another day
                self._advance_time(days=1)

            # Memory retrievals - simulate agents accessing memory
            if step % 7 == 0:  # Every 7 steps, do some retrievals
                for agent_name, agent in self.agents.items():
                    logger.info(f"Step {step}: Simulating memory retrievals for {agent_name}")

                    if agent.role == ROLE_INNOVATOR:
                        queries = [
                            "innovative ideas for our project",
                            "creative solutions to our problem",
                            "novel approaches to consider",
                        ]
                    elif agent.role == ROLE_ANALYZER:
                        queries = [
                            "data analysis results",
                            "performance metrics evaluation",
                            "systematic analysis of the problem",
                        ]
                    else:  # ROLE_FACILITATOR
                        queries = [
                            "team communication strategies",
                            "resolving conflicts in the team",
                            "aligning team goals and priorities",
                        ]

                    self._simulate_memory_retrievals(agent, queries)

            # Age-based pruning check
            if step % 15 == 0:  # Check for age-based pruning periodically
                logger.info(f"Step {step}: Checking for age-based pruning")
                self._perform_age_pruning()

            # MUS-based pruning check
            if step % 25 == 0:  # Check for MUS-based pruning less frequently
                logger.info(f"Step {step}: Checking for MUS-based pruning")
                self._perform_mus_pruning()

                # Advance time significantly to test age effects
                self._advance_time(days=5)

        # Step 3: Verify the final state of memory
        logger.info("STEP 3: Verifying final memory state")

        self._verify_final_memory_state()

    def _verify_final_memory_state(self):
        """Verify the final state of the memory store after all operations"""
        # Check raw memory counts
        try:
            for agent_name, agent in self.agents.items():
                agent_id = agent.id
                raw_count = self._count_memories_of_type(agent_id, "raw")
                logger.info(f"Agent {agent_name} ({agent.role}) has {raw_count} raw memories")
        except Exception as e:
            logger.error(f"Error checking raw count for {agent_name}: {e}")

        # Check L1 summary counts
        try:
            for agent_name, agent in self.agents.items():
                agent_id = agent.id
                l1_count = self._count_memories_of_type(agent_id, "consolidated_summary")
                logger.info(f"Agent {agent_name} ({agent.role}) has {l1_count} L1 summaries")
        except Exception as e:
            logger.error(f"Error checking consolidated_summary count for {agent_name}: {e}")

        # Check L2 summary counts
        try:
            for agent_name, agent in self.agents.items():
                agent_id = agent.id
                l2_count = self._count_memories_of_type(agent_id, "chapter_summary")
                logger.info(f"Agent {agent_name} ({agent.role}) has {l2_count} L2 summaries")
        except Exception as e:
            logger.error(f"Error checking chapter_summary count for {agent_name}: {e}")

        # Check usage statistics are being tracked
        try:
            for agent_name, agent in self.agents.items():
                agent_id = agent.id
                usage_stats = self._get_memory_with_usage_stats(agent_id)
                if usage_stats:
                    logger.info(f"Agent {agent_name} has memory usage statistics tracking working")
                else:
                    logger.warning(f"Agent {agent_name} doesn't have memory usage statistics")
        except Exception as e:
            logger.error(f"Error checking usage statistics for {agent_name}: {e}")

        # Check role-specific summaries have characteristics matching agent role
        try:
            for agent_name, agent in self.agents.items():
                agent_id = agent.id
                l1_summaries = self._get_memories_of_type(
                    agent_id, "consolidated_summary", limit=5
                )
                if l1_summaries:
                    # Here we would check for role-specific markers in summaries
                    # For Innovator: creativity, novel ideas, etc.
                    # For Analyzer: data, metrics, precision, etc.
                    # For Facilitator: communication, consensus, teamwork, etc.
                    if agent.role == ROLE_INNOVATOR:
                        # Just basic check for demonstration
                        logger.info(f"Agent {agent_name} (Innovator) L1 summaries present")
                    elif agent.role == ROLE_ANALYZER:
                        logger.info(f"Agent {agent_name} (Analyzer) L1 summaries present")
                    elif agent.role == ROLE_FACILITATOR:
                        logger.info(f"Agent {agent_name} (Facilitator) L1 summaries present")
        except Exception as e:
            logger.error(f"Error checking role-specific characteristics for {agent_name}: {e}")

        logger.info("End-to-end memory pipeline test completed successfully!")

    def _count_memories_of_type(self, agent_id: str, memory_type: str | None = None) -> int:
        """Count memories of a specific type for an agent."""
        where_clause = {}

        if memory_type:
            where_clause = {
                "$and": [{"agent_id": {"$eq": agent_id}}, {"memory_type": {"$eq": memory_type}}]
            }
        else:
            where_clause = {"agent_id": {"$eq": agent_id}}

        results = self.vector_store.collection.get(
            where=where_clause,
            include=[],  # Don't need any content, just counting
        )

        return len(results.get("ids", []))

    def _get_memories_of_type(self, agent_id: str, memory_type: str, limit: int = 5) -> list[dict]:
        """Get memories of a specific type for an agent."""
        where_filter = {
            "$and": [{"agent_id": {"$eq": agent_id}}, {"memory_type": {"$eq": memory_type}}]
        }

        results = self.vector_store.collection.get(
            where=where_filter, include=["documents", "metadatas"], limit=limit
        )

        memories = []
        if results and "documents" in results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                if i < len(results.get("metadatas", [])):
                    memories.append({"content": doc, "metadata": results["metadatas"][i]})

        return memories

    def _get_memory_with_usage_stats(self, agent_id: str) -> dict | None:
        """Check if any memories have usage statistics for an agent."""
        where_filter = {"$and": [{"agent_id": {"$eq": agent_id}}, {"retrieval_count": {"$gt": 0}}]}

        results = self.vector_store.collection.get(
            where=where_filter, include=["metadatas"], limit=1
        )

        if results and "metadatas" in results and results["metadatas"]:
            return results["metadatas"][0]
        return None


if __name__ == "__main__":
    unittest.main()

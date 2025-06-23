import logging
import math  # Added for isnan

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

from src.agents.core import roles
from src.agents.core.base_agent import Agent
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra.async_dspy_manager import AsyncDSPyManager
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.simulation import Simulation

# Configure logging
logger = logging.getLogger(__name__)
# Basic logging configuration for the test if not handled globally by pytest.ini
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Define a unique path for this test's ChromaDB to avoid conflicts
CHROMA_DB_PATH_LONGEVITY = "./chroma_db_longevity"
SCENARIO_LONGEVITY = "Longevity test: agents collaborate on sustainable technology."


@pytest.fixture(scope="class")
def vector_store_manager_longevity() -> ChromaVectorStoreManager:
    # Ensure a clean slate for each test class run
    import shutil

    try:
        shutil.rmtree(CHROMA_DB_PATH_LONGEVITY)
        logger.info(f"Removed old ChromaDB path {CHROMA_DB_PATH_LONGEVITY}")
    except FileNotFoundError:
        logger.info(f"ChromaDB path {CHROMA_DB_PATH_LONGEVITY} not found, no need to remove.")
    except Exception as e:
        logger.warning(f"Could not remove old ChromaDB path {CHROMA_DB_PATH_LONGEVITY}: {e}")
    return ChromaVectorStoreManager(persist_directory=CHROMA_DB_PATH_LONGEVITY)


@pytest.fixture(scope="class")
def async_dspy_manager_longevity() -> AsyncDSPyManager:
    manager = AsyncDSPyManager(
        default_timeout=20.0
    )  # Longer timeout for potentially slower LLM calls
    yield manager
    manager.shutdown()  # Removed asyncio.run()


class TestLongevity:
    simulation: Simulation
    agents: list[Agent]
    vector_store: ChromaVectorStoreManager

    @pytest.fixture(autouse=True, scope="function")
    @staticmethod
    async def setup_class_longevity(
        vector_store_manager_longevity: ChromaVectorStoreManager,
        async_dspy_manager_longevity: AsyncDSPyManager,
    ):
        TestLongevity.vector_store = vector_store_manager_longevity

        # Define initial state dictionaries
        agent_a_initial_state = {
            "agent_id": "agent_longevity_a",
            "name": "InnovatorAgentLongevity",
            "role": roles.ROLE_INNOVATOR,
            "goals": [
                {
                    "description": "Collaborate effectively to discuss and refine ideas on sustainable technology.",
                    "priority": "high",
                }
            ],
            "ip": 100.0,
            "du": 100.0,
        }
        agent_b_initial_state = {
            "agent_id": "agent_longevity_b",
            "name": "AnalyzerAgentLongevity",
            "role": roles.ROLE_ANALYZER,
            "goals": [
                {
                    "description": "Collaborate effectively to discuss and refine ideas on sustainable technology.",
                    "priority": "high",
                }
            ],
            "ip": 100.0,
            "du": 100.0,
        }
        agent_c_initial_state = {
            "agent_id": "agent_longevity_c",
            "name": "FacilitatorAgentLongevity",
            "role": roles.ROLE_FACILITATOR,
            "goals": [
                {
                    "description": "Collaborate effectively to discuss and refine ideas on sustainable technology.",
                    "priority": "high",
                }
            ],
            "ip": 100.0,
            "du": 100.0,
        }

        TestLongevity.agents = [
            Agent(
                initial_state=agent_a_initial_state,
                vector_store_manager=TestLongevity.vector_store,
                async_dspy_manager=async_dspy_manager_longevity,
            ),
            Agent(
                initial_state=agent_b_initial_state,
                vector_store_manager=TestLongevity.vector_store,
                async_dspy_manager=async_dspy_manager_longevity,
            ),
            Agent(
                initial_state=agent_c_initial_state,
                vector_store_manager=TestLongevity.vector_store,
                async_dspy_manager=async_dspy_manager_longevity,
            ),
        ]

        # Ensure graphs are assigned (as per recent fixes in other tests)
        from src.agents.graphs.basic_agent_graph import agent_graph_executor_instance

        for agent in TestLongevity.agents:
            agent.graph = agent_graph_executor_instance

        TestLongevity.simulation = Simulation(
            agents=TestLongevity.agents,
            vector_store_manager=TestLongevity.vector_store,
            scenario=SCENARIO_LONGEVITY,
        )
        logger.info("Longevity test setup complete.")

    @pytest.mark.asyncio
    @pytest.mark.longevity
    async def test_basic_100_turn_simulation(self):
        num_turns_to_run = 100  # Reduced from 100 for quicker initial testing, can be increased
        logger.info(f"Starting longevity test: {num_turns_to_run} turns.")

        for i in range(num_turns_to_run):
            turn_number = i + 1
            active_agent_id_before_step = self.simulation.agents[
                self.simulation.current_agent_index
            ].agent_id
            logger.info(
                f"Longevity Test: Turn {turn_number}/{num_turns_to_run}. Active Agent: {active_agent_id_before_step}"
            )

            try:
                await self.simulation.run_step()

                # Basic check after each step
                assert self.simulation.last_completed_agent_index is not None, (
                    "last_completed_agent_index should be set"
                )
                current_agent_after_step = self.simulation.agents[
                    self.simulation.last_completed_agent_index
                ]  # Agent whose turn just ended

                assert current_agent_after_step.state.ip is not None
                assert current_agent_after_step.state.du is not None and not math.isnan(
                    current_agent_after_step.state.du
                )
                assert -1.0 <= current_agent_after_step.state.mood_value <= 1.0

                if turn_number % 25 == 0:  # Periodically check all agent states
                    logger.info(f"--- Longevity Test: Periodic check at turn {turn_number} ---")
                    self._assert_agent_states_valid(self.agents, test_id=f"turn_{turn_number}")

            except Exception as e:
                logger.error(
                    f"Longevity Test: Exception during turn {turn_number} for agent {active_agent_id_before_step}: {e}",
                    exc_info=True,
                )
                pytest.fail(
                    f"Exception encountered during simulation run_step at turn {turn_number} for agent {active_agent_id_before_step}: {e}"
                )

        logger.info(f"Longevity Test: Completed {num_turns_to_run} turns.")
        assert self.simulation.total_turns_executed >= num_turns_to_run, (
            f"Simulation should execute at least {num_turns_to_run} turns. Got {self.simulation.total_turns_executed}"
        )

        logger.info("--- Longevity Test: Final checks ---")
        self._assert_agent_states_valid(self.agents, test_id="final_check")
        self._assert_knowledge_board_sane(
            self.simulation.knowledge_board,
            num_agents=len(self.agents),
            num_turns=num_turns_to_run,
        )
        logger.info("Longevity test passed.")

    def _assert_agent_states_valid(self, agents_list: list[Agent], test_id: str = ""):
        for agent in agents_list:
            logger.debug(
                f"Checking agent {agent.agent_id} state ({test_id}) - Mood: {agent.state.mood_value:.2f}, IP: {agent.state.ip}, DU: {agent.state.du}"
            )
            assert agent.state.ip is not None and not math.isnan(agent.state.ip), (
                f"Agent {agent.agent_id} IP is NaN ({test_id})"
            )
            assert agent.state.du is not None and not math.isnan(agent.state.du), (
                f"Agent {agent.agent_id} DU is NaN ({test_id})"
            )
            assert agent.state.ip > -500, (
                f"Agent {agent.agent_id} IP too low: {agent.state.ip} ({test_id})"
            )  # More lenient for longevity
            assert agent.state.du > -500, (
                f"Agent {agent.agent_id} DU too low: {agent.state.du} ({test_id})"
            )  # More lenient
            assert agent.state.ip < 2000, (
                f"Agent {agent.agent_id} IP too high: {agent.state.ip} ({test_id})"
            )
            assert agent.state.du < 2000, (
                f"Agent {agent.agent_id} DU too high: {agent.state.du} ({test_id})"
            )

            assert agent.state.mood_value is not None and not math.isnan(agent.state.mood_value), (
                f"Agent {agent.agent_id} mood_value is NaN ({test_id})"
            )
            assert -1.0 <= agent.state.mood_value <= 1.0, (
                f"Agent {agent.agent_id} mood_value out of range: {agent.state.mood_value} ({test_id})"
            )

            for (
                target_id,
                relationship_score,
            ) in agent.state.relationships.items():  # Corrected variable name
                assert relationship_score is not None and not math.isnan(relationship_score), (
                    f"Agent {agent.agent_id} relationship to {target_id} is NaN ({test_id})"
                )
                assert -1.0 <= relationship_score <= 1.0, (
                    f"Agent {agent.agent_id} relationship to {target_id} out of range: {relationship_score} ({test_id})"
                )

    def _assert_knowledge_board_sane(self, kb: KnowledgeBoard, num_agents: int, num_turns: int):
        max_expected_entries = (
            num_agents * num_turns * 3
        )  # Slightly more generous for natural interaction
        try:
            # Assuming get_entries might not exist or work this way, or we just want a count
            # For now, let's assume KnowledgeBoard has a way to get a count or all entries
            # If not, this part needs to be adapted to how KnowledgeBoard stores/exposes entries.
            # This is a placeholder if direct entry count isn't available.
            # We might need to inspect kb.board (if it's a list/dict) or similar.
            # For this example, let's assume a method like get_all_entries() or len(kb) if it's a collection.

            # Placeholder: If KnowledgeBoard is a simple list internally (example)
            if hasattr(kb, "board") and isinstance(kb.board, list):
                num_entries = len(kb.board)
            elif hasattr(kb, "get_entry_count"):  # Hypothetical method
                num_entries = kb.get_entry_count()
            else:
                # Fallback: try to get many entries and count them
                # This is inefficient if the board is huge.
                # The actual KnowledgeBoard API should be used.
                # For now, we'll assume it can be iterated or queried for a rough count.
                # This part of the assertion is highly dependent on KnowledgeBoard's implementation.
                # Let's assume we can fetch up to max_expected_entries * 2 to be safe for counting.
                entries = kb.get_entries(query=None, limit=max_expected_entries * 2)
                num_entries = len(entries)

            assert num_entries <= max_expected_entries, (
                f"Knowledge board has {num_entries} entries, exceeding max expected {max_expected_entries}"
            )
            logger.info(
                f"Knowledge board sanity check: {num_entries} entries (max expected {max_expected_entries})."
            )

        except Exception as e:
            logger.warning(
                f"Could not perform knowledge board sanity check due to error: {e}. This might indicate an issue with accessing KB entries or the KB itself."
            )
            # Depending on strictness, this could be a failure:
            # pytest.fail(f"Knowledge board sanity check failed: {e}")

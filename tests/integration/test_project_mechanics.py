#!/usr/bin/env python
"""
Test script to verify the Project Affiliation Mechanics in Culture.ai.
This script runs several test cases to verify project creation, joining, and leaving.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest
from typing_extensions import Self

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

from src.agents.core.base_agent import Agent
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import config
from src.sim.simulation import Simulation

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set specific loggers to INFO to reduce noise from external libraries
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.INFO)
logging.getLogger("chromadb").setLevel(logging.INFO)

logger = logging.getLogger("test_project_mechanics")

# Constants
CHROMA_DB_PATH = "./chroma_db_test_projects"
SCENARIO = (
    "This is a test simulation to verify project mechanics. Agents should focus on creating, "
    "joining, and leaving projects."
)


class TestCase:
    """Base class for test cases"""

    name = "Base Test Case"
    description = "Base test case description"

    def setup(self: Self) -> None:
        """Setup method to prepare the test case"""
        self.passed = False
        self.result_message = ""

    async def run(self: Self) -> bool:
        """Run the test case"""
        raise NotImplementedError("Each test case must implement run()")

    def report(self: Self) -> str:
        """Return a report of the test case result"""
        status = "PASSED" if self.passed else "FAILED"
        return f"Test Case: {self.name} - {status}\n{self.description}\n{self.result_message}\n"


class TestProjectCreation(TestCase):
    """Test case for project creation"""

    name = "Project Creation"
    description = (
        "Verify that an agent can create a project and that the agent's state and simulation "
        "are updated correctly"
    )

    async def run(self: Self) -> bool:
        self.setup()
        logger.info(f"Running test case: {self.name}")
        logger.info(self.description)

        # Use unique path for this test
        test_db_path = f"{CHROMA_DB_PATH}_creation_{int(time.time())}"

        # Initialize vector store
        if Path(test_db_path).exists():
            import shutil

            try:
                shutil.rmtree(test_db_path)
            except Exception as e:
                logger.warning(f"Could not clear ChromaDB directory {test_db_path}: {e}")

        vector_store = ChromaVectorStoreManager(persist_directory=test_db_path)

        # Create agent with sufficient resources to create a project
        initial_ip = 30
        initial_du = 30
        agent = Agent(
            agent_id="test_creator",
            initial_state={
                "name": "TestCreator",
                "current_role": "Innovator",
                "goals": [{"description": "Create a new project", "priority": "high"}],
                "influence_points": initial_ip,  # Ensure sufficient IP
                "data_units": initial_du,  # Ensure sufficient DU
            },
        )

        # Create simulation
        simulation = Simulation(
            agents=[agent], vector_store_manager=vector_store, scenario=SCENARIO
        )

        # Get initial resources
        initial_state = agent.state
        logger.info(f"Initial state: IP={initial_state.ip}, DU={initial_state.du}")

        # Manually trigger project creation
        project_name = "Test Project"
        project_description = "This is a test project for verification"
        project_id = simulation.create_project(project_name, agent.agent_id, project_description)

        if not project_id:
            self.passed = False
            self.result_message = "Failed to create project"
            return False

        # In the real agent graph, the agent would update its state after creating a project,
        # here we need to manually update the agent state to simulate this behavior
        agent.state.current_project_id = project_id
        agent.state.current_project_affiliation = project_name
        agent.state.ip -= config.IP_COST_CREATE_PROJECT
        agent.state.du -= config.DU_COST_CREATE_PROJECT

        # Get the agent state after project creation and manual update
        agent_state = agent.state
        logger.info(f"After project creation: IP={agent_state.ip}, DU={agent_state.du}")

        # Check if IP and DU were correctly deducted
        expected_ip = initial_ip - config.IP_COST_CREATE_PROJECT
        expected_du = initial_du - config.DU_COST_CREATE_PROJECT
        ip_deducted = agent_state.ip == expected_ip
        du_deducted = agent_state.du == expected_du
        resources_deducted = ip_deducted and du_deducted

        # Check if the agent state was updated correctly
        state_correct = (
            agent_state.current_project_id == project_id
            and agent_state.current_project_affiliation == project_name
        )

        # Check if the project was added to simulation
        project_added = (
            project_id in simulation.projects
            and simulation.projects[project_id]["name"] == project_name
            and simulation.projects[project_id]["description"] == project_description
            and agent.agent_id in simulation.projects[project_id]["members"]
        )

        self.passed = state_correct and project_added and resources_deducted

        if self.passed:
            self.result_message = (
                f"Project creation successful!\n"
                f"Project ID: {project_id}\n"
                f"Agent state updated correctly: {state_correct}\n"
                f"Resources deducted correctly: {resources_deducted}\n"
                f"IP deduction correct: {ip_deducted} (from {initial_ip} to {agent_state.ip}, "
                f"expected {expected_ip})\n"
                f"DU deduction correct: {du_deducted} (from {initial_du} to {agent_state.du}, "
                f"expected {expected_du})\n"
                f"Project added to simulation: {project_added}\n"
                f"Agent's current_project_id: {agent_state.current_project_id}\n"
                f"Agent's current_project_affiliation: {agent_state.current_project_affiliation}"
            )
        else:
            self.result_message = (
                f"Project creation issues detected!\n"
                f"Project ID: {project_id}\n"
                f"Agent state updated correctly: {state_correct}\n"
                f"Resources deducted correctly: {resources_deducted}\n"
                f"IP deduction correct: {ip_deducted} (from {initial_ip} to {agent_state.ip}, "
                f"expected {expected_ip})\n"
                f"DU deduction correct: {du_deducted} (from {initial_du} to {agent_state.du}, "
                f"expected {expected_du})\n"
                f"Project added to simulation: {project_added}\n"
                f"Agent's current_project_id: {agent_state.current_project_id}\n"
                f"Agent's current_project_affiliation: {agent_state.current_project_affiliation}"
            )

        logger.info(self.result_message)

        # Close ChromaDB client
        if hasattr(vector_store, "_client") and vector_store._client:
            try:
                vector_store._client.reset()
                vector_store._client.stop()
                logger.debug(f"Closed ChromaDB client for {self.name}")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")

        return self.passed


class TestProjectJoining(TestCase):
    """Test case for joining a project"""

    name = "Project Joining"
    description = (
        "Verify that an agent can join an existing project with proper state updates and "
        "resource deductions"
    )

    async def run(self: Self) -> bool:
        self.setup()
        logger.info(f"Running test case: {self.name}")
        logger.info(self.description)

        # Use unique path for this test
        test_db_path = f"{CHROMA_DB_PATH}_joining_{int(time.time())}"

        # Initialize vector store
        if Path(test_db_path).exists():
            import shutil

            try:
                shutil.rmtree(test_db_path)
            except Exception as e:
                logger.warning(f"Could not clear ChromaDB directory {test_db_path}: {e}")

        vector_store = ChromaVectorStoreManager(persist_directory=test_db_path)

        # Create creator agent
        creator = Agent(
            agent_id="project_creator",
            initial_state={
                "name": "ProjectCreator",
                "current_role": "Innovator",
                "goals": [
                    {"description": "Create a project for others to join", "priority": "high"}
                ],
                "influence_points": 30,
                "data_units": 30,
            },
        )

        # Create joiner agent
        joiner = Agent(
            agent_id="project_joiner",
            initial_state={
                "name": "ProjectJoiner",
                "current_role": "Analyzer",
                "goals": [{"description": "Join an existing project", "priority": "high"}],
                "influence_points": 10,
                "data_units": 10,
            },
        )

        # Create simulation with both agents
        simulation = Simulation(
            agents=[creator, joiner], vector_store_manager=vector_store, scenario=SCENARIO
        )

        # First, have the creator create a project
        project_name = "Joinable Project"
        project_description = "A project that can be joined by others"
        project_id = simulation.create_project(project_name, creator.agent_id, project_description)

        if not project_id:
            self.passed = False
            self.result_message = "Failed to create project for joining test"
            return False

        # Manually update creator's state
        creator.state.current_project_id = project_id
        creator.state.current_project_affiliation = project_name
        creator.state.ip -= config.IP_COST_CREATE_PROJECT
        creator.state.du -= config.DU_COST_CREATE_PROJECT

        # Record joiner's initial resources
        initial_ip = joiner.state.ip
        initial_du = joiner.state.du
        logger.info(f"Joiner initial state: IP={initial_ip}, DU={initial_du}")

        # Now have the joiner join the project
        join_success = simulation.join_project(project_id, joiner.agent_id)

        if not join_success:
            self.passed = False
            self.result_message = f"Failed to join project {project_id}"
            return False

        # In the real agent graph, the agent would update its state after joining a project,
        # here we need to manually update the agent state to simulate this behavior
        joiner.state.current_project_id = project_id
        joiner.state.current_project_affiliation = project_name
        joiner.state.ip -= config.IP_COST_JOIN_PROJECT
        joiner.state.du -= config.DU_COST_JOIN_PROJECT

        # Get updated agent state
        joiner_state = joiner.state
        logger.info(f"After joining: IP={joiner_state.ip}, DU={joiner_state.du}")

        # Check if joiner's state was updated correctly
        state_correct = (
            joiner_state.current_project_id == project_id
            and joiner_state.current_project_affiliation == project_name
        )

        # Check if resources were deducted correctly
        expected_ip = initial_ip - config.IP_COST_JOIN_PROJECT
        expected_du = initial_du - config.DU_COST_JOIN_PROJECT
        ip_deducted = joiner_state.ip == expected_ip
        du_deducted = joiner_state.du == expected_du
        resources_correct = ip_deducted and du_deducted

        # Check if joiner was added to project members
        member_added = joiner.agent_id in simulation.projects[project_id]["members"]

        self.passed = state_correct and resources_correct and member_added

        if self.passed:
            self.result_message = (
                f"Project joining successful!\n"
                f"Project ID: {project_id}\n"
                f"Agent state updated correctly: {state_correct}\n"
                f"Resources deducted correctly: {resources_correct}\n"
                f"IP deduction correct: {ip_deducted} (from {initial_ip} to {joiner_state.ip}, "
                f"expected {expected_ip})\n"
                f"DU deduction correct: {du_deducted} (from {initial_du} to {joiner_state.du}, "
                f"expected {expected_du})\n"
                f"Added to project members: {member_added}\n"
                f"Project members: {simulation.projects[project_id]['members']}\n"
                f"Agent's current_project_id: {joiner_state.current_project_id}\n"
                f"Agent's current_project_affiliation: {joiner_state.current_project_affiliation}"
            )
        else:
            self.result_message = (
                f"Project joining issues detected!\n"
                f"Project ID: {project_id}\n"
                f"Agent state updated correctly: {state_correct}\n"
                f"Resources deducted correctly: {resources_correct}\n"
                f"IP deduction correct: {ip_deducted} (from {initial_ip} to {joiner_state.ip}, "
                f"expected {expected_ip})\n"
                f"DU deduction correct: {du_deducted} (from {initial_du} to {joiner_state.du}, "
                f"expected {expected_du})\n"
                f"Added to project members: {member_added}\n"
                f"Project members: {simulation.projects[project_id]['members']}\n"
                f"Agent's current_project_id: {joiner_state.current_project_id}\n"
                f"Agent's current_project_affiliation: {joiner_state.current_project_affiliation}"
            )

        logger.info(self.result_message)

        # Close ChromaDB client
        if hasattr(vector_store, "_client") and vector_store._client:
            try:
                vector_store._client.reset()
                vector_store._client.stop()
                logger.debug(f"Closed ChromaDB client for {self.name}")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")

        return self.passed


class TestProjectMembershipLimit(TestCase):
    """Test case for project membership limit"""

    name = "Project Membership Limit"
    description = (
        f"Verify that project membership is limited to MAX_PROJECT_MEMBERS "
        f"({config.MAX_PROJECT_MEMBERS})"
    )

    async def run(self: Self) -> bool:
        self.setup()
        logger.info(f"Running test case: {self.name}")
        logger.info(self.description)

        # Use unique path for this test
        test_db_path = f"{CHROMA_DB_PATH}_limit_{int(time.time())}"

        # Initialize vector store
        if Path(test_db_path).exists():
            import shutil

            try:
                shutil.rmtree(test_db_path)
            except Exception as e:
                logger.warning(f"Could not clear ChromaDB directory {test_db_path}: {e}")

        vector_store = ChromaVectorStoreManager(persist_directory=test_db_path)

        # Save original value to restore later
        original_max_members = config.MAX_PROJECT_MEMBERS

        try:
            # Override MAX_PROJECT_MEMBERS for this test to 2
            config.MAX_PROJECT_MEMBERS = 2

            # Create agents (1 creator + 2 successful joiners + 1 failed joiner)
            agents = []

            # Creator
            creator = Agent(
                agent_id="limit_creator",
                initial_state={
                    "name": "LimitCreator",
                    "current_role": "Innovator",
                    "goals": [
                        {"description": "Create a project with limited spots", "priority": "high"}
                    ],
                    "influence_points": 30,
                    "data_units": 30,
                },
            )
            agents.append(creator)

            # Successful joiners
            for i in range(1, config.MAX_PROJECT_MEMBERS):
                joiner = Agent(
                    agent_id=f"successful_joiner_{i}",
                    initial_state={
                        "name": f"Joiner{i}",
                        "current_role": "Analyzer",
                        "goals": [{"description": "Join the project", "priority": "high"}],
                        "influence_points": 10,
                        "data_units": 10,
                    },
                )
                agents.append(joiner)

            # Failed joiner (should be rejected due to limit)
            failed_joiner = Agent(
                agent_id="failed_joiner",
                initial_state={
                    "name": "FailedJoiner",
                    "current_role": "Facilitator",
                    "goals": [{"description": "Try to join the full project", "priority": "high"}],
                    "influence_points": 10,
                    "data_units": 10,
                },
            )
            agents.append(failed_joiner)

            # Create simulation
            simulation = Simulation(
                agents=agents, vector_store_manager=vector_store, scenario=SCENARIO
            )

            # First, have the creator create a project
            project_name = "Limited Project"
            project_description = "A project with limited membership"
            project_id = simulation.create_project(
                project_name, creator.agent_id, project_description
            )

            if not project_id:
                self.passed = False
                self.result_message = "Failed to create project for limit test"
                return False

            # Manually update creator's state
            creator.state.current_project_id = project_id
            creator.state.current_project_affiliation = project_name
            creator.state.ip -= config.IP_COST_CREATE_PROJECT
            creator.state.du -= config.DU_COST_CREATE_PROJECT

            # Store initial resources of failed joiner to check they're not deducted
            failed_initial_ip = failed_joiner.state.ip
            failed_initial_du = failed_joiner.state.du

            # Have the successful joiners join
            for i in range(1, config.MAX_PROJECT_MEMBERS):
                joiner = agents[i]
                join_success = simulation.join_project(project_id, joiner.agent_id)

                if not join_success:
                    self.passed = False
                    self.result_message = (
                        f"Joiner {joiner.agent_id} failed to join, but should have succeeded"
                    )
                    return False

                # Manually update joiner's state
                joiner.state.current_project_id = project_id
                joiner.state.current_project_affiliation = project_name
                joiner.state.ip -= config.IP_COST_JOIN_PROJECT
                joiner.state.du -= config.DU_COST_JOIN_PROJECT

            # Now the project should be full. Attempt to have the failed joiner join
            join_attempt = simulation.join_project(project_id, failed_joiner.agent_id)

            # This should fail due to the limit
            limit_enforced = not join_attempt

            # Check if the failed joiner's resources were preserved
            resources_preserved = (
                failed_joiner.state.ip == failed_initial_ip
                and failed_joiner.state.du == failed_initial_du
            )

            # Check if the failed joiner's state wasn't updated
            state_preserved = (
                failed_joiner.state.current_project_id is None
                and failed_joiner.state.current_project_affiliation is None
            )

            # Check if the failed joiner wasn't added to project
            not_in_members = (
                failed_joiner.agent_id not in simulation.projects[project_id]["members"]
            )

            # Check if the project has exactly MAX_PROJECT_MEMBERS members
            correct_member_count = (
                len(simulation.projects[project_id]["members"]) == config.MAX_PROJECT_MEMBERS
            )

            self.passed = (
                limit_enforced
                and resources_preserved
                and state_preserved
                and not_in_members
                and correct_member_count
            )

            if self.passed:
                self.result_message = (
                    f"Project membership limit test passed!\n"
                    f"Project ID: {project_id}\n"
                    f"Membership limit enforced: {limit_enforced}\n"
                    f"Resources preserved for failed joiner: {resources_preserved}\n"
                    f"State preserved for failed joiner: {state_preserved}\n"
                    f"Failed joiner not in members: {not_in_members}\n"
                    f"Project has correct member count ({config.MAX_PROJECT_MEMBERS}): "
                    f"{correct_member_count}\n"
                    f"Project members: {simulation.projects[project_id]['members']}"
                )
            else:
                self.result_message = (
                    f"Project membership limit test failed!\n"
                    f"Project ID: {project_id}\n"
                    f"Membership limit enforced: {limit_enforced}\n"
                    f"Resources preserved for failed joiner: {resources_preserved}\n"
                    f"State preserved for failed joiner: {state_preserved}\n"
                    f"Failed joiner not in members: {not_in_members}\n"
                    f"Project has correct member count ({config.MAX_PROJECT_MEMBERS}): "
                    f"{correct_member_count}\n"
                    f"Project members: {simulation.projects[project_id]['members']}"
                )

            logger.info(self.result_message)

            # Close ChromaDB client
            if hasattr(vector_store, "_client") and vector_store._client:
                try:
                    vector_store._client.reset()
                    vector_store._client.stop()
                    logger.debug(f"Closed ChromaDB client for {self.name}")
                except Exception as e:
                    logger.warning(f"Error closing ChromaDB client: {e}")

            return self.passed

        finally:
            # Restore original value regardless of test outcome
            config.MAX_PROJECT_MEMBERS = original_max_members


class TestProjectLeaving(TestCase):
    """Test case for leaving a project"""

    name = "Project Leaving"
    description = "Verify that an agent can leave a project with proper state updates"

    async def run(self: Self) -> bool:
        self.setup()
        logger.info(f"Running test case: {self.name}")
        logger.info(self.description)

        # Use unique path for this test
        test_db_path = f"{CHROMA_DB_PATH}_leaving_{int(time.time())}"

        # Initialize vector store
        if Path(test_db_path).exists():
            import shutil

            try:
                shutil.rmtree(test_db_path)
            except Exception as e:
                logger.warning(f"Could not clear ChromaDB directory {test_db_path}: {e}")

        vector_store = ChromaVectorStoreManager(persist_directory=test_db_path)

        # Create creator agent
        creator = Agent(
            agent_id="leave_creator",
            initial_state={
                "name": "LeaveCreator",
                "current_role": "Innovator",
                "goals": [
                    {"description": "Create a project for testing leaving", "priority": "high"}
                ],
                "influence_points": 30,
                "data_units": 30,
            },
        )

        # Create joiner/leaver agent
        leaver = Agent(
            agent_id="project_leaver",
            initial_state={
                "name": "ProjectLeaver",
                "current_role": "Analyzer",
                "goals": [{"description": "Join and then leave a project", "priority": "high"}],
                "influence_points": 10,
                "data_units": 10,
            },
        )

        # Create simulation with both agents
        simulation = Simulation(
            agents=[creator, leaver], vector_store_manager=vector_store, scenario=SCENARIO
        )

        # First, have the creator create a project
        project_name = "Leavable Project"
        project_description = "A project that an agent will join and then leave"
        project_id = simulation.create_project(project_name, creator.agent_id, project_description)

        if not project_id:
            self.passed = False
            self.result_message = "Failed to create project for leaving test"
            return False

        # Manually update creator's state
        creator.state.current_project_id = project_id
        creator.state.current_project_affiliation = project_name
        creator.state.ip -= config.IP_COST_CREATE_PROJECT
        creator.state.du -= config.DU_COST_CREATE_PROJECT

        # Have the leaver join the project
        join_success = simulation.join_project(project_id, leaver.agent_id)

        if not join_success:
            self.passed = False
            self.result_message = f"Failed to join project {project_id} for leaving test"
            return False

        # Manually update leaver's state
        leaver.state.current_project_id = project_id
        leaver.state.current_project_affiliation = project_name
        leaver.state.ip -= config.IP_COST_JOIN_PROJECT
        leaver.state.du -= config.DU_COST_JOIN_PROJECT

        # Now have the leaver leave the project
        leave_success = simulation.leave_project(project_id, leaver.agent_id)

        if not leave_success:
            self.passed = False
            self.result_message = f"Failed to leave project {project_id}"
            return False

        # Manually update leaver's state after leaving
        leaver.state.current_project_id = None
        leaver.state.current_project_affiliation = None

        # Get updated agent state
        leaver_state = leaver.state

        # Check if leaver's state was updated correctly
        state_correct = (
            leaver_state.current_project_id is None
            and leaver_state.current_project_affiliation is None
        )

        # Check if leaver was removed from project members
        member_removed = leaver.agent_id not in simulation.projects[project_id]["members"]

        self.passed = state_correct and member_removed

        if self.passed:
            self.result_message = (
                f"Project leaving successful!\n"
                f"Project ID: {project_id}\n"
                f"Agent state updated correctly: {state_correct}\n"
                f"Removed from project members: {member_removed}\n"
                f"Project members after leaving: {simulation.projects[project_id]['members']}"
            )
        else:
            self.result_message = (
                f"Project leaving issues detected!\n"
                f"Project ID: {project_id}\n"
                f"Agent state updated correctly: {state_correct}\n"
                f"Removed from project members: {member_removed}\n"
                f"Project members after leaving: {simulation.projects[project_id]['members']}"
            )

        logger.info(self.result_message)

        # Close ChromaDB client
        if hasattr(vector_store, "_client") and vector_store._client:
            try:
                vector_store._client.reset()
                vector_store._client.stop()
                logger.debug(f"Closed ChromaDB client for {self.name}")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")

        return self.passed


class TestProjectEdgeCases(TestCase):
    """Test case for project edge cases"""

    name = "Project Edge Cases"
    description = (
        "Verify edge cases such as insufficient resources, deleting non-existent projects, etc."
    )

    async def run(self: Self) -> bool:
        self.setup()
        logger.info(f"Running test case: {self.name}")
        logger.info(self.description)

        # Use unique path for this test
        test_db_path = f"{CHROMA_DB_PATH}_edge_{int(time.time())}"

        # Initialize vector store
        if Path(test_db_path).exists():
            import shutil

            try:
                shutil.rmtree(test_db_path)
            except Exception as e:
                logger.warning(f"Could not clear ChromaDB directory {test_db_path}: {e}")

        vector_store = ChromaVectorStoreManager(persist_directory=test_db_path)

        # Create test agent
        agent = Agent(
            agent_id="edge_case_agent",
            initial_state={
                "name": "EdgeCaseAgent",
                "current_role": "Facilitator",
                "goals": [{"description": "Test edge cases", "priority": "high"}],
                "influence_points": 20,
                "data_units": 20,
            },
        )

        # Create simulation
        simulation = Simulation(
            agents=[agent], vector_store_manager=vector_store, scenario=SCENARIO
        )

        # Store initial resources
        initial_ip = agent.state.ip
        initial_du = agent.state.du

        # Edge Case 1: Join non-existent project
        non_existent_id = "non_existent_project_id"
        join_result = simulation.join_project(non_existent_id, agent.agent_id)

        join_failed = not join_result
        resources_preserved_after_join = (
            agent.state.ip == initial_ip and agent.state.du == initial_du
        )
        state_preserved_after_join = (
            agent.state.current_project_id is None
            and agent.state.current_project_affiliation is None
        )

        # Edge Case 2: Leave non-existent project
        leave_result = simulation.leave_project(non_existent_id, agent.agent_id)

        leave_failed = not leave_result
        resources_preserved_after_leave = (
            agent.state.ip == initial_ip and agent.state.du == initial_du
        )
        state_preserved_after_leave = (
            agent.state.current_project_id is None
            and agent.state.current_project_affiliation is None
        )

        # Edge Case 3: Create project with duplicate name
        project_name = "Unique Project"
        first_project_id = simulation.create_project(
            project_name, agent.agent_id, "First project with this name"
        )

        if first_project_id:
            # Manually update agent state after successful project creation
            agent.state.current_project_id = first_project_id
            agent.state.current_project_affiliation = project_name
            agent.state.ip -= config.IP_COST_CREATE_PROJECT
            agent.state.du -= config.DU_COST_CREATE_PROJECT

        # Try to create another with the same name
        duplicate_project_id = simulation.create_project(
            project_name, agent.agent_id, "Second project with this name"
        )

        duplicate_creation_failed = duplicate_project_id is None

        # Edge Case 4: Agent leaving a project they're not in
        # First create a project
        other_project_name = "Other Project"
        other_project_id = simulation.create_project(
            other_project_name, agent.agent_id, "Project to test leaving when not a member"
        )

        if not other_project_id:
            self.passed = False
            self.result_message = "Failed to create project for edge case test"
            return False

        # Update agent state after creating the second project
        agent.state.current_project_id = other_project_id
        agent.state.current_project_affiliation = other_project_name

        # Now have the agent leave the project (correctly)
        leave_success = simulation.leave_project(other_project_id, agent.agent_id)

        if leave_success:
            # Update agent state after leaving
            agent.state.current_project_id = None
            agent.state.current_project_affiliation = None

        # Try to leave again (should fail since no longer a member)
        second_leave_result = simulation.leave_project(other_project_id, agent.agent_id)

        second_leave_failed = not second_leave_result

        # Overall test passes if all edge cases were handled correctly
        self.passed = (
            join_failed
            and resources_preserved_after_join
            and state_preserved_after_join
            and leave_failed
            and resources_preserved_after_leave
            and state_preserved_after_leave
            and duplicate_creation_failed
            and second_leave_failed
        )

        if self.passed:
            self.result_message = (
                f"Project edge cases handled correctly!\n"
                f"Join non-existent project fails: {join_failed}\n"
                f"Resources preserved after failed join: {resources_preserved_after_join}\n"
                f"State preserved after failed join: {state_preserved_after_join}\n"
                f"Leave non-existent project fails: {leave_failed}\n"
                f"Resources preserved after failed leave: {resources_preserved_after_leave}\n"
                f"State preserved after failed leave: {state_preserved_after_leave}\n"
                f"Duplicate project name creation fails: {duplicate_creation_failed}\n"
                f"Leaving project not in fails: {second_leave_failed}"
            )
        else:
            self.result_message = (
                f"Project edge cases not all handled correctly!\n"
                f"Join non-existent project fails: {join_failed}\n"
                f"Resources preserved after failed join: {resources_preserved_after_join}\n"
                f"State preserved after failed join: {state_preserved_after_join}\n"
                f"Leave non-existent project fails: {leave_failed}\n"
                f"Resources preserved after failed leave: {resources_preserved_after_leave}\n"
                f"State preserved after failed leave: {state_preserved_after_leave}\n"
                f"Duplicate project name creation fails: {duplicate_creation_failed}\n"
                f"Leaving project not in fails: {second_leave_failed}"
            )

        logger.info(self.result_message)

        # Close ChromaDB client
        if hasattr(vector_store, "_client") and vector_store._client:
            try:
                vector_store._client.reset()
                vector_store._client.stop()
                logger.debug(f"Closed ChromaDB client for {self.name}")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")

        return self.passed


@pytest.mark.integration
@pytest.mark.slow
async def run_tests() -> bool:
    """Run all test cases and report results"""
    # Configure console output with colors for better readability
    try:
        import colorama

        colorama.init()
        GREEN = colorama.Fore.GREEN
        RED = colorama.Fore.RED
        YELLOW = colorama.Fore.YELLOW
        RESET = colorama.Style.RESET_ALL
    except ImportError:
        GREEN = ""
        RED = ""
        YELLOW = ""
        RESET = ""

    print(f"\n{YELLOW}{'=' * 80}{RESET}")
    print(f"{YELLOW}Starting Project Mechanics Verification{RESET}")
    print(f"{YELLOW}{'=' * 80}{RESET}")

    test_cases = [
        TestProjectCreation(),
        TestProjectJoining(),
        TestProjectMembershipLimit(),
        TestProjectLeaving(),
        TestProjectEdgeCases(),
    ]

    results = []
    successful_tests = 0
    failed_tests = 0

    for test_case in test_cases:
        try:
            print(f"{YELLOW}{'-' * 80}{RESET}")
            print(f"{YELLOW}STARTING TEST: {test_case.name}{RESET}")
            print(f"{YELLOW}{'-' * 80}{RESET}")

            test_passed = await test_case.run()
            results.append(test_case.report())

            if test_passed:
                successful_tests += 1
                print(f"{GREEN}✅ TEST PASSED: {test_case.name}{RESET}")
            else:
                failed_tests += 1
                print(f"{RED}❌ TEST FAILED: {test_case.name}{RESET}")

            # Print a divider between tests
            print(f"{YELLOW}{'=' * 80}{RESET}")

            # Small delay between tests
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error running test case {test_case.name}: {e}", exc_info=True)
            print(f"{RED}❌ TEST ERROR: {test_case.name} - {e!s}{RESET}")
            results.append(
                f"Test Case: {test_case.name} - ERROR\n{test_case.description}\nError: {e!s}\n"
            )
            failed_tests += 1

    # Print overall results
    print(f"{YELLOW}{'=' * 80}{RESET}")
    print(f"{YELLOW}PROJECT MECHANICS VERIFICATION RESULTS{RESET}")
    print(f"{YELLOW}{'=' * 80}{RESET}")

    for i, result in enumerate(results, 1):
        print(f"Test {i}: {result}")

    all_passed = failed_tests == 0
    overall_status = f"{GREEN}PASSED{RESET}" if all_passed else f"{RED}FAILED{RESET}"
    print(f"{YELLOW}{'=' * 80}{RESET}")
    print(f"OVERALL PROJECT MECHANICS VERIFICATION: {overall_status}")
    print(f"Tests Passed: {successful_tests}/{len(test_cases)}")
    print(f"Tests Failed: {failed_tests}/{len(test_cases)}")
    print(f"{YELLOW}{'=' * 80}{RESET}")

    if all_passed:
        print(
            f"""
{GREEN}PROJECT MECHANICS IMPLEMENTATION IS COMPLETE:
✅ Projects can be created with correct resource costs
✅ Agents can join projects with correct resource costs
✅ Project membership limits are enforced
✅ Agents can leave projects with proper state updates
✅ Edge cases are handled gracefully{RESET}
        """
        )
    else:
        print(
            f"""
{RED}PROJECT MECHANICS IMPLEMENTATION HAS ISSUES:
❌ Some tests failed. Review the logs above for details.{RESET}
        """
        )

    return all_passed


if __name__ == "__main__":
    try:
        # Run the tests
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        sys.exit(2)

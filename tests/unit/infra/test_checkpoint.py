import os
import random

import pytest

from src.app import create_simulation
from src.infra import config
from src.infra.checkpoint import (
    load_checkpoint,
    restore_environment,
    restore_rng_state,
    save_checkpoint,
)
from src.sim.graph_knowledge_board import GraphKnowledgeBoard

pytestmark = pytest.mark.unit


def test_checkpoint_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLE_DU_GENERATION", '{"A":1, "B":1}')
    monkeypatch.setenv("CULTURE_CUSTOM_VAR", "xyz")
    random.seed(1234)
    sim = create_simulation(num_agents=1, steps=1, scenario="test")

    _ = random.random()
    chk = tmp_path / "sim.pkl"
    save_checkpoint(sim, chk)
    expected_next = random.random()

    random.seed(999)
    loaded, meta = load_checkpoint(chk)
    restore_rng_state(meta["rng_state"])
    restore_environment(meta["environment"])

    assert os.environ["ROLE_DU_GENERATION"] == '{"A":1, "B":1}'
    assert "random" in meta["rng_state"]

    assert random.random() == expected_next
    assert loaded.agents[0].state.current_role.name == sim.agents[0].state.current_role.name


def test_deterministic_replay_multiple_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLE_DU_GENERATION", '{"A":1, "B":1}')
    random.seed(42)
    sim = create_simulation(num_agents=1, steps=1, scenario="test")
    chk = tmp_path / "sim.pkl"
    save_checkpoint(sim, chk)
    expected_val = random.random()

    for _ in range(2):
        random.seed(100)
        loaded, meta = load_checkpoint(chk)
        restore_rng_state(meta["rng_state"])
        restore_environment(meta["environment"])
        assert "random" in meta["rng_state"]
        assert random.random() == expected_val


def test_numpy_rng_restore(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("ROLE_DU_GENERATION", '{"A":1, "B":1}')
    random.seed(777)
    np.random.seed(888)
    sim = create_simulation(num_agents=1, steps=1, scenario="test")

    _ = np.random.random()
    chk = tmp_path / "sim.pkl"
    save_checkpoint(sim, chk)
    expected_next = np.random.random()

    np.random.seed(999)
    loaded, meta = load_checkpoint(chk)
    restore_rng_state(meta["rng_state"])
    restore_environment(meta["environment"])

    assert "numpy" in meta["rng_state"]
    assert np.random.random() == expected_next
    assert loaded.agents[0].state.current_role.name == sim.agents[0].state.current_role.name


def test_checkpoint_preserves_board_and_collective_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLE_DU_GENERATION", '{"A":1, "B":1}')
    sim = create_simulation(num_agents=1, steps=1, scenario="test")

    sim.collective_ip = 12.34
    sim.collective_du = 56.78
    sim.knowledge_board.add_entry("hello", sim.agents[0].agent_id, step=0)

    chk = tmp_path / "sim.pkl"
    save_checkpoint(sim, chk)

    loaded, _ = load_checkpoint(chk)

    assert loaded.collective_ip == pytest.approx(12.34)
    assert loaded.collective_du == pytest.approx(56.78)
    assert loaded.knowledge_board.get_full_entries() == sim.knowledge_board.get_full_entries()


def test_checkpoint_loads_graph_board(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLE_DU_GENERATION", '{"A":1, "B":1}')
    monkeypatch.setenv("KNOWLEDGE_BOARD_BACKEND", "graph")
    from tests.integration.knowledge_board.test_graph_backend import DummyDriver

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *a, **k: DummyDriver())
    config.load_config()
    sim = create_simulation(num_agents=1, steps=1, scenario="test")
    sim.knowledge_board.add_entry("hello", sim.agents[0].agent_id, step=0)

    chk = tmp_path / "sim.pkl"
    save_checkpoint(sim, chk)

    loaded, _ = load_checkpoint(chk)

    assert isinstance(loaded.knowledge_board, GraphKnowledgeBoard)
    assert loaded.knowledge_board.get_state() == ["Step 0 (Agent: agent_1): hello"]

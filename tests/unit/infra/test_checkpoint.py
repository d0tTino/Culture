import os
import random

import pytest

from src.app import create_simulation
from src.infra.checkpoint import (
    load_checkpoint,
    restore_environment,
    restore_rng_state,
    save_checkpoint,
)

pytestmark = pytest.mark.unit


def test_checkpoint_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLE_DU_GENERATION", '{"A":1, "B":1}')
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
    assert loaded.agents[0].state.current_role == sim.agents[0].state.current_role


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
    assert loaded.agents[0].state.current_role == sim.agents[0].state.current_role

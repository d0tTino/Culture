# mypy: ignore-errors
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

sys.modules.setdefault(
    "src.infra.llm_client",
    SimpleNamespace(
        LLMClient=object,
        LLMClientConfig=object,
        get_ollama_client=lambda: None,
        generate_text=lambda *a, **k: "",
        summarize_memory_context=lambda *a, **k: "",
        client=SimpleNamespace(),
    ),
)
sys.modules.setdefault(
    "ollama",
    SimpleNamespace(
        Client=object,
        list=lambda: [],
        pull=lambda *a, **k: None,
        show=lambda *a, **k: {},
        chat=lambda *a, **k: {},
        generate=lambda *a, **k: {},
    ),
)
sys.modules.setdefault(
    "weaviate",
    SimpleNamespace(
        classes=SimpleNamespace(),
    ),
)
sys.modules.setdefault("weaviate.classes", SimpleNamespace())
sys.modules.setdefault("neo4j", SimpleNamespace(Driver=object, GraphDatabase=object))

from src.agents.core.agent_state import AgentActionIntent
from src.governance.law_board import LawBoard
from src.infra import config
from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 1.0
    du: float = 0.0
    age: int = 0
    is_alive: bool = True
    inheritance: float = 0.0
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    current_role: str = "dummy"
    steps_in_current_role: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {}

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state


class MoveAgent(DummyAgent):
    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {
            "action_intent": AgentActionIntent.MOVE.value,
            "map_action": {"action": "move", "dx": 1, "dy": 0},
        }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_law_persisted(monkeypatch: pytest.MonkeyPatch, tmp_path):
    board = LawBoard(tmp_path / "laws.sqlite")
    import importlib

    lb_mod = importlib.import_module("src.governance.law_board")
    voting_mod = importlib.import_module("src.governance.voting")
    policy_mod = importlib.import_module("src.governance.policy")

    monkeypatch.setattr(lb_mod, "law_board", board)
    monkeypatch.setattr(voting_mod, "law_board", board)
    monkeypatch.setattr(policy_mod, "law_board", board)
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr("src.utils.policy.evaluate_with_opa", allow)

    agents = [DummyAgent("a1"), DummyAgent("a2"), DummyAgent("a3")]
    sim = Simulation(agents=agents)
    approved = await sim.propose_law("a1", "no moving")
    assert approved is True
    assert "no moving" in board.get_laws()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_action_checked_against_laws(monkeypatch: pytest.MonkeyPatch, tmp_path):
    board = LawBoard(tmp_path / "laws.sqlite")
    board.add_law("no moving")
    import importlib

    lb_mod = importlib.import_module("src.governance.law_board")
    voting_mod = importlib.import_module("src.governance.voting")
    policy_mod = importlib.import_module("src.governance.policy")

    monkeypatch.setattr(lb_mod, "law_board", board)
    monkeypatch.setattr(voting_mod, "law_board", board)
    monkeypatch.setattr(policy_mod, "law_board", board)
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")

    payloads = []

    def fake_post(url: str, json: dict, timeout: int = 2):
        payloads.append(json)

        class Res:
            def json(self_inner):
                return {"result": {"allow": False}}

        return Res()

    monkeypatch.setattr("src.governance.policy.requests.post", fake_post)

    agent = MoveAgent("a1")
    sim = Simulation(agents=[agent])
    await sim.run_step()

    pos = sim.world_map.agent_positions.get("a1")
    assert pos == (0, 0)
    assert payloads and payloads[0]["input"].get("laws") == ["no moving"]

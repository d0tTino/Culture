import pytest

from src.agents.council import (
    CouncilConfig,
    CouncilMemberConfig,
    CouncilOrchestrator,
    CouncilQuestion,
)
from src.infra import config as infra_config
from src.infra import llm_client
from src.infra import metrics as infra_metrics
from src.sim import resource_manager as resource_manager_module


class FakeLedger:
    def __init__(self) -> None:
        self.log_entries: list[tuple[str, float, str]] = []
        self.du_budgets: dict[str, float] = {}

    def calculate_gas_price(self, agent_id: str, window: int = 10) -> tuple[float, float]:
        return (1.0, 0.0)

    def log_change(
        self,
        agent_id: str,
        delta_ip: float = 0.0,
        delta_du: float = 0.0,
        reason: str = "",
        gas_price_per_call: float | None = None,
        gas_price_per_token: float | None = None,
    ) -> None:
        self.log_entries.append((agent_id, float(delta_du), reason))

    def record_du_budget(self, agent_id: str, remaining: float) -> None:
        self.du_budgets[agent_id] = float(remaining)


class FakeResourceManager:
    def __init__(self, ledger: FakeLedger) -> None:
        self.ledger = ledger
        self.du_budgets: dict[str, float] = {}
        self.initial_budgets: dict[str, float] = {}
        self.ensure_calls: list[tuple[str, float]] = []
        self.charge_calls: list[tuple[str, float]] = []
        self.set_calls: list[tuple[str, float]] = []

    def set_du_budget(self, agent_id: str, budget: float) -> None:
        self.du_budgets[agent_id] = float(budget)
        self.initial_budgets.setdefault(agent_id, float(budget))
        self.set_calls.append((agent_id, float(budget)))
        self.ledger.record_du_budget(agent_id, budget)

    def has_du_budget(self, agent_id: str) -> bool:
        return agent_id in self.du_budgets

    def reserve_du_budget(self, agent_id: str, amount: float, *, reason: str = "du_reserve") -> float:
        reserve = float(amount)
        self.charge_du(agent_id, reserve)
        self.ledger.log_change(agent_id, 0.0, -reserve, reason)
        return self.get_du_budget(agent_id)

    def ensure_du_budget(self, agent_id: str, amount: float) -> None:
        self.ensure_calls.append((agent_id, float(amount)))
        remaining = self.du_budgets.get(agent_id, 0.0)
        if amount > remaining:
            raise RuntimeError("Insufficient DU")

    def charge_du(self, agent_id: str, amount: float) -> None:
        self.charge_calls.append((agent_id, float(amount)))
        self.ensure_du_budget(agent_id, amount)
        updated = self.du_budgets.get(agent_id, 0.0) - amount
        self.du_budgets[agent_id] = updated
        self.ledger.record_du_budget(agent_id, updated)

    def get_du_budget(self, agent_id: str) -> float:
        return float(self.du_budgets.get(agent_id, 0.0))


def _sum_du_debits(ledger: FakeLedger, *, agent_id: str) -> float:
    return abs(
        sum(delta_du for logged_agent_id, delta_du, _ in ledger.log_entries if logged_agent_id == agent_id)
    )


def _build_council_config_with_two_members(*, member_budget: float = 2.0) -> CouncilConfig:
    return CouncilConfig(
        members=[
            CouncilMemberConfig(
                member_id="member-a",
                display_name="Member A",
                role="Analyzer",
                description="",
                system_prompt="",
                decision_weight=1.0,
                persona="Curious analyst persona",
                model="mistral:latest",
                temperature=0.1,
                max_tokens=32,
                is_active=True,
            ),
            CouncilMemberConfig(
                member_id="member-b",
                display_name="Member B",
                role="Generalist",
                description="",
                system_prompt="",
                decision_weight=1.0,
                persona="Helpful generalist persona",
                model="mistral:latest",
                temperature=0.1,
                max_tokens=32,
                is_active=True,
            ),
        ],
        voting_mode="judge_llm",
        enabled=True,
        du_budget_per_question=member_budget,
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_council_orchestrator_charges_du_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(infra_config, "_CONFIG", {"USE_COUNCIL_MODE": True})

    fake_ledger = FakeLedger()
    fake_resource_manager = FakeResourceManager(fake_ledger)

    monkeypatch.setattr(resource_manager_module, "_resource_manager", fake_resource_manager)
    monkeypatch.setattr(
        resource_manager_module, "get_resource_manager", lambda: fake_resource_manager
    )
    monkeypatch.setattr(
        "src.agents.council.orchestrator.get_resource_manager", lambda: fake_resource_manager
    )

    monkeypatch.setattr(llm_client, "ledger", fake_ledger)
    monkeypatch.setattr(infra_metrics, "ledger", fake_ledger)

    llm_client.enable_mock_mode(
        True,
        {
            "MemberResponseModel": {
                "answer": "stubbed",
                "reasoning": "",
                "confidence": 1.0,
                "citations": [],
            }
        },
    )

    orchestrator = CouncilOrchestrator()
    config = CouncilConfig(
        members=[
            CouncilMemberConfig(
                member_id="member-a",
                display_name="Member A",
                role="Analyzer",
                description="",
                system_prompt="",
                decision_weight=1.0,
                persona="Curious analyst persona",
                model="mistral:latest",
                temperature=0.1,
                max_tokens=32,
                is_active=True,
            ),
            CouncilMemberConfig(
                member_id="member-b",
                display_name="Member B",
                role="Generalist",
                description="",
                system_prompt="",
                decision_weight=1.0,
                persona="Helpful generalist persona",
                model="mistral:latest",
                temperature=0.1,
                max_tokens=32,
                is_active=True,
            ),
        ],
        voting_mode="judge_llm",
        enabled=True,
        du_budget_per_question=2.0,
    )
    question = CouncilQuestion(
        question_id="q1", prompt="What is DU?", user_id="question-owner", context="", task_context=None
    )

    context = orchestrator._resolve_context(config)
    outcome = await orchestrator.adeliberate(context, question)

    assert outcome.answers
    assert not outcome.metadata.get("du_exhausted", False)

    for member_id in ("member-a", "member-b"):
        initial_budget = fake_resource_manager.initial_budgets[member_id]
        member_charges = [
            amt for aid, amt in fake_resource_manager.charge_calls if aid == member_id
        ]
        remaining_budget = initial_budget - sum(member_charges)

        assert initial_budget == pytest.approx(2.0)
        assert fake_resource_manager.du_budgets[member_id] == pytest.approx(remaining_budget)
        assert fake_ledger.du_budgets[member_id] == pytest.approx(remaining_budget)
        assert member_charges, "Expected DU charges for each council member"

    assert len(fake_resource_manager.set_calls) == 3
    assert fake_resource_manager.set_calls[0] == ("question-owner", pytest.approx(2.0))
    assert fake_resource_manager.set_calls[1:] == [
        ("member-a", pytest.approx(1.0)),
        ("member-b", pytest.approx(1.0)),
    ]
    assert len(fake_resource_manager.ensure_calls) == 2 * len(fake_resource_manager.charge_calls)
    assert outcome.metadata["metrics"]["du_owner_id"] == "question-owner"
    assert outcome.metadata["metrics"]["du_owner_reserved"] == pytest.approx(2.0)

    recorded_du_spend = [entry for entry in fake_ledger.log_entries if entry[2] == "llm_gas"]
    assert len(recorded_du_spend) == len(fake_resource_manager.charge_calls)
    assert all(delta_du == -1.0 for _, delta_du, _ in recorded_du_spend)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_council_orchestrator_honors_per_member_du_budget_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(infra_config, "_CONFIG", {"USE_COUNCIL_MODE": True})

    fake_ledger = FakeLedger()
    fake_resource_manager = FakeResourceManager(fake_ledger)

    monkeypatch.setattr(resource_manager_module, "_resource_manager", fake_resource_manager)
    monkeypatch.setattr(
        resource_manager_module, "get_resource_manager", lambda: fake_resource_manager
    )
    monkeypatch.setattr(
        "src.agents.council.orchestrator.get_resource_manager", lambda: fake_resource_manager
    )

    monkeypatch.setattr(llm_client, "ledger", fake_ledger)
    monkeypatch.setattr(infra_metrics, "ledger", fake_ledger)

    llm_client.enable_mock_mode(
        True,
        {
            "MemberResponseModel": {
                "answer": "stubbed",
                "reasoning": "",
                "confidence": 1.0,
                "citations": [],
            }
        },
    )

    orchestrator = CouncilOrchestrator()
    config = CouncilConfig(
        members=[
            CouncilMemberConfig(
                member_id="member-a",
                display_name="Member A",
                role="Analyzer",
                description="",
                system_prompt="",
                decision_weight=1.0,
                persona="Curious analyst persona",
                model="mistral:latest",
                temperature=0.1,
                max_tokens=32,
                du_budget=3.0,
                is_active=True,
            ),
            CouncilMemberConfig(
                member_id="member-b",
                display_name="Member B",
                role="Generalist",
                description="",
                system_prompt="",
                decision_weight=1.0,
                persona="Helpful generalist persona",
                model="mistral:latest",
                temperature=0.1,
                max_tokens=32,
                is_active=True,
            ),
        ],
        voting_mode="judge_llm",
        enabled=True,
        du_budget_per_question=2.0,
    )
    question = CouncilQuestion(
        question_id="q1", prompt="What is DU?", user_id="question-owner", context="", task_context=None
    )

    context = orchestrator._resolve_context(config)
    outcome = await orchestrator.adeliberate(context, question)

    assert outcome.answers
    metrics = outcome.metadata.get("metrics", {}) if outcome.metadata else {}
    assert metrics.get("du_budget_per_member") == {
        "member-a": pytest.approx(1.0),
        "member-b": pytest.approx(1.0),
    }
    assert fake_resource_manager.initial_budgets == {
        "question-owner": pytest.approx(2.0),
        "member-a": pytest.approx(1.0),
        "member-b": pytest.approx(1.0),
    }
    assert fake_resource_manager.set_calls == [
        ("question-owner", pytest.approx(2.0)),
        ("member-a", pytest.approx(1.0)),
        ("member-b", pytest.approx(1.0)),
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_council_orchestrator_blocks_member_launch_when_owner_du_budget_is_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(infra_config, "_CONFIG", {"USE_COUNCIL_MODE": True})

    fake_ledger = FakeLedger()
    fake_resource_manager = FakeResourceManager(fake_ledger)

    monkeypatch.setattr(resource_manager_module, "_resource_manager", fake_resource_manager)
    monkeypatch.setattr(resource_manager_module, "get_resource_manager", lambda: fake_resource_manager)
    monkeypatch.setattr("src.agents.council.orchestrator.get_resource_manager", lambda: fake_resource_manager)

    monkeypatch.setattr(llm_client, "ledger", fake_ledger)
    monkeypatch.setattr(infra_metrics, "ledger", fake_ledger)

    llm_client.enable_mock_mode(
        True,
        {
            "MemberResponseModel": {
                "answer": "stubbed",
                "reasoning": "",
                "confidence": 1.0,
                "citations": [],
            }
        },
    )

    orchestrator = CouncilOrchestrator()
    config = _build_council_config_with_two_members(member_budget=2.0)
    question = CouncilQuestion(
        question_id="q-owner-insufficient",
        prompt="What is DU?",
        context="",
        metadata={"agent_id": "owner-1", "owner_du_budget": 0.5, "owner_du_required": 1.0},
        task_context=None,
    )

    context = orchestrator._resolve_context(config)
    outcome = await orchestrator.adeliberate(context, question)

    assert not outcome.answers
    assert outcome.metadata
    assert outcome.metadata.get("du_exhausted", False)
    assert fake_resource_manager.charge_calls == []
    assert fake_resource_manager.ensure_calls == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_council_orchestrator_reports_partial_outcome_and_owner_du_accounting_when_budget_partially_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(infra_config, "_CONFIG", {"USE_COUNCIL_MODE": True})

    fake_ledger = FakeLedger()
    fake_resource_manager = FakeResourceManager(fake_ledger)

    monkeypatch.setattr(resource_manager_module, "_resource_manager", fake_resource_manager)
    monkeypatch.setattr(resource_manager_module, "get_resource_manager", lambda: fake_resource_manager)
    monkeypatch.setattr("src.agents.council.orchestrator.get_resource_manager", lambda: fake_resource_manager)

    monkeypatch.setattr(llm_client, "ledger", fake_ledger)
    monkeypatch.setattr(infra_metrics, "ledger", fake_ledger)

    llm_client.enable_mock_mode(
        True,
        {
            "MemberResponseModel": {
                "answer": "stubbed",
                "reasoning": "",
                "confidence": 1.0,
                "citations": [],
            }
        },
    )

    orchestrator = CouncilOrchestrator()
    config = _build_council_config_with_two_members(member_budget=1.0)
    question = CouncilQuestion(
        question_id="q-owner-partial",
        prompt="What is DU?",
        context="",
        metadata={"agent_id": "owner-1", "owner_du_budget": 1.1, "owner_du_required": 2.0},
        task_context=None,
    )

    context = orchestrator._resolve_context(config)

    outcome = await orchestrator.adeliberate(context, question)

    owner_before = 1.1
    owner_after = fake_resource_manager.get_du_budget("owner-1")

    assert outcome.metadata
    assert outcome.metadata.get("partial", False)
    assert outcome.metadata.get("du_exhausted", False)
    assert 0 < len(outcome.answers) < 2

    for member_id in ("member-a", "member-b"):
        initial_budget = fake_resource_manager.initial_budgets[member_id]
        debited = _sum_du_debits(fake_ledger, agent_id=member_id)
        remaining = fake_resource_manager.get_du_budget(member_id)
        assert remaining == pytest.approx(initial_budget - debited)
        assert fake_ledger.du_budgets.get(member_id, 0.0) == pytest.approx(remaining)

    owner_debited = _sum_du_debits(fake_ledger, agent_id="owner-1")
    assert owner_after == pytest.approx(owner_before - owner_debited)
    assert fake_ledger.du_budgets.get("owner-1", 0.0) == pytest.approx(owner_after)

    metrics = outcome.metadata.get("metrics", {})
    assert metrics.get("du_budget_exhausted", False)
    assert metrics.get("owner_agent_id") == "owner-1"
    assert metrics.get("owner_du_before") == pytest.approx(owner_before)
    assert metrics.get("owner_du_after") == pytest.approx(owner_after)
    assert metrics.get("owner_du_debited") == pytest.approx(owner_debited)

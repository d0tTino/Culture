from __future__ import annotations

from src.infra.event_log import build_domain_event_batch


def test_build_domain_event_batch_is_canonical() -> None:
    batch = build_domain_event_batch(
        step=12,
        events=[{"domain": "turn", "name": "planned", "payload": {"x": 1}}],
        phase_order=["ingress", "plan", "commit"],
    )

    assert batch["type"] == "domain_event_batch"
    assert batch["step"] == 12
    assert batch["phase_order"] == ["ingress", "plan", "commit"]
    assert isinstance(batch["batch_hash"], str)
    assert len(batch["batch_hash"]) == 64

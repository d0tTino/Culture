import sys

import pytest

from scripts import visualize_agent_memory as vam


@pytest.mark.unit
def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--agent_id", "agent1"])
    args = vam.parse_args()
    assert args.agent_id == "agent1"
    assert args.output_format == "text"
    assert args.max_length == 200
    assert args.chroma_dir == "./chroma_db"


@pytest.mark.unit
def test_visualize_text(monkeypatch: pytest.MonkeyPatch) -> None:
    l2 = [{"id": "l2_1", "step": 1, "consolidation_period": "0-1", "content": "chapter"}]
    l1 = [{"id": "l1_1", "step": 1, "content": "session"}]
    monkeypatch.setattr(vam, "get_all_l2_summaries", lambda *_: l2)
    monkeypatch.setattr(vam, "get_all_l1_summaries", lambda *_: l1)

    out = vam.visualize_agent_memory(object(), "agent1", "text", 50)
    assert "AGENT MEMORY VISUALIZATION" in out
    assert "L1_Summary" in out


@pytest.mark.unit
def test_visualize_html(monkeypatch: pytest.MonkeyPatch) -> None:
    l2 = [{"id": "l2_1", "step": 1, "consolidation_period": "0-1", "content": "chapter"}]
    l1 = [{"id": "l1_1", "step": 1, "content": "session"}]
    monkeypatch.setattr(vam, "get_all_l2_summaries", lambda *_: l2)
    monkeypatch.setattr(vam, "get_all_l1_summaries", lambda *_: l1)

    out = vam.visualize_agent_memory(object(), "agent1", "html", 50)
    assert "<html" in out.lower()
    assert "L1 Summary" in out

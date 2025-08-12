from src.interfaces.dashboard_backend import board_payload_to_embed


def test_board_payload_to_embed() -> None:
    payload = {"agent_id": "agent12345678", "content": "hello", "step": 5}
    embed = board_payload_to_embed(payload)
    assert embed["title"] == "📝 New Knowledge Board Entry (Step 5)"
    assert embed["description"] == "```hello```"
    assert embed["color"] == 0xFFD700
    assert embed["author"] == {"name": "Posted by Agent agent1234"}

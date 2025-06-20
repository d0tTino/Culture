import pytest

from src import http_app


@pytest.mark.unit
def test_main_uses_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("HTTP_PORT", "5678")

    captured: dict[str, object] = {}

    def dummy_run(app: object, host: str, port: int) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(http_app.uvicorn, "run", dummy_run)

    http_app.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 5678
    assert captured["app"] is http_app.app

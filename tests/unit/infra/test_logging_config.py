import logging

import pytest

from src.infra import config, logging_config


@pytest.mark.unit
def test_custom_otel_endpoint(monkeypatch, tmp_path):
    """setup_logging should read OTEL_EXPORTER_ENDPOINT env var."""
    monkeypatch.setenv("ENABLE_OTEL", "1")
    monkeypatch.setenv("OTEL_EXPORTER_ENDPOINT", "http://example.com:4318/v1/logs")

    config.load_config()

    endpoints: list[str] = []

    class DummyExporter:
        def __init__(self, *, endpoint: str) -> None:
            endpoints.append(endpoint)

    class DummyResource:
        @staticmethod
        def create(_: dict[str, str]):
            return object()

    class DummyProvider:
        def __init__(self, resource: object):
            self.resource = resource

        def add_log_processor(self, processor: object) -> None:
            self.processor = processor

    class DummyProcessor:
        def __init__(self, exporter: object):
            self.exporter = exporter

    class DummyHandler(logging.Handler):
        def __init__(self, level: int, logger_provider: object) -> None:
            super().__init__(level)
            self.provider = logger_provider

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple stub
            pass

    monkeypatch.setattr(logging_config, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(logging_config, "OTLPLogExporter", DummyExporter, raising=False)
    monkeypatch.setattr(logging_config, "BatchLogProcessor", DummyProcessor, raising=False)
    monkeypatch.setattr(logging_config, "LoggerProvider", DummyProvider, raising=False)
    monkeypatch.setattr(logging_config, "LoggingHandler", DummyHandler, raising=False)
    monkeypatch.setattr(logging_config, "Resource", DummyResource, raising=False)

    logging_config.setup_logging(log_dir=str(tmp_path))

    assert endpoints == ["http://example.com:4318/v1/logs"]

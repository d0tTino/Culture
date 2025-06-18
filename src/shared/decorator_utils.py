import functools
import json
import sys
import time
import uuid
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from src.interfaces import metrics
from src.shared.logging_utils import get_logger

# Setup a dedicated logger for LLM performance metrics
llm_perf_logger = get_logger("llm_performance")

P = ParamSpec("P")
R = TypeVar("R")


def monitor_llm_call(
    model_param: str = "model", context: Optional[str] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for monitoring LLM call performance metrics.

    Args:
        model_param: The parameter name that contains the model name in the decorated function
        context: Optional static context identifier (e.g., "agent_turn", "memory_consolidation")
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Create request ID and capture start time
            request_id = str(uuid.uuid4())[:8]
            start_time = time.perf_counter()

            # Extract model name from kwargs or use default
            model = kwargs.get(model_param, "unknown_model")

            # Initialize metrics dictionary
            metrics_data: dict[str, Any] = {
                "request_id": request_id,
                "function": func.__name__,
                "model": model,
                "timestamp": time.time(),
                "context": context,
                "success": False,
                "duration_ms": None,
                "error_type": None,
                "error_message": None,
                "status_code": None,
            }

            result = None
            try:
                result = func(*args, **kwargs)
                if result is None:
                    metrics_data["success"] = False
                    exc_type, exc_value, _ = sys.exc_info()
                    if exc_type and exc_value:
                        metrics_data["error_type"] = exc_type.__name__
                        metrics_data["error_message"] = str(exc_value)
                        if hasattr(exc_value, "status_code"):
                            metrics_data["status_code"] = exc_value.status_code
                        if hasattr(exc_value, "response") and hasattr(exc_value.response, "text"):
                            metrics_data["error_message"] = exc_value.response.text
                    else:
                        metrics_data["error_type"] = "UnknownError"
                        metrics_data["error_message"] = (
                            "Function returned None, indicating an error"
                        )
                else:
                    metrics_data["success"] = True
                    if result is not None and hasattr(result, "usage"):
                        metrics_data["prompt_tokens"] = getattr(
                            result.usage, "prompt_tokens", None
                        )
                        metrics_data["completion_tokens"] = getattr(
                            result.usage, "completion_tokens", None
                        )
            except Exception as e:
                metrics_data["success"] = False
                metrics_data["error_type"] = type(e).__name__
                metrics_data["error_message"] = str(e)
                if hasattr(e, "status_code"):
                    metrics_data["status_code"] = e.status_code
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    metrics_data["error_message"] = e.response.text
                raise
            finally:
                end_time = time.perf_counter()
                metrics_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
                metrics.LLM_CALLS_TOTAL.inc()
                metrics.LLM_LATENCY_MS.set(metrics_data["duration_ms"])
                llm_perf_logger.info(f"LLM_CALL_METRICS: {json.dumps(metrics_data)}")
            return result

        return wrapper

    return decorator

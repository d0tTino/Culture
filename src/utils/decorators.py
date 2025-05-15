import time
import logging
import functools
import json
import uuid
import traceback
import sys
from typing import Callable, Any, Optional, Dict

# Setup a dedicated logger for LLM performance metrics
llm_perf_logger = logging.getLogger("llm_performance")

def monitor_llm_call(model_param: str = "model", context: Optional[str] = None):
    """
    Decorator for monitoring LLM call performance metrics.
    
    Args:
        model_param: The parameter name that contains the model name in the decorated function
        context: Optional static context identifier (e.g., "agent_turn", "memory_consolidation")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create request ID and capture start time
            request_id = str(uuid.uuid4())[:8]
            start_time = time.perf_counter()
            
            # Extract model name from kwargs or use default
            model = kwargs.get(model_param, "unknown_model")
            
            # Initialize metrics dictionary
            metrics: Dict[str, Any] = {
                "request_id": request_id,
                "function": func.__name__,
                "model": model,
                "timestamp": time.time(),
                "context": context,
                "success": False,
                "duration_ms": None,
                "error_type": None,
                "error_message": None,
                "status_code": None
            }
            
            result = None
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Check for None result - in our specific case a None result 
                # from generate_text indicates an error occurred
                if result is None:
                    metrics["success"] = False
                    # Attempt to capture last exception info even if it was caught
                    # by the function we're wrapping
                    exc_type, exc_value, _ = sys.exc_info()
                    if exc_type and exc_value:
                        metrics["error_type"] = exc_type.__name__
                        metrics["error_message"] = str(exc_value)
                        
                        # Try to extract status code from Ollama ResponseError
                        if hasattr(exc_value, "status_code"):
                            metrics["status_code"] = exc_value.status_code
                        
                        # Extract response message if available
                        if hasattr(exc_value, "response") and hasattr(exc_value.response, "text"):
                            metrics["error_message"] = exc_value.response.text
                    else:
                        metrics["error_type"] = "UnknownError"
                        metrics["error_message"] = "Function returned None, indicating an error"
                else:
                    metrics["success"] = True
                
                # Try to extract token information if available in result
                if result is not None and hasattr(result, "usage"):
                    metrics["prompt_tokens"] = getattr(result.usage, "prompt_tokens", None)
                    metrics["completion_tokens"] = getattr(result.usage, "completion_tokens", None)
                
            except Exception as e:
                # Capture error details
                metrics["success"] = False
                metrics["error_type"] = type(e).__name__
                metrics["error_message"] = str(e)
                
                # Try to extract status code from Ollama ResponseError
                if hasattr(e, "status_code"):
                    metrics["status_code"] = e.status_code
                
                # Extract response message if available
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    metrics["error_message"] = e.response.text
                
                # Re-raise the exception after logging
                raise
                
            finally:
                # Always calculate duration
                end_time = time.perf_counter()
                metrics["duration_ms"] = round((end_time - start_time) * 1000, 2)
                
                # Log metrics as structured JSON
                llm_perf_logger.info(f"LLM_CALL_METRICS: {json.dumps(metrics)}")
            
            return result
                
        return wrapper
    return decorator 
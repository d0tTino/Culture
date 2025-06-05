# LLM Call Performance Monitoring Strategies

## Introduction

The "Culture: An AI Genesis Engine" project relies heavily on LLM calls to power agent cognition, memory consolidation, and other critical simulation components. As the simulation scales, these calls can become a significant performance bottleneck and potential point of failure. Implementing comprehensive monitoring for LLM calls to the local Ollama instance will provide valuable insights into:

1. **Performance bottlenecks** that might be slowing down the simulation
2. **Error patterns** that could indicate issues with specific models or request types
3. **Resource utilization** to optimize token usage and processing time
4. **Simulation stability** by identifying and addressing failed LLM calls

This document outlines recommended metrics and practical implementation strategies for monitoring LLM calls via the `src.infra.llm_client.py` module, with a focus on lightweight approaches that minimize overhead while providing actionable insights.

## Key Metrics to Track

The following metrics provide comprehensive visibility into LLM call performance:

1. **Latency/Duration**
   - Total request duration (from initiation to complete response)
   - Time spent waiting for first token (initial latency)
   - Time spent receiving stream of tokens (for streaming responses)

2. **Success/Failure Status**
   - Binary success/failure indicator
   - HTTP status code (if available from Ollama API)
   - Error type and message for failed calls

3. **Token Information**
   - Prompt token count
   - Completion token count
   - Total tokens processed
   - Note: Availability depends on Ollama client library capabilities

4. **Context Information**
   - Model name used (e.g., "mistral:latest")
   - Function called (e.g., `generate_text`, `generate_structured_output`)
   - Caller context (which component/agent requested the LLM call)
   - Timestamp
   - Request ID (for correlation)

5. **Resource Utilization**
   - Memory usage before/after call (if significant memory issues are suspected)
   - CPU time (if available and relevant)

## Strategy 1: Wrapper Function with Direct Logging

### Implementation Method

This approach involves modifying the core LLM call functions in `src.infra.llm_client.py` to wrap each Ollama API call with timing and error tracking logic.

```python
def generate_text(prompt, model="mistral:latest", temperature=0.7):
    """Generate text completion from Ollama."""
    import time
    import logging
    from src.shared.logging_utils import get_logger
    import uuid
    
    # Create unique request ID for correlation
    request_id = str(uuid.uuid4())[:8]
    
    # Setup logging
    llm_logger = get_logger("llm_performance")
    
    # Capture start time with high precision
    start_time = time.perf_counter()
    
    # Prepare metrics dictionary
    metrics = {
        "request_id": request_id,
        "model": model,
        "function": "generate_text",
        "timestamp": time.time(),
        "success": False,
        "error_type": None,
        "error_message": None,
        "duration_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None
    }
    
    # Track caller if possible (e.g., agent_id)
    # This would need to be passed as a parameter or inferred from the stack
    metrics["caller_context"] = get_caller_context()  # hypothetical function
    
    try:
        # Call to Ollama
        client = ollama.Client()
        response = client.generate(model=model, prompt=prompt, options={"temperature": temperature})
        
        # Mark as successful
        metrics["success"] = True
        
        # Extract token info if available
        if hasattr(response, "prompt_tokens"):
            metrics["prompt_tokens"] = response.prompt_tokens
        if hasattr(response, "completion_tokens"):
            metrics["completion_tokens"] = response.completion_tokens
        
        # Calculate duration
        metrics["duration_seconds"] = time.perf_counter() - start_time
        
        # Log metrics
        llm_logger.info(f"LLM_CALL_METRICS: {json.dumps(metrics)}")
        
        # Return the original response
        return response.get('response', '')
        
    except Exception as e:
        # Calculate duration even for errors
        metrics["duration_seconds"] = time.perf_counter() - start_time
        
        # Capture error details
        metrics["error_type"] = type(e).__name__
        metrics["error_message"] = str(e)
        
        # Log error metrics
        llm_logger.error(f"LLM_CALL_ERROR: {json.dumps(metrics)}")
        
        # Re-raise or handle based on project requirements
        raise
```

### Data Recording

- Structured JSON logging to a dedicated log file (`llm_performance.log`)
- Can be easily extended to output to a separate CSV file for analysis
- Logs both successful calls and errors with consistent format

### Pros & Cons

**Pros:**
- Simple implementation requiring changes to only a few core functions
- Complete visibility into every LLM call
- Detailed error capturing and reporting
- Minimal dependencies (just Python standard library)

**Cons:**
- Increases code complexity in critical LLM client functions
- Some duplication of monitoring code across different functions
- May be verbose in logs if LLM calls are frequent
- Challenging to compute aggregate metrics (requires post-processing)

## Strategy 2: Decorator-Based Monitoring

### Implementation Method

This approach creates a decorator that can be applied to any function making LLM calls, keeping the core functions clean while providing consistent monitoring.

```python
# In a new file: src/infra/monitoring.py
import functools
import time
import logging
import json
from src.shared.logging_utils import get_logger
import uuid
from typing import Callable, Any

def monitor_llm_call(model_param: str = "model", context: str = None):
    """
    Decorator for monitoring LLM calls.
    
    Args:
        model_param: The parameter name that contains the model name in the decorated function
        context: Optional static context identifier
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Setup logging
            llm_logger = get_logger("llm_performance")
            
            # Create request ID and get start time
            request_id = str(uuid.uuid4())[:8]
            start_time = time.perf_counter()
            
            # Extract model name from kwargs or use default
            model = kwargs.get(model_param, "unknown_model")
            
            # Initialize metrics
            metrics = {
                "request_id": request_id,
                "function": func.__name__,
                "model": model,
                "timestamp": time.time(),
                "context": context,
                "success": False,
                "duration_seconds": None,
                "error_type": None,
                "error_message": None
            }
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Mark as successful
                metrics["success"] = True
                
                # Calculate duration
                metrics["duration_seconds"] = time.perf_counter() - start_time
                
                # Try to extract token information if available in result
                if hasattr(result, "usage"):
                    metrics["prompt_tokens"] = getattr(result.usage, "prompt_tokens", None)
                    metrics["completion_tokens"] = getattr(result.usage, "completion_tokens", None)
                
                # Log metrics
                llm_logger.info(f"LLM_CALL_METRICS: {json.dumps(metrics)}")
                
                return result
                
            except Exception as e:
                # Calculate duration even for errors
                metrics["duration_seconds"] = time.perf_counter() - start_time
                
                # Capture error details
                metrics["error_type"] = type(e).__name__
                metrics["error_message"] = str(e)
                
                # Log error metrics
                llm_logger.error(f"LLM_CALL_ERROR: {json.dumps(metrics)}")
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator

# Then in llm_client.py, use the decorator:
@monitor_llm_call(model_param="model", context="text_generation")
def generate_text(prompt, model="mistral:latest", temperature=0.7):
    # Original function body without monitoring code
    client = ollama.Client()
    response = client.generate(model=model, prompt=prompt, options={"temperature": temperature})
    return response.get('response', '')
```

### Data Recording

- Same structured JSON logging as Strategy 1
- Centralized logging logic in the decorator implementation
- Optional parameter to specify context for better categorization

### Pros & Cons

**Pros:**
- Keeps core LLM functions clean and focused on their primary responsibility
- Consistent monitoring across all decorated functions
- Easy to apply to new functions as they're added
- Centralizes monitoring logic for easier maintenance
- Can include custom context information

**Cons:**
- Slightly more complex implementation initially
- May be challenging to access function-specific details
- Additional import dependencies across files
- Requires understanding of decorators for maintenance

## Strategy 3: Context Manager with Aggregation

### Implementation Method

This approach uses a context manager to wrap LLM calls and includes periodic aggregation of statistics to reduce log volume while maintaining visibility.

```python
# In src/infra/monitoring.py
import time
import logging
import json
import statistics
import threading
from typing import Dict, List, Optional
from src.shared.logging_utils import get_logger
from contextlib import contextmanager

class LLMPerformanceMonitor:
    """Monitors and aggregates LLM call performance metrics."""
    
    def __init__(self, aggregation_interval: int = 60):
        self.metrics_store: List[Dict] = []
        self.lock = threading.Lock()
        self.logger = get_logger("llm_performance")
        self.aggregation_interval = aggregation_interval
        self.last_aggregation = time.time()
    
    @contextmanager
    def monitor(self, function_name: str, model: str, context: Optional[str] = None):
        """Context manager for monitoring LLM calls."""
        start_time = time.perf_counter()
        metrics = {
            "function": function_name,
            "model": model,
            "context": context,
            "timestamp": time.time(),
            "success": True,
            "duration_seconds": None,
            "error_type": None,
            "error_message": None
        }
        
        try:
            yield metrics  # The result can be modified by the caller
            
        except Exception as e:
            metrics["success"] = False
            metrics["error_type"] = type(e).__name__
            metrics["error_message"] = str(e)
            raise
            
        finally:
            # Always record duration
            metrics["duration_seconds"] = time.perf_counter() - start_time
            
            # Store metrics
            with self.lock:
                self.metrics_store.append(metrics)
                
                # Check if it's time to aggregate
                current_time = time.time()
                if current_time - self.last_aggregation >= self.aggregation_interval:
                    self._aggregate_and_log()
                    self.last_aggregation = current_time
    
    def _aggregate_and_log(self):
        """Aggregate and log metrics, then clear the store."""
        if not self.metrics_store:
            return
            
        # Group by model and function
        grouped_metrics = {}
        for m in self.metrics_store:
            key = (m["function"], m["model"])
            if key not in grouped_metrics:
                grouped_metrics[key] = []
            grouped_metrics[key].append(m)
        
        # Compute aggregates for each group
        for (func, model), metrics_list in grouped_metrics.items():
            # Extract durations and count successes/failures
            durations = [m["duration_seconds"] for m in metrics_list if m["success"]]
            success_count = sum(1 for m in metrics_list if m["success"])
            failure_count = len(metrics_list) - success_count
            
            # Compute statistics if we have successful calls
            if durations:
                agg_metrics = {
                    "function": func,
                    "model": model,
                    "period_seconds": self.aggregation_interval,
                    "call_count": len(metrics_list),
                    "success_rate": success_count / len(metrics_list),
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p90_duration": sorted(durations)[int(len(durations) * 0.9)] if len(durations) >= 10 else None,
                    "failure_count": failure_count
                }
                
                if failure_count > 0:
                    # Collect error types
                    error_types = {}
                    for m in metrics_list:
                        if not m["success"]:
                            error_type = m["error_type"]
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                    agg_metrics["error_types"] = error_types
                
                # Log the aggregated metrics
                self.logger.info(f"LLM_AGGREGATED_METRICS: {json.dumps(agg_metrics)}")
        
        # Clear the metrics store
        self.metrics_store.clear()

# Create a singleton instance
llm_monitor = LLMPerformanceMonitor(aggregation_interval=300)  # 5 minutes

# Example usage in llm_client.py
def generate_text(prompt, model="mistral:latest", temperature=0.7):
    """Generate text completion from Ollama."""
    with llm_monitor.monitor("generate_text", model, context="agent_turn"):
        client = ollama.Client()
        response = client.generate(model=model, prompt=prompt, options={"temperature": temperature})
        return response.get('response', '')
```

### Data Recording

- Combines individual call logging with periodic aggregation
- Reduces log volume while maintaining statistical visibility
- Computes important aggregates like p90 latency and error rates
- Groups metrics by function and model

### Pros & Cons

**Pros:**
- Balances detailed monitoring with reduced log volume
- Provides statistical insights that individual logs don't offer
- Thread-safe implementation for concurrent LLM calls
- Supports both individual call monitoring and trending analysis

**Cons:**
- Most complex implementation of the three strategies
- Introduces shared state (the metrics store)
- Requires careful threading consideration
- May delay visibility into some issues until aggregation occurs
- Memory usage increases between aggregation intervals

## Recommendation

For the "Culture: An AI Genesis Engine" project, I recommend implementing **Strategy 2: Decorator-Based Monitoring** as the initial approach for the following reasons:

1. **Balance of Simplicity and Cleanliness**: The decorator approach provides a clean separation of concerns, keeping monitoring logic out of core LLM functions while remaining relatively simple to implement.

2. **Extensibility**: As new LLM functions are added to the project, the decorator can be easily applied to ensure consistent monitoring.

3. **Immediate Visibility**: Unlike aggregation-based approaches, it provides immediate visibility into issues without waiting for periodic reporting.

4. **Low Overhead**: The decorator adds minimal computational overhead to LLM calls, which are already relatively slow operations.

The implementation roadmap should be:

1. Create the monitoring decorator in a separate module
2. Apply it to all existing LLM call functions in `src.infra.llm_client.py`
3. Set up appropriate logging configuration to capture the metrics
4. Consider building a simple analysis script that can process the logs to generate performance reports

If log volume becomes an issue as the project scales, elements from Strategy 3 (periodic aggregation) can be incorporated to reduce verbosity while maintaining visibility into performance trends. 

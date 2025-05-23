"""
Integration module for using Ollama models with DSPy.
Provides a proper implementation of DSPy's LM interface for Ollama models.
"""

import logging
import time
import json
from typing import List, Dict, Any, Union, Optional

# Import DSPy and Ollama
try:
    import dspy
    import ollama
    DSPY_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing DSPy or Ollama: {e}")
    DSPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dspy_ollama")

class OllamaLM(dspy.LM):
    """
    A DSPy-compatible language model implementation for Ollama.
    
    This class implements DSPy's LM interface for local Ollama models,
    ensuring compatibility with DSPy optimizers like BootstrapFewShot.
    """
    
    def __init__(
        self, 
        model_name: str = "mistral:latest",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        request_timeout: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the OllamaLM with model configuration.
        
        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
            temperature: Sampling temperature for generation
            max_tokens: Maximum number of tokens to generate
            request_timeout: Timeout for API requests (in seconds)
            **kwargs: Additional arguments to pass to the Ollama client
        """
        # Initialize the parent class with the model name
        super().__init__(model=model_name)
        
        # Store configuration
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.kwargs = kwargs
        
        # Initialize Ollama client
        try:
            self.client = ollama.Client(host=api_base)
            logger.info(f"Initialized OllamaLM with model {model_name} at {api_base}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
        
        # Track statistics for debugging
        self.total_calls = 0
        self.total_tokens = 0
        self.failed_calls = 0
    
    def basic_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Make a basic request to the Ollama API.
        This is the core method called by DSPy internals.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Dict containing the model's response and metadata
        """
        self.total_calls += 1
        start_time = time.time()
        
        # Merge default kwargs with provided kwargs
        request_kwargs = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens
        }
        request_kwargs.update(self.kwargs)
        request_kwargs.update(kwargs)
        
        try:
            logger.debug(f"Sending request to Ollama: model={self.model_name}, prompt length={len(prompt)}")
            logger.debug(f"Request params: {request_kwargs}")
            
            # Make the API call to Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options=request_kwargs
            )
            
            # Extract the response text and metadata
            response_text = response.get('response', '')
            
            # Track token usage if available
            if 'eval_count' in response:
                self.total_tokens += response['eval_count']
            
            # Format the response for DSPy
            dspy_response = {
                "choices": [{
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "model": self.model_name,
                "usage": {
                    "completion_tokens": len(response_text.split()),  # Approximation
                    "prompt_tokens": len(prompt.split()),  # Approximation
                    "total_tokens": len(prompt.split()) + len(response_text.split())  # Approximation
                },
                "raw_ollama_response": response
            }
            
            duration = time.time() - start_time
            logger.debug(f"Ollama request completed in {duration:.2f}s")
            return dspy_response
            
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Error during Ollama API call: {e}")
            # Return a minimal error response that DSPy can handle
            return {
                "choices": [{
                    "text": f"ERROR: {str(e)}",
                    "index": 0,
                    "finish_reason": "error"
                }],
                "model": self.model_name,
                "error": str(e)
            }
    
    def __call__(self, prompt=None, messages=None, **kwargs) -> List[str]:
        """
        Call the language model with a prompt.
        This is the primary interface used by DSPy modules.
        
        Args:
            prompt: String prompt (when provided directly)
            messages: List of message dicts for chat models (alternative input method)
            **kwargs: Additional parameters for the language model
            
        Returns:
            List of completion strings
        """
        # Make sure we have either prompt or messages
        if prompt is None and messages is None:
            logger.error("Either 'prompt' or 'messages' must be provided")
            return ["Error: No prompt or messages provided"]
            
        # If we have messages but no prompt, convert messages to a prompt string
        if prompt is None and messages is not None:
            logger.debug(f"Converting chat-style messages with {len(messages)} items to string")
            prompt = self._convert_messages_to_string(messages)
        
        # Make the request
        response = self.basic_request(prompt, **kwargs)
        
        # Extract the completion text
        if "choices" in response and response["choices"]:
            completion = response["choices"][0]["text"]
            return [completion]  # DSPy expects a list of completions
        else:
            logger.warning("Empty or invalid response from Ollama")
            return [""]  # Return empty string as fallback
    
    def _convert_messages_to_string(self, messages: List[Dict]) -> str:
        """
        Convert a list of chat messages to a single string prompt.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            String representation of the chat messages
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "failed_calls": self.failed_calls
        }
    
    # Implement required methods from the dspy.LM interface
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> List[str]:
        """
        Generate a completion for the given prompt(s).
        
        Args:
            prompt: A string or list of strings to generate completions for
            kwargs: Additional keyword arguments for generation
            
        Returns:
            List of completions for each prompt
        """
        if isinstance(prompt, list):
            # Handle batch of prompts
            results = []
            for p in prompt:
                result = self.__call__(prompt=p, **kwargs)
                results.extend(result)
            return results
        else:
            # Handle single prompt
            return self.__call__(prompt=prompt, **kwargs)

def configure_dspy_with_ollama(
    model_name: str = "mistral:latest",
    api_base: str = "http://localhost:11434",
    temperature: float = 0.1
) -> Optional[OllamaLM]:
    """
    Configure DSPy to use an Ollama model globally.
    
    Args:
        model_name: Name of the Ollama model to use
        api_base: Base URL for the Ollama API
        temperature: Sampling temperature for generation
        
    Returns:
        The configured OllamaLM instance or None if configuration failed
    """
    if not DSPY_AVAILABLE:
        logger.error("DSPy or Ollama not available. Cannot configure DSPy.")
        return None
    
    try:
        # Create and configure the OllamaLM instance
        ollama_lm = OllamaLM(
            model_name=model_name,
            api_base=api_base,
            temperature=temperature
        )
        
        # Configure DSPy to use this LM globally
        dspy.settings.configure(lm=ollama_lm)
        logger.info(f"DSPy configured to use Ollama model {model_name}")
        
        return ollama_lm
    
    except Exception as e:
        logger.error(f"Failed to configure DSPy with Ollama: {e}")
        return None 
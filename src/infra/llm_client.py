# src/infra/llm_client.py
"""
Provides a client for interacting with the Ollama LLM service.
"""
import ollama
import logging
import json
from .config import OLLAMA_API_BASE # Import base URL from config
from pydantic import BaseModel, Field, ValidationError
from typing import Type, TypeVar, Optional, List, Any, Dict, Union
import requests
from src.utils.decorators import monitor_llm_call

logger = logging.getLogger(__name__)

# Define a generic type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

# Mock implementation variables and functions
_MOCK_ENABLED = False
_MOCK_RESPONSES = {
    "default": "This is a mock response from the LLM client.",
    "text_generation": "This is a mock text generation response.",
    "structured_output": {"action_intent": "continue_collaboration", "reasoning": "Mock reasoning", "action": "Mock action"},
    "memory_summarization": "This is a mock memory summary.",
    "sentiment_analysis": "positive"
}

def enable_mock_mode(enabled=True, mock_responses=None):
    """
    Enable or disable mock mode for testing.
    
    Args:
        enabled (bool): Whether to enable mock mode
        mock_responses (Dict[str, Any], optional): Custom mock responses to use
    """
    global _MOCK_ENABLED, _MOCK_RESPONSES
    _MOCK_ENABLED = enabled
    if mock_responses is not None:
        _MOCK_RESPONSES.update(mock_responses)
    logger.info(f"LLM client mock mode {'enabled' if enabled else 'disabled'}")

def is_mock_mode_enabled():
    """Return whether mock mode is enabled."""
    return _MOCK_ENABLED

def is_ollama_available() -> bool:
    """
    Check if Ollama service is available.
    
    Returns:
        bool: True if Ollama is available, False otherwise
    """
    if _MOCK_ENABLED:
        return True  # When in mock mode, pretend Ollama is available
        
    try:
        # Try to connect to Ollama with a small timeout
        response = requests.get(f"{OLLAMA_API_BASE}", timeout=1)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama is not available: {e}")
        return False

# Check if OLLAMA_API_BASE is set, if not use the default
if not OLLAMA_API_BASE:
    OLLAMA_API_BASE = 'http://localhost:11434'
    logger.warning(f"OLLAMA_API_BASE not set in config, using default: {OLLAMA_API_BASE}")
else:
    logger.info(f"Using OLLAMA_API_BASE: {OLLAMA_API_BASE}")

# Initialize the Ollama client globally using the configured base URL
# This assumes the Ollama service is running and accessible.
try:
    # Ensure OLLAMA_API_BASE is correctly formatted (e.g., 'http://localhost:11434')
    client = ollama.Client(host=OLLAMA_API_BASE)
    # Optional: Perform a quick check to see if the client can connect.
    # client.list() # This might be too slow or throw errors if Ollama is busy/starting.
    # A basic check might be better handled during the first actual call.
    logger.info(f"Ollama client initialized for host: {OLLAMA_API_BASE}")
except Exception as e:
    logger.error(f"Failed to initialize Ollama client for host {OLLAMA_API_BASE}: {e}", exc_info=True)
    # Set client to None or raise an error if Ollama connection is critical at startup
    client = None
    # Consider raising an error if connection is essential:
    # raise ConnectionError(f"Could not connect to Ollama at {OLLAMA_API_BASE}") from e

def get_ollama_client():
    """Returns the initialized Ollama client instance."""
    if client is None:
        logger.error("Ollama client is not available. Check connection and configuration.")
        # Depending on requirements, could try to re-initialize here or just return None
    return client

@monitor_llm_call(model_param="model", context="text_generation")
def generate_text(prompt: str, model: str = "mistral:latest", temperature: float = 0.7) -> str | None:
    """
    Generates text using the configured Ollama client.

    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The Ollama model to use (e.g., "mistral:latest", "llama2:latest").
                     Ensure this model is pulled in your Ollama instance (`ollama pull model_name`).
        temperature (float): The generation temperature (creativity).

    Returns:
        str | None: The generated text, or None if an error occurred or client is unavailable.
    """
    # In mock mode, return a mock response
    if _MOCK_ENABLED:
        logger.debug("Using mock response for text generation")
        # Check for context-specific responses first
        if "summarize" in prompt.lower():
            return _MOCK_RESPONSES.get("summarization", _MOCK_RESPONSES.get("default"))
        return _MOCK_RESPONSES.get("text_generation", _MOCK_RESPONSES.get("default"))
    
    ollama_client = get_ollama_client()
    if not ollama_client:
        logger.warning("Attempted to generate text but Ollama client is unavailable.")
        return None # Client not available

    try:
        logger.debug(f"Sending prompt to Ollama model '{model}':\n---PROMPT START---\n{prompt}\n---PROMPT END---")
        # Use ollama.chat for conversational models
        response = ollama_client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature} # Pass temperature if needed
        )
        # Check if response structure is as expected
        if 'message' in response and 'content' in response['message']:
            generated_text = response['message']['content']
            logger.debug(f"Received response from Ollama: {generated_text}")
            return generated_text.strip()
        else:
            logger.error(f"Unexpected response structure from Ollama: {response}")
            return None

    except Exception as e:
        logger.error(f"Error during Ollama API call to model '{model}': {e}", exc_info=True)
        logger.error(f"Request details: model={model}, temperature={temperature}")
        logger.error(f"Ollama server URL: {OLLAMA_API_BASE}")
        logger.error("Please ensure the Ollama server is running and the model is available")
        # Additional debugging info for network errors
        if "connection" in str(e).lower():
            logger.error(f"Connection error. Check if Ollama server is running at {OLLAMA_API_BASE}")
        elif "404" in str(e):
            logger.error(f"Model '{model}' not found. Available models can be listed with 'ollama list'")
        return None

@monitor_llm_call(model_param="model", context="memory_summarization")
def summarize_memory_context(memories: List[str], goal: str, current_context: str, model: str = "mistral:latest", temperature: float = 0.3) -> str:
    """
    Summarizes a list of retrieved memories based on the agent's goal and current context.
    
    Args:
        memories (List[str]): List of raw memory strings retrieved from the vector store
        goal (str): The agent's goal or objective
        current_context (str): Current context (e.g., previous thought or retrieval query)
        model (str): The Ollama model to use for summarization
        temperature (float): Temperature to control creativity/determinism (lower for summarization)
        
    Returns:
        str: A concise summary of the memories relevant to the goal and context
    """
    if not memories:
        return "(No relevant past memories found via RAG)"
    
    # In mock mode, return a mock summary
    if _MOCK_ENABLED:
        logger.debug("Using mock response for memory summarization")
        return _MOCK_RESPONSES.get("memory_summarization", 
            f"This is a mock summary of {len(memories)} memories related to '{goal}'.")
    
    ollama_client = get_ollama_client()
    if not ollama_client:
        logger.warning("Attempted to summarize memories but Ollama client is unavailable.")
        return "(Memory summarization failed: LLM client unavailable)"
    
    # Format memories as a bulleted list for the prompt
    formatted_memories = "\n".join([f"• {memory}" for memory in memories])
    
    # Construct the summarization prompt
    prompt = f"""Summarize the key points from the following memories relevant to the agent's goal ('{goal}') and the current context ('{current_context}'). 
Be concise and focus on information useful for the agent's next step. 
Respond ONLY with the summary text.

MEMORIES:
{formatted_memories}

CONCISE SUMMARY:"""
    
    try:
        logger.debug(f"Sending memory summarization prompt with {len(memories)} memories, goal='{goal}', context='{current_context}'")
        
        response = ollama_client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )
        
        # Extract the summary text from the response
        if 'message' in response and 'content' in response['message']:
            summary = response['message']['content'].strip()
            logger.debug(f"Memory summarization result: {summary}")
            
            # If the summary is empty or too short, return a default message
            if not summary or len(summary) < 10:
                return "(Memory summarization yielded no significant points)"
                
            return summary
        else:
            logger.error(f"Unexpected response structure from Ollama during summarization: {response}")
            return "(Memory summarization failed: Unexpected response format)"
            
    except Exception as e:
        logger.error(f"Error during memory summarization: {e}", exc_info=True)
        return "(Memory summarization failed due to an error)"

@monitor_llm_call(model_param="model", context="sentiment_analysis")
def analyze_sentiment(text: str, model: str = "mistral:latest") -> str | None:
    """
    Analyzes the sentiment of a given text using Ollama.

    Args:
        text (str): The text to analyze.
        model (str): The Ollama model to use.

    Returns:
        str | None: The sentiment classification ('positive', 'negative', 'neutral')
                   or None if analysis fails.
    """
    # In mock mode, return a mock sentiment
    if _MOCK_ENABLED:
        logger.debug("Using mock response for sentiment analysis")
        return _MOCK_RESPONSES.get("sentiment_analysis", "neutral")
    
    ollama_client = get_ollama_client()
    if not ollama_client or not text:
        return None # Client not available or empty text

    # Simple prompt for sentiment classification
    prompt = (
        f"Analyze the sentiment of the following message. Respond with only one word: "
        f"'positive', 'negative', or 'neutral'.\n\nMessage: \"{text}\"\n\nSentiment:"
    )

    try:
        logger.debug(f"Sending sentiment analysis request for text: \"{text}\"")
        response = ollama_client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1} # Low temperature for classification
        )

        if 'message' in response and 'content' in response['message']:
            sentiment = response['message']['content'].strip().lower()
            # Basic validation
            if sentiment in ['positive', 'negative', 'neutral']:
                logger.debug(f"Sentiment analysis result: '{sentiment}' for text: \"{text}\"")
                return sentiment
            else:
                logger.warning(f"Sentiment analysis returned unexpected result: '{sentiment}'. Defaulting to neutral.")
                return 'neutral'
        else:
            logger.error(f"Unexpected response structure from Ollama during sentiment analysis: {response}")
            return None

    except Exception as e:
        logger.error(f"Error during Ollama sentiment analysis call: {e}", exc_info=True)
        return None

@monitor_llm_call(model_param="model", context="structured_output")
def generate_structured_output(
    prompt: str,
    response_model: Type[T],
    model: str = "mistral:latest", # Use a model known to be good at JSON output if possible
    temperature: float = 0.2 # Lower temp often better for structured output
) -> Optional[T]:
    """
    Generate a structured output using the LLM and parse it into the given Pydantic model.
    If mock mode is enabled, returns a mock response that fits the response_model.
    
    Args:
        prompt (str): Instruction prompt for the LLM
        response_model (Type[T]): The Pydantic model to parse the response into
        model (str): The Ollama model to use
        temperature (float): The temperature for generation
        
    Returns:
        Optional[T]: An instance of the response_model, or None if parsing failed
    """
    # In mock mode, generate a compatible mock response
    if _MOCK_ENABLED:
        logger.debug(f"Using mock response for {response_model.__name__}")
        try:
            # Check if we have a specific mock for this model type
            model_name = response_model.__name__
            if model_name in _MOCK_RESPONSES:
                mock_data = _MOCK_RESPONSES[model_name]
                if isinstance(mock_data, dict):
                    # Return a Pydantic model from the dict
                    return response_model(**mock_data)
                else:
                    # Try to parse the string as JSON
                    try:
                        mock_dict = json.loads(mock_data)
                        return response_model(**mock_dict)
                    except json.JSONDecodeError:
                        pass
            
            # Create a minimal valid instance
            field_dict = {}
            for field_name, field in response_model.__fields__.items():
                if field.required:
                    if field.type_ == str:
                        field_dict[field_name] = f"Mock {field_name}"
                    elif field.type_ == int:
                        field_dict[field_name] = 1
                    elif field.type_ == float:
                        field_dict[field_name] = 1.0
                    elif field.type_ == bool:
                        field_dict[field_name] = False
                    elif field.type_ in (list, List):
                        field_dict[field_name] = []
                    elif field.type_ in (dict, Dict):
                        field_dict[field_name] = {}
            
            return response_model(**field_dict)
        except Exception as e:
            logger.error(f"Error generating mock structured output: {e}")
            return None
    
    # Get the Ollama client instance
    ollama_client = get_ollama_client()
    if not ollama_client:
        logger.warning("Attempted to generate structured output but Ollama client is unavailable.")
        return None

    # Add instructions to the prompt about the desired JSON format
    # Include the schema itself in the prompt and a clear example
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    
    # Create a simplified example based on the model's fields
    example = {}
    for field_name, field in response_model.model_fields.items():
        if field.annotation == str:
            example[field_name] = "Example text for " + field_name
        elif field.annotation == Optional[str]:
            # For optional string fields, give an example string value (not None)
            example[field_name] = "Optional example for " + field_name
    
    example_json = json.dumps(example, indent=2)
    
    structured_prompt = (
        f"{prompt}\n\n"
        f"Please respond ONLY with a valid JSON object containing your actual output, NOT the schema itself.\n"
        f"Schema for reference:\n"
        f"```json\n{schema_json}\n```\n\n"
        f"Example response format (use your own content, not these placeholders):\n"
        f"```json\n{example_json}\n```\n\n"
        f"YOUR RESPONSE:"
    )

    try:
        logger.debug(f"Sending structured prompt to Ollama model '{model}':")
        logger.debug(f"---PROMPT START---\n{structured_prompt}\n---PROMPT END---")

        # Call to Ollama's API
        response = requests.post(
            f"http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": structured_prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "num_predict": 400
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")
        
        # Log the full raw response
        logger.debug(f"FULL RAW LLM RESPONSE: {response_text}")
        
        # Try to parse the JSON from the response text
        try:
            # First try to directly extract JSON (cleaner approach if possible)
            logger.debug(f"Received potential JSON response from Ollama: {response_text}")
            
            # If we have a Pydantic model, use it to validate the structure
            if response_model:
                json_data = json.loads(response_text)
                parsed_output = response_model(**json_data)
                logger.debug(f"Successfully parsed structured output: {parsed_output}")
                return parsed_output
            else:
                # Just return the raw JSON object if no model provided
                return json.loads(response_text)
                
        except (json.JSONDecodeError, Exception) as e:
            # If direct JSON parsing failed, try to find a JSON object in the text
            logger.warning(f"Failed to parse JSON from Ollama response: {e}")
            logger.warning(f"Raw response: {response_text}")
            return None
            
    except Exception as e:
        logger.error(f"Error in generate_structured_output: {e}")
        return None 

def get_default_llm_client():
    """
    Creates and returns a default LLM client instance for use in simulations.
    This function is a convenience wrapper that returns the global client.
    
    Returns:
        The initialized Ollama client instance
    """
    return get_ollama_client()

def generate_response(prompt: str, model: str = "mistral:latest", temperature: float = 0.7) -> str | None:
    """
    Generates a response to the given prompt.
    This is an alias for generate_text for backward compatibility.
    
    If mock mode is enabled, returns a predefined mock response.
    """
    # In mock mode, return predefined response
    if _MOCK_ENABLED:
        logger.debug("Using mock response in generate_response")
        return _MOCK_RESPONSES.get("default")
    
    # Otherwise use the real client
    return generate_text(prompt, model, temperature) 
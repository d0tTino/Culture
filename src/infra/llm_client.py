# src/infra/llm_client.py
"""
Provides a client for interacting with the Ollama LLM service.
"""
import ollama
import logging
from .config import OLLAMA_API_BASE # Import base URL from config

logger = logging.getLogger(__name__)

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

def generate_text(prompt: str, model: str = "llama3:latest", temperature: float = 0.7) -> str | None:
    """
    Generates text using the configured Ollama client.

    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The Ollama model to use (e.g., "llama3:latest", "mistral:latest").
                     Ensure this model is pulled in your Ollama instance (`ollama pull model_name`).
        temperature (float): The generation temperature (creativity).

    Returns:
        str | None: The generated text, or None if an error occurred or client is unavailable.
    """
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
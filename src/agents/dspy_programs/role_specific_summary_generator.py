"""
Role-Specific Summary Generator

This module provides DSPy-based summary generators optimized for specific agent roles.
It builds on the base L1 and L2 summary generators but loads role-specific optimized models
when available.
"""

import logging
import os
import re
from collections import Counter
from typing import Optional, Dict, Any, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import dspy
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
    from src.agents.dspy_programs.l1_summary_generator import GenerateL1SummarySignature, L1SummaryGenerator
    from src.agents.dspy_programs.l2_summary_generator import GenerateL2SummarySignature, L2SummaryGenerator
    
    # Configure DSPy with Ollama LM
    configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
    logger.info("Successfully configured DSPy with Ollama for role-specific summarization")
except ImportError as e:
    logger.error(f"Failed to import DSPy or configure Ollama integration: {e}")
    raise

# Common English stopwords for keyword extraction
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "i",
    "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought", "i'm", "you're",
    "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd",
    "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't",
    "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't",
    "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "let's", "that's",
    "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's"
}

def _extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract the most common non-stopwords from text.
    
    Args:
        text (str): The text to extract keywords from
        num_keywords (int): Maximum number of keywords to extract
        
    Returns:
        List[str]: List of extracted keywords
    """
    try:
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 1]
        
        # Count word frequencies
        word_counter = Counter(filtered_words)
        
        # Get the most common words
        common_words = word_counter.most_common(num_keywords)
        
        # Extract just the words (not counts)
        keywords = [word for word, _ in common_words]
        
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        # Return the first few words if keyword extraction fails
        words = text.split()[:num_keywords]
        return words

class RoleSpecificSummaryGenerator:
    """
    Class for generating summaries using role-specific optimized DSPy models.
    
    This generator selects and uses the appropriate role-specific model based on the agent's role,
    providing summaries that better reflect the agent's perspective and thinking style.
    """
    
    SUPPORTED_ROLES = ["Innovator", "Analyzer", "Facilitator"]
    
    def __init__(self):
        """
        Initialize the role-specific summary generator by preparing predictors for each role.
        """
        try:
            # Create dictionaries to store role-specific predictors
            self.l1_predictors = {}
            self.l2_predictors = {}
            
            # Initialize fallback generators
            self.fallback_l1_generator = L1SummaryGenerator()
            self.fallback_l2_generator = L2SummaryGenerator()
            
            # Base path for compiled models
            base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            compiled_dir = base_path / "agents" / "dspy_programs" / "compiled"
            
            # Load role-specific models if available
            for role in self.SUPPORTED_ROLES:
                role_lower = role.lower()
                
                # L1 predictor for this role
                try:
                    l1_path = compiled_dir / f"optimized_l1_{role_lower}_summarizer.json"
                    if l1_path.exists():
                        l1_predictor = dspy.Predict(GenerateL1SummarySignature)
                        l1_predictor.load(str(l1_path))
                        self.l1_predictors[role] = l1_predictor
                        logger.info(f"Loaded optimized L1 {role} summarizer")
                    else:
                        logger.warning(f"No optimized L1 {role} summarizer found at {l1_path}, will use fallback")
                except Exception as e:
                    logger.error(f"Error loading L1 {role} summarizer: {e}")
                
                # L2 predictor for this role
                try:
                    l2_path = compiled_dir / f"optimized_l2_{role_lower}_summarizer.json"
                    if l2_path.exists():
                        l2_predictor = dspy.Predict(GenerateL2SummarySignature)
                        l2_predictor.load(str(l2_path))
                        self.l2_predictors[role] = l2_predictor
                        logger.info(f"Loaded optimized L2 {role} summarizer")
                    else:
                        logger.warning(f"No optimized L2 {role} summarizer found at {l2_path}, will use fallback")
                except Exception as e:
                    logger.error(f"Error loading L2 {role} summarizer: {e}")
            
            # Log the number of role-specific models loaded
            logger.info(f"Loaded {len(self.l1_predictors)} L1 role-specific summarizers and {len(self.l2_predictors)} L2 role-specific summarizers")
            
        except Exception as e:
            logger.error(f"Error initializing RoleSpecificSummaryGenerator: {e}")
            # Ensure we have fallback generators
            self.fallback_l1_generator = L1SummaryGenerator()
            self.fallback_l2_generator = L2SummaryGenerator()
    
    def generate_l1_summary(self, agent_role: str, recent_events: str, current_mood: Optional[str] = None) -> str:
        """
        Generate an L1 summary using the role-specific model if available.
        
        Args:
            agent_role (str): The agent's role
            recent_events (str): Text describing recent events/thoughts/actions
            current_mood (Optional[str]): The agent's current mood, if available
            
        Returns:
            str: A role-specific L1 summary
        """
        try:
            # Normalize role name to match our standard format (capitalize first letter)
            normalized_role = agent_role.strip().capitalize()
            
            # Check if we have a specific predictor for this role
            if normalized_role in self.l1_predictors:
                predictor = self.l1_predictors[normalized_role]
                logger.info(f"Using role-specific L1 summarizer for {normalized_role}")
                
                # Use the role-specific predictor
                prediction = predictor(
                    agent_role=agent_role,
                    recent_events=recent_events,
                    current_mood=current_mood
                )
                
                return prediction.l1_summary
            else:
                # Fall back to the default summarizer
                logger.info(f"No role-specific L1 summarizer available for {agent_role}, using fallback")
                return self.fallback_l1_generator.generate_summary(
                    agent_role=agent_role,
                    recent_events=recent_events,
                    current_mood=current_mood
                )
                
        except Exception as e:
            logger.error(f"Error generating role-specific L1 summary: {e}")
            
            # Try the default summarizer first
            try:
                summary = self.fallback_l1_generator.generate_summary(
                    agent_role=agent_role,
                    recent_events=recent_events,
                    current_mood=current_mood
                )
                return summary
            except Exception as fallback_error:
                # If all DSPy approaches fail, use template-based fallback
                logger.error(f"All DSPy L1 summarization attempts failed for agent {agent_role}: {fallback_error}. Using template fallback.")
                return self._generate_l1_template_fallback(agent_role, recent_events, current_mood)
    
    def _generate_l1_template_fallback(self, agent_role: str, recent_events: str, current_mood: Optional[str] = None) -> str:
        """
        Generate a template-based L1 summary when all DSPy approaches fail.
        
        Args:
            agent_role (str): The agent's role
            recent_events (str): Text describing recent events/thoughts/actions
            current_mood (Optional[str]): The agent's current mood, if available
            
        Returns:
            str: A template-based L1 summary
        """
        # Count the number of events (approximately by line count)
        event_count = len([line for line in recent_events.split('\n') if line.strip()])
        
        # Extract keywords
        keywords = _extract_keywords(recent_events)
        keywords_text = ", ".join(keywords)
        
        # Determine approximate step from the content if possible
        step_match = re.search(r'step[:\s]+(\d+)', recent_events, re.IGNORECASE)
        step_text = f"around step {step_match.group(1)}" if step_match else "in recent activity"
        
        # Create mood text
        mood_text = f" with {current_mood} mood" if current_mood else ""
        
        # Generate the template-based summary
        return (f"L1 Summary (Fallback): Agent {agent_role} processed {event_count} events {step_text}. "
                f"Key topics included: {keywords_text}. "
                f"Mood:{mood_text}.")
    
    def generate_l2_summary(self, agent_role: str, l1_summaries_context: str, 
                           overall_mood_trend: Optional[str] = None,
                           agent_goals: Optional[str] = None) -> str:
        """
        Generate an L2 summary using the role-specific model if available.
        
        Args:
            agent_role (str): The agent's role
            l1_summaries_context (str): A compilation of L1 summaries to be synthesized
            overall_mood_trend (Optional[str]): The agent's mood trend over time
            agent_goals (Optional[str]): The agent's goals, if available
            
        Returns:
            str: A role-specific L2 summary
        """
        try:
            # Normalize role name to match our standard format (capitalize first letter)
            normalized_role = agent_role.strip().capitalize()
            
            # Check if we have a specific predictor for this role
            if normalized_role in self.l2_predictors:
                predictor = self.l2_predictors[normalized_role]
                logger.info(f"Using role-specific L2 summarizer for {normalized_role}")
                
                # Use the role-specific predictor
                prediction = predictor(
                    agent_role=agent_role,
                    l1_summaries_context=l1_summaries_context,
                    overall_mood_trend=overall_mood_trend,
                    agent_goals=agent_goals
                )
                
                return prediction.l2_summary
            else:
                # Fall back to the default summarizer
                logger.info(f"No role-specific L2 summarizer available for {agent_role}, using fallback")
                return self.fallback_l2_generator.generate_summary(
                    agent_role=agent_role,
                    l1_summaries_context=l1_summaries_context,
                    overall_mood_trend=overall_mood_trend,
                    agent_goals=agent_goals
                )
                
        except Exception as e:
            logger.error(f"Error generating role-specific L2 summary: {e}")
            
            # Try the default summarizer first
            try:
                summary = self.fallback_l2_generator.generate_summary(
                    agent_role=agent_role,
                    l1_summaries_context=l1_summaries_context,
                    overall_mood_trend=overall_mood_trend,
                    agent_goals=agent_goals
                )
                return summary
            except Exception as fallback_error:
                # If all DSPy approaches fail, use template-based fallback
                logger.error(f"All DSPy L2 summarization attempts failed for agent {agent_role}: {fallback_error}. Using template fallback.")
                return self._generate_l2_template_fallback(agent_role, l1_summaries_context, overall_mood_trend, agent_goals)
    
    def _generate_l2_template_fallback(self, agent_role: str, l1_summaries_context: str, 
                                      overall_mood_trend: Optional[str] = None,
                                      agent_goals: Optional[str] = None) -> str:
        """
        Generate a template-based L2 summary when all DSPy approaches fail.
        
        Args:
            agent_role (str): The agent's role
            l1_summaries_context (str): A compilation of L1 summaries to be synthesized
            overall_mood_trend (Optional[str]): The agent's mood trend over time
            agent_goals (Optional[str]): The agent's goals, if available
            
        Returns:
            str: A template-based L2 summary
        """
        # Extract start and end steps if present
        step_pattern = r'step[:\s]+(\d+)'
        steps = re.findall(step_pattern, l1_summaries_context, re.IGNORECASE)
        steps = [int(s) for s in steps if s.isdigit()]
        
        if steps:
            start_step = min(steps)
            end_step = max(steps)
            steps_text = f"from step {start_step} to {end_step}"
        else:
            steps_text = "across multiple steps"
        
        # Extract first few words of content (up to 30 words)
        first_words = ' '.join(l1_summaries_context.split()[:30])
        
        # Create goals text
        goals_text = f" Goals: {agent_goals}." if agent_goals else ""
        
        # Create mood trend text
        mood_text = f" Mood Trend: {overall_mood_trend}." if overall_mood_trend else ""
        
        # Generate the template-based summary
        return (f"L2 Summary (Fallback): Agent {agent_role} consolidated L1 summaries {steps_text}. "
                f"Content involved: {first_words}...{goals_text}{mood_text}") 
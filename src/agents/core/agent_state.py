from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Deque, Tuple
from collections import deque

class AgentState(BaseModel):
    """
    Represents the complete internal state of an agent.
    This model is used for the persistent state of an agent instance.
    """
    agent_id: str
    name: str
    role: str
    steps_in_current_role: int = 0
    mood: str = "neutral"
    descriptive_mood: str = "neutral"  # More detailed mood description
    ip: float # Will be initialized from config in BaseAgent
    du: float # Will be initialized from config in BaseAgent
    relationships: Dict[str, float] = Field(default_factory=dict) # Maps agent_id to relationship score
    short_term_memory: Deque[Dict[str, Any]] = Field(default_factory=deque) # Deque of recent memories (dictionaries)
    goals: List[Dict[str, Any]] = Field(default_factory=list) # List of agent goals (dictionaries)
    current_project_id: Optional[str] = None # ID of the project the agent is working on
    current_project_affiliation: Optional[str] = None # Alias for current_project_id used in some parts of the code
    messages_sent_count: int = 0
    messages_received_count: int = 0
    actions_taken_count: int = 0
    last_message_step: int = -1 # -1 indicates no message sent yet
    last_action_step: int = -1 # -1 indicates no action taken yet
    available_action_intents: List[str] = Field(default_factory=list) # Possible action intents for the agent
    step_counter: int = 0  # Count of steps taken in the simulation
    
    # Clarification tracking fields
    last_clarification_question: Optional[str] = None  # Stores the last clarification question asked
    last_clarification_downgraded: bool = False  # Tracks if the last clarification was downgraded due to insufficient DU

    # History fields to track changes over simulation steps
    mood_history: Deque[Tuple[int, str]] = Field(default_factory=deque) # Deque of (step, mood) tuples
    relationship_history: Deque[Tuple[int, Dict[str, float]]] = Field(default_factory=deque) # Deque of (step, relationships dict)
    ip_history: Deque[Tuple[int, float]] = Field(default_factory=deque) # Deque of (step, ip) tuples
    du_history: Deque[Tuple[int, float]] = Field(default_factory=deque) # Deque of (step, du) tuples
    role_history: Deque[Tuple[int, str]] = Field(default_factory=deque) # Deque of (step, role) tuples
    project_history: Deque[Tuple[int, Optional[str]]] = Field(default_factory=deque) # Deque of (step, project_id) tuples

    # Configuration fields (optional, could be passed separately, but useful to have here)
    # These should likely be set during initialization and not change
    max_short_term_memory: int # Will be initialized from config in BaseAgent
    short_term_memory_decay_rate: float # Will be initialized from config in BaseAgent
    relationship_decay_rate: float # Will be initialized from config in BaseAgent
    min_relationship_score: float # Will be initialized from config in BaseAgent
    max_relationship_score: float # Will be initialized from config in BaseAgent
    mood_decay_rate: float # Will be initialized from config in BaseAgent
    mood_update_rate: float # Will be initialized from config in BaseAgent
    ip_cost_per_message: float # Will be initialized from config in BaseAgent
    du_cost_per_action: float # Will be initialized from config in BaseAgent
    role_change_cooldown: int # Will be initialized from config in BaseAgent
    role_change_ip_cost: float # Will be initialized from config in BaseAgent 
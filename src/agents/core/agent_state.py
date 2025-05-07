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
    short_term_memory: Deque[Dict[str, Any]] = Field(default_factory=deque) # Queue of memory entries
    goals: List[Dict[str, Any]] = Field(default_factory=list) # List of goals/objectives for this agent
    role_history: List[Tuple[int, str]] = Field(default_factory=list) # List of (step, role) tuples
    mood_history: List[Tuple[int, str]] = Field(default_factory=list) # List of (step, mood) tuples
    ip_history: List[Tuple[int, float]] = Field(default_factory=list) # List of (step, ip) tuples
    du_history: List[Tuple[int, float]] = Field(default_factory=list) # List of (step, du) tuples
    relationship_history: List[Tuple[int, Dict[str, float]]] = Field(default_factory=list) # List of (step, relationships) tuples
    
    # Project affiliation for Group Affiliation Mechanism
    current_project_id: Optional[str] = None # Current project ID (if any)
    current_project_affiliation: Optional[str] = None # Name of current project for prompting
    
    # Message and action counters
    messages_sent_count: int = 0
    messages_received_count: int = 0
    actions_taken_count: int = 0
    last_message_step: Optional[int] = None
    last_action_step: Optional[int] = None
    
    # Available actions and action intents for this agent
    available_action_intents: List[str] = Field(default_factory=list)
    
    # Step counter for tracking simulation steps
    step_counter: int = 0
    
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
    
    # New relationship dynamics parameters
    positive_relationship_learning_rate: float # Learning rate for positive sentiment interactions
    negative_relationship_learning_rate: float # Learning rate for negative sentiment interactions
    targeted_message_multiplier: float # Multiplier for relationship changes from targeted messages
    
    def add_memory(self, step: int, memory_type: str, content: str):
        """
        Adds a memory to the agent's short-term memory deque.
        
        Args:
            step (int): The simulation step in which the memory occurred
            memory_type (str): Type of memory (e.g., 'thought', 'message_sent', 'message_received')
            content (str): The content of the memory
        """
        memory_entry = {"step": step, "type": memory_type, "content": content}
        self.short_term_memory.append(memory_entry)
    
    def update_relationship(self, other_agent_id: str, delta: float, is_targeted: bool = False):
        """
        Updates the relationship score with another agent using a non-linear formula that considers
        both the sentiment (delta) and the current relationship score.
        
        Args:
            other_agent_id (str): ID of the other agent
            delta (float): Change in relationship score (positive or negative) based on sentiment
            is_targeted (bool): Whether this update is from a targeted message (True) or broadcast (False)
        """
        current_score = self.relationships.get(other_agent_id, 0.0)
        
        # Apply targeted message multiplier if applicable
        effective_delta = delta
        if is_targeted:
            effective_delta = delta * self.targeted_message_multiplier
        
        # Calculate change amount using non-linear formula
        if effective_delta > 0:
            # Positive sentiment: diminishing returns as relationship approaches max
            # Higher current relationships see smaller increases from positive interactions
            change_amount = effective_delta * (self.max_relationship_score - current_score) * self.positive_relationship_learning_rate
        elif effective_delta < 0:
            # Negative sentiment: diminishing returns as relationship approaches min
            # Lower current relationships see smaller decreases from negative interactions
            change_amount = effective_delta * (current_score - self.min_relationship_score) * self.negative_relationship_learning_rate
        else:
            change_amount = 0
        
        # Apply change and clamp to min/max bounds
        new_score = max(self.min_relationship_score, min(self.max_relationship_score, current_score + change_amount))
        self.relationships[other_agent_id] = new_score
        
        return new_score
    
    def update_mood(self, sentiment_score: float):
        """
        Updates the agent's mood based on a sentiment score.
        
        Args:
            sentiment_score (float): The sentiment score to apply to the mood
        """
        from src.agents.graphs.basic_agent_graph import get_mood_level, get_descriptive_mood
        
        # Calculate new mood value using configured decay and update rates
        current_mood_value = float(self.mood_history[-1][1]) if self.mood_history else 0.0
        new_mood_value = current_mood_value * (1 - self.mood_decay_rate) + sentiment_score * self.mood_update_rate
        new_mood_value = max(-1.0, min(1.0, new_mood_value))  # Keep within [-1, 1]
        
        # Update mood and descriptive mood
        new_mood = get_mood_level(new_mood_value)
        new_descriptive_mood = get_descriptive_mood(new_mood_value)
        
        # Update state
        self.mood = new_mood
        self.descriptive_mood = new_descriptive_mood
        self.mood_history.append((self.last_action_step or 0, new_mood))

    def __init__(self, **data):
        super().__init__(**data)
        self.max_short_term_memory = self.max_short_term_memory
        self.short_term_memory_decay_rate = self.short_term_memory_decay_rate
        self.relationship_decay_rate = self.relationship_decay_rate
        self.min_relationship_score = self.min_relationship_score
        self.max_relationship_score = self.max_relationship_score
        self.mood_decay_rate = self.mood_decay_rate
        self.mood_update_rate = self.mood_update_rate
        self.ip_cost_per_message = self.ip_cost_per_message
        self.du_cost_per_action = self.du_cost_per_action
        self.role_change_cooldown = self.role_change_cooldown
        self.role_change_ip_cost = self.role_change_ip_cost
        self.positive_relationship_learning_rate = self.positive_relationship_learning_rate
        self.negative_relationship_learning_rate = self.negative_relationship_learning_rate
        self.targeted_message_multiplier = self.targeted_message_multiplier

    def update_mood(self, new_mood: str):
        self.mood = new_mood
        self.mood_history.append((self.step_counter, new_mood))

    def update_relationship(self, agent_id: str, new_score: float):
        self.relationships[agent_id] = new_score
        self.relationship_history.append((self.step_counter, self.relationships))

    def update_ip(self, new_ip: float):
        self.ip = new_ip
        self.ip_history.append((self.step_counter, new_ip))

    def update_du(self, new_du: float):
        self.du = new_du
        self.du_history.append((self.step_counter, new_du))

    def update_role(self, new_role: str):
        self.role = new_role
        self.role_history.append((self.step_counter, new_role))

    def update_project(self, project_id: Optional[str]):
        self.current_project_id = project_id
        self.project_history.append((self.step_counter, project_id))

    def update_step_counter(self):
        self.step_counter += 1

    def update_short_term_memory(self, memory: Dict[str, Any]):
        self.short_term_memory.append(memory)

    def update_goals(self, goals: List[Dict[str, Any]]):
        self.goals.extend(goals)

    def update_messages_sent_count(self):
        self.messages_sent_count += 1

    def update_messages_received_count(self):
        self.messages_received_count += 1

    def update_actions_taken_count(self):
        self.actions_taken_count += 1

    def update_last_message_step(self, step: int):
        self.last_message_step = step

    def update_last_action_step(self, step: int):
        self.last_action_step = step

    def update_available_action_intents(self, intents: List[str]):
        self.available_action_intents.extend(intents)

    def update_last_clarification_question(self, question: Optional[str]):
        self.last_clarification_question = question

    def update_last_clarification_downgraded(self, downgraded: bool):
        self.last_clarification_downgraded = downgraded

    def update_relationship_history(self, step: int, relationships: Dict[str, float]):
        self.relationship_history.append((step, relationships))

    def update_ip_history(self, step: int, ip: float):
        self.ip_history.append((step, ip))

    def update_du_history(self, step: int, du: float):
        self.du_history.append((step, du))

    def update_role_history(self, step: int, role: str):
        self.role_history.append((step, role))

    def update_project_history(self, step: int, project_id: Optional[str]):
        self.project_history.append((step, project_id))

    def update_current_project_affiliation(self, affiliation: Optional[str]):
        self.current_project_affiliation = affiliation

    def update_current_project_id(self, project_id: Optional[str]):
        self.current_project_id = project_id
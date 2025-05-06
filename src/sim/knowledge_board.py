"""
Defines the Knowledge Board class for maintaining shared knowledge among agents.
"""

import logging
from typing import List, Optional

# Configure logger
logger = logging.getLogger(__name__)

class KnowledgeBoard:
    """
    Represents a shared knowledge board that agents can read from and eventually write to.
    Maintains a list of knowledge entries as strings.
    """
    
    def __init__(self):
        """
        Initialize an empty knowledge board.
        """
        self.entries: List[str] = []
        logger.info("KnowledgeBoard initialized with empty entries list.")
    
    def get_state(self, max_entries: int = 10) -> List[str]:
        """
        Returns the current state of the knowledge board, limited to the most recent entries.
        
        Args:
            max_entries (int): Maximum number of entries to return, starting from most recent.
                               Default is 10.
                               
        Returns:
            List[str]: The most recent entries on the board, up to max_entries.
        """
        # Return the most recent entries, up to max_entries
        # Slice notation [-max_entries:] gets the last max_entries items
        recent_entries = self.entries[-max_entries:] if self.entries else []
        logger.debug(f"KnowledgeBoard: Returning {len(recent_entries)} entries (of {len(self.entries)} total)")
        return recent_entries
    
    def add_entry(self, entry: str, agent_id: str, step: int) -> bool:
        """
        Adds an entry to the knowledge board.
        
        Args:
            entry (str): The knowledge entry to add to the board.
            agent_id (str): ID of the agent proposing the entry.
            step (int): The simulation step when this entry was proposed.
            
        Returns:
            bool: True if the entry was successfully added, False otherwise.
        """
        try:
            formatted_entry = f"Step {step} (Agent: {agent_id}): {entry}"
            self.entries.append(formatted_entry)
            logger.info(f"KnowledgeBoard: Added entry by {agent_id} at step {step}: '{entry}'")
            
            # Optional: Limit board size if needed (e.g., keep only last 100 entries)
            # max_board_size = 100
            # if len(self.entries) > max_board_size:
            #     self.entries = self.entries[-max_board_size:]
            
            return True
        except Exception as e:
            logger.error(f"Failed to add entry to knowledge board: {e}")
            return False 
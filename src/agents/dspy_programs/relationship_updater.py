import dspy

class RelationshipUpdater(dspy.Signature):
    """
    Updates the relationship score between two agents based on their interaction, personas, and sentiment.
    
    Inputs:
        current_relationship_score: float - Current relationship score between the two agents (e.g., -1.0 to 1.0)
        interaction_summary: str - Summary of the most recent interaction between the agents
        agent1_persona: str - Persona/role description of agent 1
        agent2_persona: str - Persona/role description of agent 2
        interaction_sentiment: float - Sentiment score of the interaction (-1.0 to 1.0)
    Outputs:
        new_relationship_score: float - Updated relationship score after the interaction
        relationship_change_rationale: str - Explanation for the relationship score change
    """
    current_relationship_score = dspy.InputField(desc="Current relationship score between the two agents (e.g., -1.0 to 1.0)")
    interaction_summary = dspy.InputField(desc="Summary of the most recent interaction between the agents")
    agent1_persona = dspy.InputField(desc="Persona/role description of agent 1")
    agent2_persona = dspy.InputField(desc="Persona/role description of agent 2")
    interaction_sentiment = dspy.InputField(desc="Sentiment score of the interaction (-1.0 to 1.0)")
    new_relationship_score = dspy.OutputField(desc="Updated relationship score after the interaction")
    relationship_change_rationale = dspy.OutputField(desc="Explanation for the relationship score change") 
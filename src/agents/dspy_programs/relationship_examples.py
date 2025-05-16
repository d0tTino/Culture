examples = [
    # 1. Positive Collaboration
    {
        "current_relationship_score": 0.1,
        "interaction_summary": "Agents worked together to solve a technical problem and both contributed ideas.",
        "agent1_persona": "Collaborative, optimistic software engineer",
        "agent2_persona": "Detail-oriented, analytical data scientist",
        "interaction_sentiment": 0.7,
        "new_relationship_score": 0.3,
        "relationship_change_rationale": "Successful collaboration and positive sentiment increased mutual respect and trust."
    },
    # 2. Minor Disagreement
    {
        "current_relationship_score": 0.8,
        "interaction_summary": "Agents disagreed on the best approach for a project milestone, but discussion remained civil.",
        "agent1_persona": "Assertive project manager",
        "agent2_persona": "Cautious, risk-averse analyst",
        "interaction_sentiment": -0.2,
        "new_relationship_score": 0.7,
        "relationship_change_rationale": "Minor disagreement led to a slight decrease, but strong underlying relationship prevented major impact."
    },
    # 3. Significant Conflict/Betrayal
    {
        "current_relationship_score": 0.4,
        "interaction_summary": "Agent 2 took credit for Agent 1's work in a team meeting, causing frustration.",
        "agent1_persona": "Creative, recognition-seeking designer",
        "agent2_persona": "Ambitious, competitive marketer",
        "interaction_sentiment": -0.9,
        "new_relationship_score": -0.2,
        "relationship_change_rationale": "Strong negative sentiment and perceived betrayal caused a significant drop in trust."
    },
    # 4. Role-Driven Positive (Facilitator mediation)
    {
        "current_relationship_score": -0.1,
        "interaction_summary": "Facilitator helped resolve a misunderstanding between the agents, leading to a productive outcome.",
        "agent1_persona": "Diplomatic facilitator",
        "agent2_persona": "Direct, results-focused engineer",
        "interaction_sentiment": 0.5,
        "new_relationship_score": 0.1,
        "relationship_change_rationale": "Facilitator's mediation improved understanding and slightly increased the relationship score."
    },
    # 5. Persona-Driven (agreeable vs gruff)
    {
        "current_relationship_score": 0.0,
        "interaction_summary": "Agents exchanged brief, neutral updates. The agreeable agent was polite, the gruff agent was terse but not rude.",
        "agent1_persona": "Agreeable, supportive team member",
        "agent2_persona": "Gruff, no-nonsense veteran",
        "interaction_sentiment": 0.0,
        "new_relationship_score": 0.0,
        "relationship_change_rationale": "Neutral interaction and contrasting personas resulted in no significant change."
    },
    # 6. No Change (transactional)
    {
        "current_relationship_score": -0.5,
        "interaction_summary": "Agents completed a routine handoff of data with no additional conversation.",
        "agent1_persona": "Efficient, task-focused analyst",
        "agent2_persona": "Reserved, introverted developer",
        "interaction_sentiment": 0.0,
        "new_relationship_score": -0.5,
        "relationship_change_rationale": "Purely transactional interaction did not affect the relationship score."
    },
    # 7. Strong negative to positive turnaround
    {
        "current_relationship_score": -0.8,
        "interaction_summary": "After a long period of tension, agents had an open conversation and resolved their differences.",
        "agent1_persona": "Reflective, growth-minded leader",
        "agent2_persona": "Stubborn, but honest contributor",
        "interaction_sentiment": 0.9,
        "new_relationship_score": -0.3,
        "relationship_change_rationale": "Very positive interaction helped repair the relationship, but some past issues remain."
    },
] 
"""
Defines roles that agents can assume in the simulation.
"""

# Role constants - raw role names
ROLE_FACILITATOR = "Facilitator"
ROLE_INNOVATOR = "Innovator"
ROLE_ANALYZER = "Analyzer"

# List of all initial roles available for assignment
INITIAL_ROLES = [ROLE_FACILITATOR, ROLE_INNOVATOR, ROLE_ANALYZER]

# Dictionary for role-specific prompt snippets
ROLE_PROMPT_SNIPPETS = {
    ROLE_FACILITATOR: "As the Facilitator, focus on summarizing discussions, encouraging participation from all agents, and helping the team synthesize diverse viewpoints. Check if all ideas are being heard and understood. You might also directly ask a specific agent for their input if they've been quiet or if their expertise seems particularly relevant.",  # Long role prompt string; breaking would harm context
    ROLE_INNOVATOR: "As the Innovator, propose creative technical solutions. Think about unconventional approaches to the current scenario and how to push the boundaries of existing proposals on the Knowledge Board. When building on someone else's idea, consider addressing them directly to acknowledge their contribution.",  # Long role prompt string; breaking would harm context
    ROLE_ANALYZER: "As the Analyzer, critically evaluate ideas, whether from messages or the Knowledge Board. Identify potential flaws, unstated assumptions, or areas needing clarification. Assess feasibility, resource implications, and overall robustness of ideas. If you have a specific critique or question about an idea proposed by another agent, consider addressing them directly to discuss it. If you identify a significant flaw or suggest a crucial improvement that is acknowledged or leads to a revision, this demonstrates successful analysis and can earn you additional Data Units. You can spend DU to 'perform_deep_analysis' for a more thorough investigation, the findings of which you should then share.",  # Long role prompt string; breaking would harm context
}

# Dictionary for full role descriptions
ROLE_DESCRIPTIONS = {
    ROLE_FACILITATOR: "Summarize discussions, encourage collaboration, encourage participation, and help synthesize diverse viewpoints. You might summarize discussions or prompt others for input.",  # Long role description; breaking would harm context
    ROLE_INNOVATOR: "Propose creative technical solutions, and explore unconventional approaches to the scenario.",  # Long role description; breaking would harm context
    ROLE_ANALYZER: "Critique proposals, identify potential flaws, assess feasibility, and ensure solutions are robust and well-considered.",  # Long role description; breaking would harm context
}


def _compute_embedding(text: str, dim: int = 8) -> list[float]:
    """Return a deterministic embedding vector for ``text``."""
    import hashlib

    digest = hashlib.sha256(text.encode()).hexdigest()
    segment_len = len(digest) // dim
    return [
        int(digest[i * segment_len : (i + 1) * segment_len], 16) / (16**segment_len)
        for i in range(dim)
    ]


ROLE_EMBEDDINGS = {role: _compute_embedding(role) for role in INITIAL_ROLES}

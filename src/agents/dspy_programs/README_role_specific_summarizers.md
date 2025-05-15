# Role-Specific Summary Generator

This module provides DSPy-based summary generators optimized for specific agent roles. It builds on the base L1 and L2 summary generators but loads role-specific optimized models when available.

## Overview

The `RoleSpecificSummaryGenerator` class enhances the memory subsystem by providing summaries tailored to each agent's specific role (Innovator, Analyzer, or Facilitator). This ensures that summarized memories reflect the unique perspective and thinking style of each role.

## Fallback Hierarchy

The role-specific summarizer implements a robust fallback hierarchy to ensure that summaries are always generated, even in the face of various failure scenarios:

1. **Role-Specific Optimized Model**: First attempts to use the role-specific optimized DSPy model for the agent's current role (e.g., `optimized_l1_innovator_summarizer.json` for an Innovator agent).

2. **Generic Optimized Model**: If no role-specific model is available, falls back to the generic optimized summarizer (`optimized_l1_summarizer.json`).

3. **Default DSPy Predictor**: If the generic optimized model is unavailable or fails to load, falls back to the default `dspy.Predict` module with the appropriate signature.

4. **Template-Based Fallback**: As the ultimate fallback, if all DSPy-based methods fail (e.g., due to LLM unavailability, DSPy errors, etc.), the system will generate a simple template-based summary using regex and basic text extraction.

This multi-tiered fallback system ensures the memory pipeline remains functional even in the most extreme failure scenarios, maintaining critical data flow throughout the agent system.

## Template-Based Fallback Mechanism

The template-based fallback generates structured summaries using basic text processing:

### L1 Summary Template

```
L1 Summary (Fallback): Agent [Role] processed [N] events around step [X]. 
Key topics included: [Keywords]. Mood: [Mood].
```

Features:
- Counts the number of events in the input text
- Extracts keywords using frequency analysis and stopword removal
- Identifies step numbers from the text where possible
- Includes mood information when available

### L2 Summary Template

```
L2 Summary (Fallback): Agent [Role] consolidated L1 summaries from step [Start] to [End].
Content involved: [First few words of content]... Goals: [Goals]. Mood Trend: [Trend].
```

Features:
- Extracts step range from the input text where possible
- Provides a glimpse of the content from the L1 summaries
- Includes goals and mood trend information when available

## Usage

The `RoleSpecificSummaryGenerator` is designed as a drop-in replacement for the existing `L1SummaryGenerator` and `L2SummaryGenerator` classes:

```python
from src.agents.dspy_programs.role_specific_summary_generator import RoleSpecificSummaryGenerator

# Initialize the generator
summarizer = RoleSpecificSummaryGenerator()

# Generate an L1 summary appropriate for the agent's role
l1_summary = summarizer.generate_l1_summary(
    agent_role="Innovator",
    recent_events="Event 1: Agent discussed new ideas\nEvent 2: Agent shared a prototype",
    current_mood="enthusiastic"
)

# Generate an L2 summary appropriate for the agent's role
l2_summary = summarizer.generate_l2_summary(
    agent_role="Innovator",
    l1_summaries_context="L1 Summary 1: Developed innovative approach\nL1 Summary 2: Prototyped solution",
    overall_mood_trend="increasingly confident",
    agent_goals="Create novel solutions to complex problems"
)
```

## System Robustness

The multi-tiered fallback hierarchy makes the memory subsystem extremely robust:

1. **Resilient to DSPy/LLM Failures**: Even if the DSPy framework or underlying LLM service is completely unavailable, summaries will still be generated.

2. **Graceful Degradation**: The system degrades gracefully, attempting increasingly simpler summarization methods rather than failing completely.

3. **Continuous Operation**: Ensures the memory pipeline continues to function even under exceptional circumstances.

4. **Transparent Fallback**: Clear logging and explicit "Fallback" markers in the output make it obvious when template-based fallbacks are being used.

## Testing

Unit tests for the `RoleSpecificSummaryGenerator` specifically verify template-based fallback functionality:

- `test_l1_template_fallback_when_all_dspy_fails`: Verifies template-based L1 summarization when all DSPy approaches fail
- `test_l2_template_fallback_when_all_dspy_fails`: Verifies template-based L2 summarization when all DSPy approaches fail
- `test_l1_template_fallback_extracts_keywords`: Ensures keyword extraction works in template fallbacks
- `test_l1_template_fallback_without_mood`: Tests template generation without optional mood information
- `test_l2_template_fallback_without_optional_fields`: Tests template generation without optional fields
- `test_normal_operation_uses_fallbacks_properly`: Verifies the normal fallback chain when role-specific models aren't available 
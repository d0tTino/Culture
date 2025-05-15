# Relationship Dynamics Refinement Report

## Overview

This document outlines the refinements made to the relationship dynamics system in the Culture.ai simulation. The goal was to implement a more sophisticated, non-linear formula for relationship updates that accounts for both the sentiment of interactions and the existing relationship score, and to more deeply integrate relationship influence into agent decision-making.

## Implementation Details

### 1. Sentiment Mapping and Update Formula

The relationship update mechanism was refined to use a non-linear formula that creates more realistic relationship dynamics:

- **Sentiment Mapping**: String sentiments ('positive', 'negative', 'neutral') are mapped to numerical values (1.0, -1.0, 0.0) using the `SENTIMENT_TO_NUMERIC` dictionary in `config.py`.

- **Non-Linear Update Formula**:
  - For positive sentiment: `change_amount = sentiment * (max_score - current_score) * positive_learning_rate`
  - For negative sentiment: `change_amount = sentiment * (current_score - min_score) * negative_learning_rate`

- **Targeted vs. Broadcast Impact**: Targeted messages have a stronger impact (3x) on relationships compared to broadcasts, implemented via the `targeted_message_multiplier`.

- **Relationship Decay**: Relationships naturally decay toward neutral at a rate of 1% per turn, requiring active maintenance.

- **Relationship Labels**: Numerical relationship scores (-1.0 to 1.0) are mapped to descriptive labels:
  - "Hostile" (-1.0 to -0.7)
  - "Negative" (-0.7 to -0.4)
  - "Cautious" (-0.4 to -0.1)
  - "Neutral" (-0.1 to 0.1)
  - "Cordial" (0.1 to 0.4)
  - "Positive" (0.4 to 0.7)
  - "Allied" (0.7 to 1.0)

### 2. Bidirectional Relationship Updates

Relationships are now updated bidirectionally:

- **Sender's Perspective**: When an agent sends a targeted message, their relationship with the recipient is updated based on the sentiment of their own message.

- **Receiver's Perspective**: When an agent receives a message, their relationship with the sender is updated based on the sentiment of the received message.

This creates more dynamic relationship networks where both parties' sentiments contribute to the relationship.

### 3. Enhanced Prompting with Relationship Guidance

The prompting system was enhanced with more explicit relationship-based guidance:

- **Formatted Relationship Display**: Agents now see a clear list of their relationships with other agents, including both numerical scores and descriptive labels.

- **Explicit Guidance**: The prompt now includes specific sections on how relationships should influence:
  - Target selection for messages
  - Message tone and content
  - Action intent selection
  - Decision-making and weighing of others' input

### 4. Integration with Agent Decision Making

Relationships now influence key agent decisions:

- **Target Selection**: Agents are explicitly prompted to prefer targeting agents with positive relationships for collaboration.

- **Message Tone**: The prompt instructs agents to adjust their tone based on relationship status (warm for positive, formal for negative).

- **Action Intent Selection**: Agents are guided to choose different intents based on relationship status (e.g., collaboration with allies, cautious clarification with rivals).

- **Input Weighting**: Agents are instructed to give more credibility to input from those they have positive relationships with.

## Configuration Parameters

The following configuration parameters in `config.py` control relationship dynamics:

- `POSITIVE_RELATIONSHIP_LEARNING_RATE = 0.3`: Learning rate for positive sentiment interactions
- `NEGATIVE_RELATIONSHIP_LEARNING_RATE = 0.4`: Learning rate for negative sentiment interactions (slightly higher for more impact)
- `TARGETED_MESSAGE_MULTIPLIER = 3.0`: Multiplier for relationship changes from targeted messages vs broadcasts
- `RELATIONSHIP_DECAY_RATE = 0.01`: Rate at which relationships decay toward neutral each turn

## Expected Behaviors

With these refinements, we expect to observe:

1. Relationships developing more naturally, with extremes (both positive and negative) being harder to reach and maintain.

2. Reciprocal relationships forming, where both parties influence the relationship quality.

3. Targeted messages having significantly more impact than broadcasts on relationship development.

4. Agents forming "cliques" or subgroups based on positive relationships, with more collaboration within these groups.

5. Agent communication patterns reflecting their relationships through tone, target selection, and intent.

6. Relationships requiring ongoing positive interactions to maintain, due to the decay factor.

## Limitations and Future Work

Current limitations of the system include:

1. No memory of past relationship states beyond the basic history tracking.

2. Limited influence of shared project affiliations on relationships.

3. No explicit modeling of group-level relationships (only pairwise).

Future work could address these limitations and further enhance the system by:

1. Implementing relationship "momentum" where rapid changes trigger special events.

2. Adding relationship thresholds that unlock new collaborative capabilities.

3. Creating more complex group dynamics with coalitions and alliances.

4. Developing relationship repair mechanisms for recovering from negative interactions.

## Conclusion

The refined relationship dynamics system creates more realistic, nuanced social behaviors between agents. By implementing a non-linear update formula, bidirectional updates, and explicit integration with decision-making, the system better models how relationships form, evolve, and influence behavior in human social settings. 
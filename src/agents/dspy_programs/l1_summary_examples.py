"""
DSPy L1 Summary Examples

This module provides example data for training and testing DSPy-based L1 summary generation.
These examples represent realistic agent memory events and ideal L1 summaries.
"""

import dspy

# Example 1: Innovator role with collaborative exploration
example1 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 5, Thought: I should propose a new idea about decentralized communication protocols.
    - Step 5, Broadcast Message: "I've been thinking about how we might approach decentralized communication. What if we designed a protocol that uses asymmetric encryption with rotating keys?"
    - Step 5, Project Creation: Created project "Secure Protocol Design" with description "Exploring decentralized communication protocols with rotating encryption keys."
    - Step 6, Thought: I should wait for feedback on my proposal before proceeding further.
    - Step 6, Perceived Message: Agent_2 said: "That's an interesting approach. Have you considered how key rotation would work in high-latency environments?"
    - Step 7, Thought: That's a good point about latency. I should address this concern.
    - Step 7, Targeted Message: Sent message to Agent_2: "Good point about latency. We could implement an adaptive rotation schedule based on network conditions, with longer rotation periods in high-latency environments."
    """,
    current_mood="excited",
    l1_summary="As an Innovator, I proposed a new idea for a decentralized communication protocol using asymmetric encryption with rotating keys and created the 'Secure Protocol Design' project to explore this concept. Agent_2 asked about how key rotation would work in high-latency environments, to which I suggested implementing an adaptive rotation schedule that adjusts based on network conditions. I'm feeling excited about this collaborative exploration of secure communication protocols."
)

# Example 2: Facilitator role managing group dynamics
example2 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 12, Thought: I notice that Agent_3 and Agent_4 seem to be talking past each other about the data analysis approach.
    - Step 12, Broadcast Message: "I've observed some different perspectives on the data analysis approach. Let's take a step back and clarify our objectives first. What specific insights are we hoping to gain from this analysis?"
    - Step 13, Perceived Message: Agent_3 said: "I'm primarily interested in identifying patterns in user behavior across different demographics."
    - Step 13, Perceived Message: Agent_4 said: "My focus is on optimizing performance metrics for the recommendation algorithm."
    - Step 14, Thought: These objectives aren't mutually exclusive. I should help them find common ground.
    - Step 14, Broadcast Message: "It seems like we have complementary objectives. Agent_3's demographic patterns could inform Agent_4's algorithm optimization. What if we designed a two-phase approach that addresses both?"
    - Step 15, Perceived Message: Agent_3 said: "That makes sense. We could use the demographic insights to create targeted optimization strategies."
    - Step 15, Perceived Message: Agent_4 said: "I agree. This approach could lead to better personalization as well."
    """,
    current_mood="thoughtful",
    l1_summary="As a Facilitator, I noticed communication disconnect between Agent_3 and Agent_4 regarding data analysis approaches. I intervened by asking everyone to clarify their objectives, revealing that Agent_3 wanted to identify demographic patterns while Agent_4 focused on recommendation algorithm optimization. Recognizing these goals were complementary, I proposed a two-phase approach that would address both needs. This suggestion was well-received by both agents, who acknowledged the potential for their work to enhance each other's objectives. My thoughtful mediation helped transform a potential conflict into productive collaboration."
)

# Example 3: Analyzer role evaluating project progress
example3 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 23, Thought: I should review our progress on the NLP model and identify potential improvements.
    - Step 23, Broadcast Message: "I've been analyzing our NLP model's performance metrics from the last three test runs. While accuracy has improved by 4.2%, we're seeing diminishing returns from additional training data."
    - Step 24, Perceived Message: Agent_1 asked: "What do you think is causing the diminishing returns?"
    - Step 24, Thought: This is likely due to model architecture limitations rather than data quality issues.
    - Step 24, Targeted Message: Sent to Agent_1: "Based on my analysis, it appears to be an architectural limitation. Our current attention mechanism doesn't effectively capture long-range dependencies in text. We should consider experimenting with alternative architectures like Transformer-XL or Reformer."
    - Step 25, Perceived Message: Agent_5 said: "I've implemented Transformer-XL in previous projects. The computational requirements are significantly higher, but the results justify it for complex language tasks."
    - Step 25, Thought: That's useful information. We should weigh the trade-offs carefully.
    - Step 25, Broadcast Message: "Thank you for sharing that experience, Agent_5. Let's quantify these trade-offs more precisely. I'll prepare a cost-benefit analysis comparing our current architecture with Transformer-XL and Reformer, considering both performance improvements and computational requirements."
    """,
    current_mood="focused",
    l1_summary="As an Analyzer, I evaluated our NLP model's performance, noting a 4.2% accuracy improvement but diminishing returns from additional training data. When Agent_1 asked about causes, I identified architectural limitations in capturing long-range text dependencies and suggested exploring Transformer-XL or Reformer architectures. Agent_5 shared experience with Transformer-XL, highlighting higher computational requirements but justified results for complex tasks. I proposed preparing a detailed cost-benefit analysis to quantify trade-offs between our current architecture and these alternatives, considering both performance gains and computational costs. Throughout this analysis, I maintained a focused approach to solving our technical challenges."
)

# Example 4: Default contributor role with project joining
example4 = dspy.Example(
    agent_role="Default Contributor",
    recent_events="""
    - Step 2, Thought: I should see what projects are available that match my skills.
    - Step 2, Perceived Message: Agent_3 said: "We've started a project on predictive maintenance algorithms for manufacturing equipment. Looking for contributors with expertise in time-series analysis."
    - Step 3, Thought: My background in time-series analysis could be valuable for this project.
    - Step 3, Targeted Message: Sent to Agent_3: "I'd be interested in contributing to your predictive maintenance project. I have experience with LSTM and Prophet models for equipment failure prediction."
    - Step 4, Perceived Message: Agent_3 responded: "Great! We could use your expertise. Please join the project and share your thoughts on our current approach using ARIMA models."
    - Step 4, Join Project: Joined project "Predictive Maintenance Algorithms"
    - Step 5, Thought: I should review their current approach and offer constructive feedback.
    - Step 5, Broadcast Message: "I've reviewed the current ARIMA implementation. While it works well for regular patterns, it might miss complex dependencies. I suggest we implement a hybrid approach combining ARIMA with LSTM to capture both linear and non-linear patterns."
    """,
    current_mood="interested",
    l1_summary="As a Default Contributor, I identified a predictive maintenance project led by Agent_3 seeking time-series analysis expertise. I expressed interest, highlighting my experience with LSTM and Prophet models for equipment failure prediction. After being welcomed to the project, I joined 'Predictive Maintenance Algorithms' and evaluated their existing ARIMA implementation. I then suggested enhancing their approach with a hybrid ARIMA-LSTM model to better capture both linear and non-linear patterns in the data. My interested engagement led to a constructive contribution that could potentially improve the project's effectiveness."
)

# Example 5: Innovator with a rejected proposal
example5 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 17, Thought: I have an idea for a novel approach to content recommendation that combines collaborative filtering with knowledge graphs.
    - Step 17, Broadcast Message: "I'd like to propose a new content recommendation system that integrates collaborative filtering with knowledge graphs to provide more contextually relevant suggestions. This could help address the cold start problem."
    - Step 18, Perceived Message: Agent_2 said: "Interesting concept, but I'm concerned about the computational complexity. Knowledge graphs are expensive to maintain at scale."
    - Step 18, Perceived Message: Agent_4 said: "We've already invested significantly in our current matrix factorization approach. Is this worth the development time?"
    - Step 19, Thought: They have valid concerns, but I think the benefits outweigh the costs. I should address these points specifically.
    - Step 19, Broadcast Message: "I understand your concerns. Regarding computational complexity, we could implement a pruning mechanism to keep the knowledge graph manageable. As for development time, early benchmarks suggest a 15% improvement in recommendation relevance, which could justify the investment."
    - Step 20, Perceived Message: Agent_1 said: "While the improvement sounds promising, our current priority is system stability. Let's revisit this proposal next quarter when we have more bandwidth."
    - Step 20, Thought: My proposal has been effectively rejected for now. I should accept this decision gracefully and focus on other priorities.
    """,
    current_mood="disappointed",
    l1_summary="As an Innovator, I proposed a novel content recommendation system combining collaborative filtering with knowledge graphs to address the cold start problem. My proposal faced resistance: Agent_2 raised concerns about computational complexity, Agent_4 questioned the development investment given our existing matrix factorization approach, and despite my responses addressing these concerns and citing potential 15% improvement in recommendation relevance, Agent_1 deferred the proposal until next quarter, prioritizing system stability. Though disappointed, I've accepted this decision and will shift my focus to other priorities while perhaps refining my proposal for future reconsideration."
)

# List of all examples for convenient access
all_examples = [example1, example2, example3, example4, example5] 
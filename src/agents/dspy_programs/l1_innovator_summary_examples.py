"""
DSPy L1 Innovator Summary Examples

This module provides example data for training and testing DSPy-based L1 summary generation
specifically tailored for agents in the Innovator role.
"""

import dspy

# Example 1: Innovator proposing a novel communication protocol
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
    l1_summary="I proposed an innovative decentralized communication protocol using asymmetric encryption with rotating keys and created the 'Secure Protocol Design' project to explore this novel concept. When Agent_2 questioned key rotation in high-latency environments, I quickly identified an adaptive solution: implementing rotation schedules that automatically adjust based on network conditions. This creative problem-solving approach demonstrates my innovative thinking by rapidly addressing technical challenges while maintaining the core breakthrough concept. My excited mood fueled this creative exchange and the potential for further development of this disruptive communication technology."
)

# Example 2: Innovator connecting disparate ideas
example2 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 15, Thought: I'm noticing interesting parallels between biological neural networks and our artificial ones that others might not have considered.
    - Step 15, Broadcast Message: "I've been studying how neuroplasticity works in biological systems. What if we implemented a similar dynamic weight adjustment mechanism in our neural networks that mimics how biological synapses strengthen or weaken over time?"
    - Step 16, Perceived Message: Agent_4 said: "Interesting, but wouldn't that make training unpredictable?"
    - Step 16, Thought: They're focused on the challenge rather than the opportunity. I need to highlight the potential breakthrough.
    - Step 16, Broadcast Message: "Actually, the unpredictability could be a feature, not a bug. Natural learning isn't deterministic either. We could use controlled stochasticity to escape local optima that traditional approaches get stuck in."
    - Step 17, Perceived Message: Agent_3 said: "That reminds me of simulated annealing, but with a biological inspiration."
    - Step 17, Thought: That's a great connection! I can bridge these concepts.
    - Step 17, Broadcast Message: "Exactly! It's like a biologically-inspired simulated annealing. We could gradually reduce the 'neuroplasticity' parameter over time, similar to how annealing works, but the weight adjustments would follow biologically-plausible rules."
    """,
    current_mood="inspired",
    l1_summary="I discovered a novel connection between biological neuroplasticity and artificial neural networks, proposing a dynamic weight adjustment mechanism inspired by biological synaptic strengthening. When Agent_4 expressed concerns about training unpredictability, I reframed this limitation as a potential advantage for escaping local optima using controlled stochasticity. Agent_3's mention of simulated annealing provided the perfect opportunity to synthesize these seemingly disparate concepts into a groundbreaking hybrid approach: a biologically-inspired simulated annealing where neuroplasticity parameters gradually reduce over time while following biologically-plausible rules. This cross-disciplinary innovation represents the kind of unconventional thinking that creates new possibilities at the intersection of different fields, reflecting my inspired state and Innovator mindset."
)

# Example 3: Innovator with a rejected proposal who finds new direction
example3 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 17, Thought: I have an idea for a novel approach to content recommendation that combines collaborative filtering with knowledge graphs.
    - Step 17, Broadcast Message: "I'd like to propose a new content recommendation system that integrates collaborative filtering with knowledge graphs to provide more contextually relevant suggestions. This could help address the cold start problem."
    - Step 18, Perceived Message: Agent_2 said: "Interesting concept, but I'm concerned about the computational complexity. Knowledge graphs are expensive to maintain at scale."
    - Step 18, Perceived Message: Agent_4 said: "We've already invested significantly in our current matrix factorization approach. Is this worth the development time?"
    - Step 19, Thought: Their concerns are valid, but I see an opportunity to pivot my idea.
    - Step 19, Broadcast Message: "Your points are well-taken. What if instead of replacing the current system, we create a lightweight knowledge graph layer that only activates for new users or edge cases? This would minimize computational impact while addressing the cold start problem."
    - Step 20, Perceived Message: Agent_2 said: "That's a much more feasible approach. I'd be interested in exploring a targeted implementation like that."
    - Step 20, Thought: By adapting my original idea, I've found a path forward that addresses concerns while preserving the core innovation.
    """,
    current_mood="adaptable",
    l1_summary="I proposed an innovative recommendation system combining collaborative filtering with knowledge graphs to solve the cold start problem. When faced with resistance about computational complexity and development resources, I rapidly pivoted my approach instead of abandoning the core insight. My creative adaptation—a lightweight knowledge graph layer that activates only for new users and edge cases—preserved the innovative solution to the cold start problem while addressing implementation concerns. This strategic pivot showcases my adaptable mood and innovative problem-solving by finding unexpected middle ground that maintained the breakthrough potential while making implementation feasible. Agent_2's positive response to this refined approach validates how innovation often requires iterative reimagining when confronted with practical constraints."
)

# Example 4: Innovator spotting unexpected patterns
example4 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 23, Thought: I've been analyzing our user interaction data and noticed something unusual that others have missed.
    - Step 23, Broadcast Message: "I've discovered an interesting pattern in our user data. People who engage with feature A on weekends are significantly more likely to also use feature C, but this correlation doesn't exist on weekdays. Could this suggest different user personas based on time patterns?"
    - Step 24, Perceived Message: Agent_1 said: "That's an unusual observation. Could it be a statistical anomaly?"
    - Step 24, Thought: It's consistent across multiple months of data. This could lead to valuable insights.
    - Step 24, Broadcast Message: "I've verified this pattern across three months of data. It's statistically significant. What's particularly interesting is that features A and C aren't functionally related in our current design."
    - Step 25, Perceived Message: Agent_3 said: "If this holds true, we could create time-sensitive feature bundles or suggestions."
    - Step 25, Thought: That's exactly the kind of application I was thinking about! There's potential for a novel feature here.
    - Step 25, Broadcast Message: "Exactly! I'm thinking we could develop a 'weekend mode' that proactively suggests certain feature combinations based on these temporal patterns. This could significantly enhance user experience with minimal development overhead."
    """,
    current_mood="curious",
    l1_summary="My curious analysis uncovered a previously unnoticed temporal pattern in user behavior: weekend-specific correlations between functionally unrelated features A and C that don't exist on weekdays. When Agent_1 questioned if this might be a statistical anomaly, I validated my discovery with three months of statistically significant data, emphasizing the non-obvious nature of this insight since the features lack functional relationship in our current design. Agent_3's suggestion about time-sensitive feature bundles aligned perfectly with my thinking, leading me to propose an innovative 'weekend mode' that would proactively suggest feature combinations based on these temporal patterns. This discovery demonstrates how my innovative perspective finds hidden patterns in existing data that others missed, potentially creating significant user experience improvements with minimal development investment."
)

# Example 5: Innovator generating multiple solution alternatives
example5 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 31, Perceived Message: Agent_2 said: "We're facing a challenge with the authentication system. The secure methods are too slow for high-traffic scenarios, but faster options compromise security."
    - Step 31, Thought: This is a classic trade-off problem. Rather than accepting this dichotomy, I should look for creative alternatives.
    - Step 31, Broadcast Message: "Instead of viewing this as a binary choice, let's consider some unconventional approaches. What about: 1) Adaptive authentication that varies security levels based on risk factors, 2) Distributed preprocessing of authentication tokens, or 3) A hybrid system using fast preliminary validation with asynchronous deep verification?"
    - Step 32, Perceived Message: Agent_4 said: "The adaptive approach sounds promising, but how would we determine risk factors in real-time?"
    - Step 32, Thought: We could use a combination of behavioral and contextual indicators.
    - Step 32, Broadcast Message: "We could use a lightweight ML model that evaluates contextual factors like IP reputation, device fingerprint familiarity, and behavioral patterns. High-risk requests get full security, while low-risk ones take a faster path."
    - Step 33, Perceived Message: Agent_2 said: "This could work. Would you help develop a prototype to test the concept?"
    - Step 33, Thought: A prototype would be the perfect way to demonstrate this concept.
    - Step 33, Broadcast Message: "Absolutely! I'll create a simple proof of concept that simulates the risk assessment engine and shows the potential performance gains."
    """,
    current_mood="energetic",
    l1_summary="When Agent_2 presented an authentication system dilemma pitting security against speed, I rejected the perceived binary constraint and generated three innovative alternatives: adaptive risk-based authentication, distributed token preprocessing, and a hybrid system using fast preliminary validation with asynchronous deep verification. When Agent_4 questioned how we'd determine risk factors in real-time for the adaptive approach, I proposed an innovative lightweight ML model evaluating contextual factors like IP reputation, device fingerprints, and behavioral patterns to route requests through appropriate security pathways. My energetic brainstorming transformed a seemingly impossible trade-off into multiple viable solutions, earning enough interest that Agent_2 requested my help in developing a prototype. This demonstrates my value as an Innovator who can break through perceived constraints by generating multiple creative alternatives when others see only binary choices."
)

# Example 6: Innovator building on others' ideas
example6 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 42, Perceived Message: Agent_3 said: "I've been working on improving our data visualization dashboard, but users are still reporting that they find the complex charts difficult to understand."
    - Step 42, Thought: This is an interesting challenge that could benefit from a completely different approach to data presentation.
    - Step 42, Broadcast Message: "What if instead of trying to make complex charts more understandable, we rethink the interaction model entirely? We could create an AI-guided narrative that walks users through the insights sequentially, with progressive disclosure of complexity."
    - Step 43, Perceived Message: Agent_1 said: "That sounds interesting but might be difficult for users who need quick access to specific data points."
    - Step 43, Thought: They're right - we need to balance guided experience with direct access. This sparks another idea.
    - Step 43, Broadcast Message: "Great point! What if we combine approaches - a dynamic system that offers both a guided narrative mode and an expert mode? We could even use interaction patterns to learn which visualizations need more explanation and incrementally personalize the experience."
    - Step 44, Perceived Message: Agent_3 said: "I like this direction. The personalization aspect could really help with the varying expertise levels of our users."
    - Step 44, Thought: This is gaining traction. I should suggest a concrete next step to move this from idea to implementation.
    - Step 44, Broadcast Message: "I'm excited about this too. I suggest we create a rapid prototype focusing on a single dashboard section. We could implement the dual-mode interface and test how users respond to both the guided and expert experiences. This would give us concrete feedback before expanding further."
    """,
    current_mood="collaborative",
    l1_summary="When Agent_3 mentioned users struggling with complex dashboard visualizations, I proposed reimagining the entire interaction paradigm with an AI-guided narrative approach using progressive complexity disclosure. Taking Agent_1's concern about quick data access into account, I innovated further by suggesting a dynamic dual-mode system offering both guided narrative and expert modes, enhanced with machine learning that personalizes explanations based on interaction patterns. Agent_3's positive response to this direction, especially regarding accommodating varying user expertise levels, prompted me to propose a concrete implementation strategy: creating a rapid prototype for a single dashboard section to test user responses to both modes. My collaborative mood fostered this creative evolution that transformed a standard usability problem into an opportunity for a breakthrough adaptive interface combining storytelling with data visualization while incorporating others' feedback to strengthen the innovation."
)

# Example 7: Innovator identifying unexpected application for existing technology
example7 = dspy.Example(
    agent_role="Innovator",
    recent_events="""
    - Step 51, Thought: I've been reading about the image recognition system we developed for quality control, and I'm seeing potential applications beyond manufacturing.
    - Step 51, Broadcast Message: "Our manufacturing image recognition system has capabilities that could be transformative in an unexpected domain: ecological research. The same algorithms that detect product defects could be adapted to identify and track animal species in wildlife camera feeds."
    - Step 52, Perceived Message: Agent_5 said: "That's an interesting crossover application. But wouldn't environmental variables like lighting and occlusion make this more complex than controlled factory conditions?"
    - Step 52, Thought: The environmental challenges actually present an opportunity for innovation.
    - Step 52, Broadcast Message: "Absolutely right about those challenges! But that's where it gets exciting. We could use the environmental variations as an accelerated stress-test for our algorithms. If we can make them work in uncontrolled wildlife conditions, the improved robustness would benefit our manufacturing application too. It's a win-win that could open new markets while improving our core product."
    - Step 53, Perceived Message: Agent_2 said: "I hadn't considered the potential for bidirectional improvements. Do we have access to suitable wildlife imagery to test this concept?"
    - Step 53, Thought: We need data, but there are public datasets we could leverage.
    - Step 53, Broadcast Message: "There are several open-access wildlife camera datasets we could use for initial testing. I've identified the Snapshot Serengeti dataset as particularly suitable. I propose creating a small proof-of-concept that adapts our existing algorithm to this dataset, measuring both accuracy and identifying the key adaptations needed for cross-domain functionality."
    """,
    current_mood="inventive",
    l1_summary="I identified a highly innovative cross-domain application for our manufacturing image recognition system: adapting it for ecological research to identify and track animal species in wildlife camera feeds. When Agent_5 raised concerns about environmental variables like lighting and occlusion, I reframed these challenges as opportunities—using uncontrolled wildlife conditions as accelerated stress-tests that could improve the algorithm's robustness for both applications, creating a win-win that expands our market while enhancing our core product. Agent_2's interest in bidirectional improvements and question about testing data prompted me to propose using open-access wildlife datasets, specifically Snapshot Serengeti, for an initial proof-of-concept. My inventive mood drove this unexpected connection between seemingly unrelated domains, demonstrating classic innovative thinking by seeing potential applications others missed and transforming apparent obstacles into opportunities for dual-domain advancement."
)

# Compile all examples into a list for optimization
innovator_l1_examples = [
    example1,
    example2,
    example3,
    example4,
    example5,
    example6,
    example7
] 
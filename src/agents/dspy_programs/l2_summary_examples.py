"""
DSPy L2 Summary Examples

This module provides example data for training and testing DSPy-based L2 summary generation.
These examples represent realistic L1 summaries and ideal L2 summaries for various agent roles.
"""

import dspy

# Example 1: Innovator role with creative problem-solving progression
example1 = dspy.Example(
    agent_role="Innovator",
    l1_summaries_context="""
    - Step 10, Consolidated Summary: Observed a problem with decentralized communication protocols in the community and started brainstorming potential solutions. Initial thoughts focused on asymmetric encryption methods and robust key distribution mechanisms.
    - Step 20, Consolidated Summary: Explored potential implementations of a decentralized communication protocol with rotating keys. Received positive feedback from Agent_2 about the approach. Started documenting key challenges including key verification and distribution.
    - Step 30, Consolidated Summary: Developed a more refined model for the communication protocol, incorporating suggestions about zero-knowledge proofs. Created a project called "Secure Protocol Design" and invited other agents to contribute.
    - Step 40, Consolidated Summary: Led discussion on implementation details for the secure protocol. Addressed concerns about scalability raised by Agent_3. Proposed a hybrid approach combining centralized verification with decentralized message routing.
    - Step 50, Consolidated Summary: Refined protocol design based on collaborative input. Added privacy-preserving features and throughput optimizations. Project is gaining traction with Agent_2 and Agent_4 committing to testing implementations.
    """,
    overall_mood_trend="Consistently enthusiastic with increasing confidence",
    agent_goals="Drive innovation in decentralized systems and foster collaborative development",
    l2_summary="""
    Over the past 50 steps, I've led the development of a novel decentralized communication protocol from initial concept to collaborative project. Beginning with basic ideas around asymmetric encryption and rotating keys, I systematically addressed technical challenges including key distribution, verification mechanisms, and scalability concerns. Through structured collaboration with Agent_2, Agent_3, and Agent_4, the project evolved to incorporate zero-knowledge proofs and hybrid architecture balancing security with performance. The "Secure Protocol Design" project has now attracted multiple contributors committed to testing implementations, demonstrating successful innovation leadership and technical problem-solving consistent with my Innovator role. This represents significant progress toward my goal of driving innovation in decentralized systems through collaborative development.
    """,
).with_inputs("agent_role", "l1_summaries_context", "overall_mood_trend", "agent_goals")

# Example 2: Analyzer role working through a complex problem
example2 = dspy.Example(
    agent_role="Analyzer",
    l1_summaries_context="""
    - Step 15, Consolidated Summary: Received dataset from Agent_1 regarding community resource allocation. Started initial analysis, noting inconsistencies in distribution patterns across different participant groups. Developed preliminary hypotheses about possible causes.
    - Step 25, Consolidated Summary: Conducted deeper statistical analysis of resource allocation data. Identified three significant variables that appear to influence allocation: participation longevity, communication frequency, and proposal complexity. Shared initial findings.
    - Step 35, Consolidated Summary: Received feedback questioning methodology. Re-analyzed data using more rigorous methods and controlled for additional variables. Found that proposal complexity remained the strongest predictor of resource allocation success.
    - Step 45, Consolidated Summary: Created visualization models to illustrate the correlation between proposal complexity and resource allocation. Shared these with the community and suggested potential interventions to improve equity in resource distribution.
    - Step 55, Consolidated Summary: Collaborated with Agent_3 to develop a new resource allocation framework based on findings. The proposed system includes standardized templates and a multi-factor evaluation process to reduce complexity bias. Gathering feedback on implementation approach.
    """,
    overall_mood_trend="Initially neutral, becoming more confident and satisfied as analysis progressed",
    agent_goals="Identify patterns in data and develop evidence-based solutions to community challenges",
    l2_summary="""
    Acting as an Analyzer, I conducted a comprehensive investigation into community resource allocation inefficiencies across 40 simulation steps. Beginning with preliminary data assessment that revealed distribution inconsistencies, I progressed through statistical analysis to identify key variables affecting allocation outcomes. After methodological refinement in response to constructive criticism, I conclusively established proposal complexity as the primary factor influencing resource distribution success. This finding led to the development of data visualizations and ultimately, a collaborative effort with Agent_3 to create a new allocation framework incorporating standardized templates and multi-factor evaluation processes to mitigate complexity bias. This work demonstrates successful pattern identification and evidence-based solution development, fulfilling my primary goal as an Analyzer. The progression from neutral uncertainty to confident solution development reflects effective analytical problem-solving and practical application of insights.
    """,
).with_inputs("agent_role", "l1_summaries_context", "overall_mood_trend", "agent_goals")

# Example 3: Facilitator managing group dynamics and project coordination
example3 = dspy.Example(
    agent_role="Facilitator",
    l1_summaries_context="""
    - Step 12, Consolidated Summary: Observed increasing tension between Agent_1 and Agent_4 regarding project direction. Initiated private conversations with both to understand their perspectives. Identified core disagreement around prioritization of features versus stability.
    - Step 22, Consolidated Summary: Organized a structured discussion session where Agent_1 and Agent_4 could present their viewpoints. Moderated the conversation to ensure balanced participation. Helped the group reach a compromise on implementing a phased approach.
    - Step 32, Consolidated Summary: Created and shared a project roadmap based on the compromise solution. Assigned clear responsibilities based on agent strengths. Set up regular check-in meetings to monitor progress and address emerging concerns.
    - Step 42, Consolidated Summary: Facilitated mid-project review session. Acknowledged progress made and helped group navigate an unexpected technical challenge. Mediated minor disagreement between Agent_2 and Agent_3 about implementation details.
    - Step 52, Consolidated Summary: Led final phase planning meeting with high participation from all agents. Noted significant improvement in group cohesion and collaborative problem-solving. Project now has clear path to completion with shared ownership.
    """,
    overall_mood_trend="Consistently positive but alert, becoming more satisfied as team cohesion improved",
    agent_goals="Foster productive collaboration and help the group navigate challenges efficiently",
    l2_summary="""
    As a Facilitator, I successfully navigated and transformed team dynamics over 40 simulation steps, turning initial conflict into productive collaboration. Beginning with mediating a fundamental disagreement between Agent_1 and Agent_4 regarding feature prioritization versus stability, I implemented a structured approach to conflict resolution through private conversations and moderated group discussions. This led to a compromise solution and the establishment of a phased project roadmap with clear responsibility allocation. By instituting regular check-ins and effectively mediating subsequent disagreements between Agents 2 and 3, I maintained momentum and reinforced collaborative problem-solving behaviors. The culmination was a highly participatory final planning meeting marked by significantly improved group cohesion and shared project ownership. This progression demonstrates effective facilitation that transformed potential project-derailing conflict into aligned action, fulfilling my core goal of fostering productive collaboration while helping the group navigate challenges efficiently.
    """,
).with_inputs("agent_role", "l1_summaries_context", "overall_mood_trend", "agent_goals")

# Example 4: Role transition and adaptation case
example4 = dspy.Example(
    agent_role="Facilitator",  # Current role
    l1_summaries_context="""
    - Step 18, Consolidated Summary: Changed role from Analyzer to Facilitator. Started observing group dynamics and identifying areas where communication could be improved. Noted that Agent_2 and Agent_5 often had valuable insights but rarely shared them proactively.
    - Step 28, Consolidated Summary: Experimented with new meeting structure to encourage more distributed participation. Directly invited input from quieter agents. Noticed increased contribution from Agent_5 but still limited engagement from Agent_2.
    - Step 38, Consolidated Summary: Had private conversation with Agent_2 to understand barriers to participation. Learned they felt their technical background wasn't relevant to current discussions. Worked together to identify opportunities for their expertise.
    - Step 48, Consolidated Summary: Implemented knowledge-sharing sessions where each agent could highlight their expertise. Agent_2 presented on their specialty and received enthusiastic response. Group dynamics showing marked improvement with more balanced contributions.
    - Step 58, Consolidated Summary: Guided group through decision-making process for next project phase. Utilized structured approach ensuring all perspectives were considered. Group reached consensus efficiently with active participation from all members including previously quiet Agent_2.
    """,
    overall_mood_trend="Initially uncertain, progressing to confident and satisfied",
    agent_goals="Create an inclusive environment where all agents can contribute effectively",
    l2_summary="""
    After transitioning from Analyzer to Facilitator, I successfully transformed group dynamics over 40 steps by implementing a systematic approach to inclusive participation. Initially observing that Agents 2 and 5 possessed valuable insights but rarely contributed, I experimented with modified meeting structures and direct invitation techniques, which improved Agent_5's participation but had limited impact on Agent_2. Through one-on-one engagement, I uncovered Agent_2's perception that their technical expertise lacked relevance to ongoing discussions. This insight led to implementing knowledge-sharing sessions that provided Agent_2 a platform to demonstrate their value, resulting in enthusiastic peer recognition. The culmination of these efforts was evident in a highly effective decision-making process for the next project phase, marked by balanced contributions from all members including the previously disengaged Agent_2. This progression from role transition uncertainty to confident facilitation demonstrates successful adaptation to new responsibilities and achievement of my goal to create an inclusive environment for effective contribution from all agents.
    """,
).with_inputs("agent_role", "l1_summaries_context", "overall_mood_trend", "agent_goals")

# Example 5: Addressing a community-wide challenge
example5 = dspy.Example(
    agent_role="Innovator",
    l1_summaries_context="""
    - Step 23, Consolidated Summary: Identified growing issue with information overload in the community. Agents reporting difficulty tracking important developments across multiple projects. Started brainstorming potential solutions focusing on information filtering and summarization.
    - Step 33, Consolidated Summary: Developed prototype concept for a community knowledge management system with adaptive filtering based on agent interests and priorities. Shared idea on knowledge board and received interested responses from several agents.
    - Step 43, Consolidated Summary: Collaborated with Agent_6 (with technical background) to refine the knowledge management concept. Added features for collaborative tagging and relevance feedback. Began developing simple demonstration to illustrate the concept.
    - Step 53, Consolidated Summary: Presented knowledge management system demonstration to community. Received positive feedback and constructive suggestions for improvement. Agent_3 and Agent_7 volunteered to help implement a working version.
    - Step 63, Consolidated Summary: Led working group with Agents 3, 6, and 7 to develop implementation plan. Assigned development tasks based on skills and interests. Established timeline and success metrics for initial deployment to address the information overload problem.
    """,
    overall_mood_trend="Increasingly enthusiastic and determined",
    agent_goals="Identify emerging community needs and develop creative solutions through collaboration",
    l2_summary="""
    As an Innovator, I successfully led the identification and solution development for a significant community-wide challenge of information overload across a 40-step period. Beginning with recognizing that agents were struggling to track developments across multiple projects, I progressed through systematic innovation stages: from initial brainstorming of filtering and summarization approaches, to developing a prototype concept for an adaptive knowledge management system, to collaborative refinement with Agent_6 that added collaborative tagging and relevance feedback mechanisms. The concept demonstration garnered positive community reception and volunteer support from Agents 3 and 7, culminating in the formation of a working group with clear implementation plans, assigned development responsibilities, and established metrics for success. This progression demonstrates effective innovation leadership that addresses an emergent community need through collaborative solution development, fulfilling my primary goal as an Innovator. The consistently rising enthusiasm throughout this process reflects successful engagement with a meaningful challenge and confidence in the developed solution.
    """,
).with_inputs("agent_role", "l1_summaries_context", "overall_mood_trend", "agent_goals")

# Compile all examples into a list for potential future optimization
l2_summary_examples = [example1, example2, example3, example4, example5]

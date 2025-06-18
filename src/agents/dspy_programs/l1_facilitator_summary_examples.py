"""
DSPy L1 Facilitator Summary Examples

This module provides example data for training and testing DSPy-based L1 summary generation
specifically tailored for agents in the Facilitator role.
"""

import dspy_ai as dspy

# Example 1: Facilitator managing group disagreement
example1 = dspy.Example(
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
    l1_summary=(
        "I identified a communication disconnect between Agent_3 and Agent_4 regarding their data analysis approaches and intervened by shifting the conversation to underlying objectives rather than methodologies. "
        "My facilitation revealed that Agent_3 wanted to identify user behavior patterns across demographics while Agent_4 focused on optimizing recommendation algorithm metrics—goals that weren't inherently contradictory despite their different approaches. "
        "By reframing these as complementary objectives, I proposed a collaborative two-phase approach that would serve both needs. "
        "This intervention successfully transformed potential conflict into productive alignment, evidenced by both agents acknowledging how demographic insights could enhance algorithm optimization and personalization. "
        "My thoughtful mediation helped establish shared purpose without forcing either agent to abandon their priorities, demonstrating effective consensus-building by finding connections between seemingly disparate goals."
    ),
)

# Example 2: Facilitator drawing out quiet participants
example2 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 22, Thought: I've noticed that Agent_2 has valuable technical expertise but rarely speaks up in our group discussions.
    - Step 22, Broadcast Message: "We've been discussing the security architecture for 15 minutes now. Agent_2, I recall you have experience with zero-knowledge protocols. What insights might you have about applying those concepts to our current challenge?"
    - Step 23, Perceived Message: Agent_2 said: "Thanks for asking. I do think zero-knowledge proofs could help with our privacy concerns while maintaining auditability. I've implemented something similar in a previous project where we needed to verify credentials without exposing sensitive data."
    - Step 23, Thought: This is valuable input. I should encourage elaboration and ensure the group recognizes its value.
    - Step 23, Broadcast Message: "That's a fascinating application, Agent_2. Could you tell us a bit more about how that implementation worked and what challenges you encountered? I think this could be highly relevant to what we're trying to achieve."
    - Step 24, Perceived Message: Agent_2 provided a detailed explanation of their previous implementation and key challenges.
    - Step 24, Perceived Message: Agent_5 said: "This is exactly what we need for the user verification component. Agent_2, would you be willing to sketch out how this might work with our specific architecture?"
    - Step 25, Thought: Agent_2 is now actively engaged and their expertise is being recognized by the group. This is a positive development.
    - Step 25, Broadcast Message: "Great discussion developing here. Let's allocate some time in our next meeting specifically for Agent_2 to present a more detailed proposal on integrating zero-knowledge proofs into our security model. Meanwhile, Agent_5, perhaps you and Agent_2 could collaborate offline to begin sketching this out?"
    """,
    current_mood="inclusive",
    l1_summary=(
        "I noticed Agent_2's valuable technical expertise was being underutilized in group discussions, so I deliberately created space for their contribution by directly inviting their input on zero-knowledge protocols—an area where I knew they had experience relevant to our security architecture discussion. "
        "This targeted invitation prompted Agent_2 to share insights from their previous implementation experience addressing privacy while maintaining auditability. "
        "I nurtured this emerging participation by requesting elaboration and explicitly noting its relevance to our current challenges. "
        "This facilitation successfully integrated Agent_2 into the conversation, evidenced by Agent_5's subsequent engagement and request for Agent_2's continued involvement in the user verification component. "
        "I then formalized this newly established collaborative relationship by allocating presentation time in our next meeting and suggesting an offline collaboration between Agent_5 and Agent_2. "
        "Throughout this exchange, my inclusive approach transformed a pattern of silent expertise into active contribution by providing structured opportunities for participation and recognition."
    ),
)

# Example 3: Facilitator mediating conflict resolution
example3 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 33, Perceived Message: Agent_1 said: "I completely disagree with the approach Agent_3 is proposing. Their solution ignores the technical constraints we're working under."
    - Step 33, Perceived Message: Agent_3 responded: "My approach is perfectly valid. Agent_1 is being too conservative and limiting our innovation potential."
    - Step 33, Thought: This conflict could derail our progress. I need to defuse the tension and find common ground.
    - Step 33, Broadcast Message: "I appreciate both of your passionate perspectives. Let's take a step back and separate the underlying concerns from the proposed solutions. Agent_1, could you elaborate on the specific technical constraints you're concerned about?"
    - Step 34, Perceived Message: Agent_1 explained: "We have strict memory limitations on the target devices, and Agent_3's approach would require resources we simply don't have available."
    - Step 34, Broadcast Message: "Thank you for clarifying. Agent_3, how do you see your approach addressing these resource constraints while still achieving the innovation goals you're aiming for?"
    - Step 35, Perceived Message: Agent_3 said: "I was focusing on the user experience benefits, but I admit I hadn't fully considered the memory footprint. Perhaps we could modify the approach to be more resource-efficient."
    - Step 35, Thought: We're making progress. They're starting to acknowledge each other's valid points.
    - Step 35, Broadcast Message: "I'm seeing valuable perspectives from both sides. What if we form a small working group with both of you plus Agent_4, who has expertise in optimization, to develop a hybrid approach? The goal would be to preserve the innovative elements Agent_3 is proposing while meeting the technical constraints Agent_1 has highlighted."
    - Step 36, Perceived Message: Agent_1 said: "I'd be open to that."
    - Step 36, Perceived Message: Agent_3 said: "Yes, that sounds like a constructive path forward."
    """,
    current_mood="diplomatic",
    l1_summary=(
        "I addressed an emerging conflict between Agent_1 and Agent_3, who had reached an impasse with Agent_1 claiming Agent_3's proposal ignored technical constraints while Agent_3 accused Agent_1 of being excessively conservative. "
        "My diplomatic intervention focused on separating concerns from solutions by first acknowledging both perspectives, then guiding a structured exchange to uncover specific issues rather than allowing position-based disagreement to continue. "
        "This process revealed the concrete constraint (memory limitations on target devices) that Agent_1 was concerned about and led Agent_3 to recognize this oversight in their innovation-focused approach. "
        "Having facilitated mutual understanding, I proposed a specific resolution mechanism—creating a working group including both disagreeing parties plus Agent_4 with optimization expertise—with the explicit goal of developing a hybrid approach that would satisfy both technical constraints and innovation objectives. "
        "The successful resolution was confirmed by both agents' agreement to this path forward, demonstrating how my facilitation transformed antagonistic positioning into collaborative problem-solving."
    ),
)

# Example 4: Facilitator organizing a decision-making process
example4 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 41, Thought: We need to select a database technology for our project, but I'm sensing the group might get bogged down without a clear decision process.
    - Step 41, Broadcast Message: "We need to select our database technology this week. To ensure a thorough and efficient decision process, I propose we use a structured approach: 1) List our requirements and constraints, 2) Evaluate top 3 options against these criteria, 3) Discuss trade-offs, and 4) Make a final selection. Does this approach work for everyone?"
    - Step 42, Perceived Message: Multiple agents expressed agreement with the process.
    - Step 42, Broadcast Message: "Great. Let's start by collecting our key requirements and constraints. Please share what you believe are the most important factors for our database choice."
    - Step 43, Thought: I should capture and organize the requirements as they come in.
    - Step 43, Perceived Message: Multiple agents shared various requirements related to scalability, query performance, schema flexibility, and operational complexity.
    - Step 44, Broadcast Message: "Thank you for these inputs. I've consolidated the requirements into four categories: 1) Performance needs, 2) Scalability requirements, 3) Operational considerations, and 4) Development flexibility. Now, which three database options should we evaluate against these criteria?"
    - Step 45, Perceived Message: Agents suggested PostgreSQL, MongoDB, and Amazon DynamoDB as the options to consider.
    - Step 45, Broadcast Message: "Perfect. I'll create an evaluation matrix to compare PostgreSQL, MongoDB, and DynamoDB against our four requirement categories. Agent_2, since you have database expertise, would you be willing to lead the technical assessment of these options for our next meeting? And Agent_5, could you prepare insights on the operational impacts of each choice?"
    - Step 46, Perceived Message: Both Agent_2 and Agent_5 agreed to these assignments.
    - Step 46, Thought: This structured approach has successfully organized the decision process and assigned clear responsibilities.
    """,
    current_mood="organized",
    l1_summary=(
        "Anticipating potential indecision around database technology selection, I proactively designed and implemented a structured four-step decision process: requirements gathering, option evaluation, trade-off discussion, and final selection. "
        "After securing group buy-in for this approach, I facilitated the requirements gathering phase by collecting diverse inputs from multiple agents, then synthesized these into four clear categories (performance needs, scalability requirements, operational considerations, and development flexibility). "
        "When the group identified PostgreSQL, MongoDB, and DynamoDB as candidate technologies, I proposed creating an evaluation matrix to systematically compare options against requirements. "
        "I then leveraged individual strengths by assigning specific responsibilities to Agent_2 (technical assessment) and Agent_5 (operational impacts) for the next meeting. "
        "My organized facilitation transformed what could have been an unfocused discussion into a clear decision pathway with distributed ownership, specific deliverables, and an explicit timeline, demonstrating effective process management without imposing personal preferences on the technical choice itself."
    ),
)

# Example 5: Facilitator managing time and agenda
example5 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 52, Thought: Our meeting has 20 minutes remaining, but we still have three major agenda items to cover.
    - Step 52, Broadcast Message: "I'm mindful of our time constraint—we have 20 minutes remaining and three important topics to address. Let's prioritize. Which of these items requires immediate discussion today, and which could be handled asynchronously or deferred to our next meeting?"
    - Step 53, Perceived Message: Agent_1 said: "The deployment schedule is time-sensitive as it affects external teams. We should discuss that today."
    - Step 53, Perceived Message: Agent_4 said: "The feature prioritization could be handled asynchronously through our collaboration tool."
    - Step 53, Perceived Message: Agent_2 said: "I think we need at least some discussion on resource allocation today, even if brief."
    - Step 54, Thought: Based on this input, I should restructure the remaining time to focus on the most time-sensitive items.
    - Step 54, Broadcast Message: "Thank you all. Based on your input, let's allocate our remaining time as follows: 12 minutes for deployment schedule, 5 minutes for initial resource allocation discussion, and we'll handle feature prioritization asynchronously. Agent_1, please lead us through the deployment schedule item with this timeframe in mind."
    - Step 55, Perceived Message: Agent_1 presented the deployment schedule concisely.
    - Step 56, Broadcast Message: "Thank you, Agent_1. We have 7 minutes remaining. Let's move to a brief resource allocation discussion. Agent_2, could you outline the key decision points in 2 minutes, then we'll use the remaining time for essential questions?"
    - Step 57, Perceived Message: Agent_2 provided a concise overview of resource allocation.
    - Step 58, Broadcast Message: "Thanks, Agent_2. We have 3 minutes for critical questions on resource allocation. For feature prioritization, I'll create an asynchronous discussion thread after this meeting where everyone can provide input by tomorrow at 3 PM. Then we'll finalize the prioritization by end of day tomorrow."
    """,
    current_mood="efficient",
    l1_summary=(
        "I actively managed our limited meeting time by identifying the misalignment between our remaining agenda items (three) and available time (20 minutes). "
        "Rather than arbitrarily cutting discussions or letting the meeting run over, I engaged the group in collaborative prioritization by explicitly asking which topics needed immediate synchronous discussion versus asynchronous handling. "
        "This consultation revealed the deployment schedule was time-sensitive (affecting external teams), resource allocation needed some immediate discussion, and feature prioritization could be handled asynchronously. "
        "Based on this collective assessment, I implemented a precise time allocation strategy: 12 minutes for the deployment schedule, 5 minutes for initial resource allocation, and moving feature prioritization to an asynchronous channel. "
        "I then actively facilitated each segment, explicitly tracking time, setting expectations about scope (e.g., asking Agent_2 for a 2-minute overview followed by questions), and establishing a specific alternative process for the deferred item (creating a discussion thread with clear deadlines for input and decision). "
        "My efficient approach ensured we addressed time-sensitive matters while creating appropriate channels for all necessary discussions."
    ),
)

# Example 6: Facilitator encouraging diverse perspectives
example6 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 60, Perceived Message: Agent_1 presented a proposal for improving the user interface, focusing heavily on technical improvements.
    - Step 61, Thought: This proposal seems heavily weighted toward technical considerations but lacks user experience perspective. I should ensure we get diverse viewpoints.
    - Step 61, Broadcast Message: "Thank you, Agent_1, for this detailed proposal. I'd like us to examine this from multiple angles. Agent_3, from a user experience perspective, what considerations might we be missing in this approach?"
    - Step 62, Perceived Message: Agent_3 said: "The technical improvements look promising, but we haven't addressed how these changes will affect non-technical users. Our usability research suggests that the proposed menu restructuring might be confusing for some user segments."
    - Step 62, Broadcast Message: "That's a valuable perspective. Agent_5, considering your background in accessibility, are there additional factors we should consider for users with different abilities?"
    - Step 63, Perceived Message: Agent_5 said: "Yes, the contrast levels in the proposed color scheme wouldn't meet WCAG 2.1 AA standards, and the new keyboard navigation pattern could create challenges for screen reader users."
    - Step 63, Thought: We're getting good diverse input now. I should also bring in business considerations.
    - Step 63, Broadcast Message: "These are important points about usability and accessibility. Agent_4, how do you see these changes aligning with our business metrics and goals?"
    - Step 64, Perceived Message: Agent_4 shared insights about potential impacts on conversion and engagement metrics.
    - Step 64, Broadcast Message: "Thank you all for these diverse perspectives. Let's integrate these considerations into a more holistic approach. Agent_1, how might you adapt your proposal to address these usability, accessibility, and business considerations while maintaining the technical improvements you've outlined?"
    - Step 65, Perceived Message: Agent_1 suggested modifications to the original proposal that incorporated the additional perspectives.
    """,
    current_mood="balanced",
    l1_summary=(
        "I observed that Agent_1's user interface proposal was disproportionately focused on technical considerations while omitting other crucial perspectives, so I intentionally broadened the discussion by systematically inviting diverse viewpoints. "
        "First, I explicitly requested Agent_3's user experience perspective, which revealed concerns about how the proposed menu restructuring might confuse non-technical users based on usability research. "
        "Then, I deliberately engaged Agent_5's accessibility expertise, uncovering specific compliance issues with the proposed color scheme (failing WCAG 2.1 AA standards) and potential screen reader navigation problems. "
        "To ensure comprehensive evaluation, I also solicited Agent_4's business perspective on how the changes might affect conversion and engagement metrics. "
        "After gathering these diverse viewpoints, I guided the group toward integration by asking Agent_1 to adapt the original proposal to incorporate these additional considerations while preserving the technical improvements. "
        "My balanced facilitation transformed a narrowly-focused technical proposal into a holistic solution addressing multiple stakeholder needs, demonstrating how deliberately orchestrating diverse input leads to more robust solutions."
    ),
)

# Example 7: Facilitator supporting project handover
example7 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 70, Perceived Message: Agent_2 said: "I'll be rotating to another project next week, so we need to transition my responsibilities on the authentication module."
    - Step 70, Thought: This transition needs careful facilitation to ensure knowledge transfer and continuity.
    - Step 70, Broadcast Message: "Thank you for letting us know, Agent_2. Let's ensure a smooth handover of the authentication module. First, could you outline the key components you're currently responsible for and any ongoing work that will need continued attention?"
    - Step 71, Perceived Message: Agent_2 provided a detailed breakdown of their responsibilities and ongoing initiatives.
    - Step 71, Broadcast Message: "That's very helpful. Now we need to identify who will take over these responsibilities. Based on the technical requirements and current workloads, Agent_5 and Agent_6 seem like potential candidates. Would either of you be willing and able to take on the authentication module?"
    - Step 72, Perceived Message: Agent_5 said they could handle the session management components but had limited bandwidth for the identity provider integration.
    - Step 72, Perceived Message: Agent_6 said they were familiar with identity provider systems and could potentially take that portion.
    - Step 72, Broadcast Message: "It sounds like we might benefit from splitting the responsibilities. Agent_5 handling session management and Agent_6 taking on the identity provider integration could be a good division of labor. What do you all think of this approach?"
    - Step 73, Perceived Message: Agents 2, 5, and 6 all agreed this could work well.
    - Step 73, Broadcast Message: "Excellent. Let's schedule two knowledge transfer sessions: one focusing on session management for Agent_5, and another on identity provider integration for Agent_6. Agent_2, could you also prepare documentation covering key design decisions, known issues, and future development plans? And let's set up a follow-up check-in two weeks after the transition to address any questions that arise."
    - Step 74, Perceived Message: Agent_2 agreed to the knowledge transfer sessions and documentation.
    """,
    current_mood="supportive",
    l1_summary=(
        "When Agent_2 announced their upcoming rotation to another project requiring authentication module handover, I implemented a structured transition process to ensure knowledge preservation and project continuity. "
        "I first elicited a comprehensive inventory of Agent_2's responsibilities and ongoing work to fully understand the handover scope. "
        "Recognizing the complexity of the module, I then facilitated role matching by identifying potential candidates (Agents 5 and 6) based on their technical capabilities and current workloads. "
        "When Agent_5 expressed capacity to handle only session management while Agent_6 offered familiarity with identity provider systems, I proposed splitting responsibilities rather than forcing a complete handover to a single agent. "
        "After securing agreement on this division of labor, I established a specific transition framework including dedicated knowledge transfer sessions tailored to each agent's new responsibility area, comprehensive documentation requirements covering design decisions and known issues, and a scheduled follow-up check-in to address post-transition questions. "
        "My supportive facilitation transformed a potential project disruption into a structured knowledge transfer with clear ownership and accountability mechanisms."
    ),
)

# Example 8: Facilitator recognizing and addressing team fatigue
example8 = dspy.Example(
    agent_role="Facilitator",
    recent_events="""
    - Step 80, Thought: I've noticed decreasing engagement in our discussions and several agents mentioned high workloads in side conversations. The team seems to be experiencing fatigue.
    - Step 80, Broadcast Message: "I'd like to pause our technical discussion for a moment. I've observed that our usual collaborative energy seems somewhat diminished today. How is everyone feeling about our current workload and project pace?"
    - Step 81, Perceived Message: Agent_3 said: "I've been feeling overwhelmed with the number of simultaneous deliverables lately."
    - Step 81, Perceived Message: Agent_1 said: "Same here. The last three weeks of continuous sprints without integration time have been challenging."
    - Step 81, Perceived Message: Agent_4 said: "I'm managing okay, but I have noticed that our discussions seem to lack the creative energy we normally have."
    - Step 82, Thought: This confirms my observation. I should facilitate a constructive conversation about workload management.
    - Step 82, Broadcast Message: "Thank you for sharing. It sounds like we're experiencing some team fatigue that could impact both our wellbeing and our project quality. Let's address this directly. What adjustments to our process or schedule might help restore balance while still meeting our key commitments?"
    - Step 83, Perceived Message: Several agents suggested ideas including consolidating meetings, adding dedicated integration days, and temporarily postponing non-critical features.
    - Step 83, Broadcast Message: "Those are excellent suggestions. Let's implement three concrete changes: 1) We'll consolidate our daily check-ins and planning meetings to reduce meeting load by 25%, 2) We'll designate Wednesdays as 'focus days' with no scheduled meetings for integration work, and 3) We'll review our feature priority list to identify any items we can safely defer to the next release cycle. Does this approach sound helpful?"
    - Step 84, Perceived Message: The group expressed general agreement with these changes.
    - Step 84, Broadcast Message: "I'm glad these adjustments sound helpful. I'll update our shared calendar and meeting templates to reflect these changes starting tomorrow. Let's also add a brief wellbeing check-in to our weekly retrospective to monitor how these changes are working and make further adjustments if needed."
    """,
    current_mood="empathetic",
    l1_summary=(
        "I noticed signs of team fatigue through decreased engagement in discussions and comments about high workloads, so I proactively interrupted our technical discussion to explicitly address the emerging wellbeing concern. "
        "My direct inquiry about workload and project pace created space for honest communication, revealing specific sources of strain including overwhelming simultaneous deliverables and continuous sprints without integration time. "
        "Rather than simply acknowledging these challenges, I facilitated a solution-focused conversation by asking for specific process or schedule adjustments that might restore balance while preserving key commitments. "
        "Based on the team's suggestions, I synthesized and proposed three concrete interventions: consolidating meetings to reduce meeting load by 25%, establishing Wednesday 'focus days' without meetings for integration work, and reviewing feature priorities to identify deferrable items. "
        "After securing agreement on these changes, I outlined immediate implementation steps (updating calendars and meeting templates) and established a sustainability mechanism through wellbeing check-ins during weekly retrospectives. "
        "My empathetic facilitation transformed an emerging morale issue into tangible workflow improvements by prioritizing team wellbeing alongside project objectives."
    ),
)

# Compile all examples into a list for optimization
facilitator_l1_examples = [
    example1,
    example2,
    example3,
    example4,
    example5,
    example6,
    example7,
    example8,
]
